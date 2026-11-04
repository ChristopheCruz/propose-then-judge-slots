"""Chargeur Multi3WOZ (Hu et al., 2023) vers le format unifié, au niveau du tour utilisateur.

Accès et licence
----------------
Librement téléchargeable, licence MIT (dépôt https://github.com/cambridgeltl/multi3woz, fichier
`data.zip`, 90,6 Mo, version corrigée du 15/12/2023). Décompresser sous `data/raw/multi3woz/` ; le
chargeur cherche `<Langue>/data.json`, `valListFile.txt`, `testListFile.txt` et `slot_descriptions.json`
(Langue = English, French, Arabic, Turkish). `download()` le fait (réseau requis).
Effectifs publiés : 9 160 dialogues par langue, 7 440 / 860 / 860 (train / dev / test). Les listes
`valListFile.txt` et `testListFile.txt` reprennent celles de MultiWOZ (1 000 noms chacune) ; seuls 860
de chaque figurent dans `data.json`. Tout dialogue hors de ces listes est en train.

Format source
-------------
Format MultiWOZ 2.x : `data.json` = {id_dialogue: {"goal", "log"}} ; `log` alterne tours USER et SYSTEM.
L'état de dialogue (`metadata`, domaine -> {"semi", "book"}) est porté par les tours SYSTEM : c'est
l'état APRÈS le tour utilisateur qui précède. Les tours USER ont `dialog_act` et `span_info`
([acte, slot, valeur, début, fin] en offsets caractères du texte localisé). Les valeurs d'état sont
canoniques : en français, les catégoriels restent en anglais (« friday », « college ») et les entités
sont localisées (« le Bambou Royal »).

Choix de conversion
-------------------
- Un exemple par tour USER. `history` = tous les tours précédents (USER et SYSTEM, texte localisé),
  sans troncature (la fenêtre d'historique est un choix d'expérience). `state_before` = état cumulé
  après le tour utilisateur précédent.
- Slots du tour = DIFFÉRENCE d'état : slots dont la valeur après le tour diffère de `state_before`.
  C'est la définition cohérente avec la Joint Goal Accuracy (état_t = état_{t-1} + slots du tour).
  Les slots qui disparaissent de l'état (rare) ne sont pas des slots du tour ; ils sont listés dans
  `meta["removed"]`.
- Domaines : les 5 domaines usuels du suivi d'état sur MultiWOZ (hotel, train, restaurant, attraction,
  taxi ; 30 slots), comme TRADE/T5DST. bus, police, hospital sont ignorés.
- Noms de slots : « domaine-slot » pour `semi`, « domaine-book slot » pour `book`, en minuscules,
  comme dans `slot_descriptions.json` (ex. « hotel-book stay », « train-leaveat »).
- Valeurs : « not mentioned », « none » et chaîne vide = slot absent ; « dontcare » est une valeur
  explicite. Toutes les valeurs passent par `normalize_value`.
- Spans : pour chaque slot du tour, on cherche dans `span_info` du tour une annotation de même slot
  (domaine identique ou « Booking »), en préférant la même valeur normalisée ; à défaut, la valeur
  est cherchée telle quelle dans l'énoncé. `surface` = texte du span dans l'énoncé (peut différer de
  `value` : « vendredi » pour « friday »). Sans span : `origin="utterance"` si un acte du tour porte ce
  slot (ex. dontcare), `origin="history"` si la valeur apparaît dans un tour précédent (offre du
  système acceptée), sinon `origin="context"`. La provenance est dans `meta["span_source"]`.
- `Example.intent` = None (pas d'intention par tour dans le jeu) ; `domain` = domaine majoritaire des
  slots du tour, sinon celui des actes du tour. L'ontologie a une « intention » par domaine, qui
  regroupe ses slots. Descriptions : `slot_descriptions.json` (en anglais dans toutes les langues).
"""
from __future__ import annotations

import json
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

from ..interfaces import DatasetLoader
from ..normalize import normalize_value
from ..schema import Example, IntentSpec, Ontology, SlotSpec, SlotValue, Split, Turn
from .common import SPLITS, find_file, humanize, raw_dir_for

URL = "https://raw.githubusercontent.com/cambridgeltl/multi3woz/main/data.zip"
LANG_DIRS = {"en": "English", "fr": "French", "ar": "Arabic", "tr": "Turkish"}
DOMAINS = ("hotel", "train", "restaurant", "attraction", "taxi")
ABSENT = {"", "not mentioned", "none", "not given"}
DONTCARE = "dontcare"

# Nom de slot des actes de dialogue -> nom de base dans l'état.
ACT_SLOT_MAP = {"name": "name", "area": "area", "price": "pricerange", "type": "type", "food": "food",
                "day": "day", "time": "time", "people": "people", "stay": "stay", "stars": "stars",
                "internet": "internet", "parking": "parking", "dest": "destination",
                "depart": "departure", "leave": "leaveat", "arrive": "arriveby"}

DOMAIN_DESC = {"hotel": "find or book a hotel", "train": "find or book a train",
               "restaurant": "find or book a restaurant", "attraction": "find an attraction",
               "taxi": "book a taxi"}

# Slots dont la liste de valeurs (relevée sur le train) est publiée dans l'ontologie.
CATEGORICAL_BASES = {"area", "pricerange", "parking", "internet", "stars", "day", "people", "stay"}
CATEGORICAL_SLOTS = {"hotel-type", "attraction-type"}


def slot_base(name: str) -> str:
    """« hotel-book stay » -> « stay », « train-leaveat » -> « leaveat »."""
    return name.split("-", 1)[1].removeprefix("book ")


def flatten_state(metadata: dict, lang: str) -> dict[str, str]:
    """État MultiWOZ (domaine -> semi/book) -> {slot: valeur normalisée}, valeurs absentes retirées."""
    state = {}
    for dom in DOMAINS:
        parts = metadata.get(dom) or {}
        for part, prefix in (("semi", ""), ("book", "book ")):
            for s, v in (parts.get(part) or {}).items():
                if s == "booked" or not isinstance(v, str):
                    continue
                if v.strip().lower() in ABSENT:
                    continue
                name = f"{dom}-{prefix}{s.lower()}"
                state[name] = normalize_value(name, v, lang)
    return state


def turn_speaker(turn: dict, index: int) -> str:
    """Locuteur d'un tour : champ « speaker » s'il existe (versions AR/FR/TR), sinon parité de
    l'indice comme dans MultiWOZ (la version anglaise n'a pas ce champ)."""
    sp = str(turn.get("speaker", "")).upper()
    if sp in ("USER", "SYSTEM"):
        return sp.lower()
    return "user" if index % 2 == 0 else "system"


class Multi3WozLoader(DatasetLoader):
    name = "multi3woz"
    source_version = "Multi3WOZ data.zip (corrected release 2023-12-15)"

    def __init__(self, raw_dir: str | Path | None = None, langs: list[str] | None = None):
        self.raw_dir = raw_dir_for(self.name, raw_dir)
        self._langs = langs or ["en", "fr"]
        self._dialogues: dict[str, dict] = {}
        self._cache: dict[tuple[str, str], list[Example]] = {}

    def languages(self) -> list[str]:
        return list(self._langs)

    def download(self) -> Path:
        """Télécharge et décompresse `data.zip` du dépôt officiel (réseau requis, ~90 Mo)."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        archive = self.raw_dir / "data.zip"
        if not archive.exists() or archive.stat().st_size == 0:
            urllib.request.urlretrieve(URL, archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(self.raw_dir)
        return archive

    def _file(self, lang: str, fname: str) -> Path:
        return find_file(self.raw_dir, f"{LANG_DIRS[lang]}/{fname}")

    def source_files(self, lang: str) -> list[Path]:
        return [self._file(lang, f) for f in ("data.json", "valListFile.txt", "testListFile.txt")]

    def _split_of(self, lang: str) -> dict[str, Split]:
        out: dict[str, Split] = {}
        for split, fname in (("dev", "valListFile.txt"), ("test", "testListFile.txt")):
            for line in self._file(lang, fname).read_text(encoding="utf-8").split():
                out[line.strip().removesuffix(".json")] = split
        return out

    def _raw(self, lang: str) -> dict:
        if lang not in self._dialogues:
            with self._file(lang, "data.json").open(encoding="utf-8") as f:
                self._dialogues[lang] = json.load(f)
        return self._dialogues[lang]

    def _read(self, lang: str, split: Split) -> list[Example]:
        if (lang, split) not in self._cache:
            split_of = self._split_of(lang)
            by_split: dict[str, list[Example]] = {s: [] for s in SPLITS}
            for key in sorted(self._raw(lang)):
                dial_id = key.removesuffix(".json")
                s = split_of.get(dial_id, "train")
                by_split[s].extend(self.convert_dialogue(dial_id, self._raw(lang)[key], lang, s))
            for s in SPLITS:
                self._cache[(lang, s)] = by_split[s]
            self._dialogues.pop(lang, None)          # libère la mémoire : tout est converti
        return self._cache[(lang, split)]

    # ------------------------------------------------------------------
    def convert_dialogue(self, dial_id: str, dialogue: dict, lang: str, split: Split) -> list[Example]:
        log = dialogue.get("log", [])
        examples, history = [], []
        before: dict[str, str] = {}
        for i, turn in enumerate(log):
            speaker = turn_speaker(turn, i)
            text = turn.get("text", "")
            if speaker == "system":
                history.append(Turn("system", text))
                if turn.get("metadata"):
                    before = flatten_state(turn["metadata"], lang)
                continue
            nxt = log[i + 1] if i + 1 < len(log) else None
            after = before
            if nxt is not None and turn_speaker(nxt, i + 1) == "system" and nxt.get("metadata"):
                after = flatten_state(nxt["metadata"], lang)
            examples.append(self._turn_example(dial_id, turn, i, text, before, after, history, lang, split))
            history.append(Turn("user", text))
        return examples

    def _turn_example(self, dial_id, turn, i, text, before, after, history, lang, split) -> Example:
        acts = turn.get("dialog_act") or {}
        spans = turn.get("span_info") or []
        act_slots = {(a.split("-")[0].lower(), ACT_SLOT_MAP.get(str(sv[0]).lower()))
                     for a, svs in acts.items() for sv in (svs or []) if sv}
        hist_text = " ".join(t.text for t in history).lower()
        slots, sources = [], {}
        for name in sorted(after):
            value = after[name]
            if before.get(name) == value:
                continue
            dom, base = name.split("-", 1)[0], slot_base(name)
            span, src = self._find_span(text, spans, dom, base, name, value, lang)
            if span is not None:
                origin = "utterance"
            elif (dom, base) in act_slots or ("booking", base) in act_slots:
                origin, src = "utterance", "act"
            elif value.lower() in hist_text:
                origin, src = "history", "history"
            else:
                origin, src = "context", "none"
            sources[name] = src
            slots.append(SlotValue(name=name, value=value, surface=text[span[0]:span[1]] if span else None,
                                   span=span, origin=origin))
        doms = Counter(s.name.split("-", 1)[0] for s in slots)
        if doms:
            domain = doms.most_common(1)[0][0]
        else:
            act_doms = [a.split("-")[0].lower() for a in acts]
            domain = next((d for d in act_doms if d in DOMAINS), None)
        return Example(
            id=f"{self.name}:{lang}:{split}:{dial_id}:{turn.get('turn_id', i)}", dataset="multi3woz",
            lang=lang, split=split, utterance=text, intent=None, domain=domain, slots=slots,
            history=list(history), state_before=dict(before),
            meta={"dialogue_id": dial_id, "turn_id": turn.get("turn_id", i), "dialog_act": acts,
                  "span_source": sources, "removed": sorted(set(before) - set(after))},
        )

    @staticmethod
    def _find_span(text, spans, dom, base, name, value, lang) -> tuple[tuple[int, int] | None, str]:
        matches = []
        for sp in spans:
            if len(sp) < 5:
                continue
            act, sname, sval, start, end = sp[:5]
            act_dom = str(act).split("-")[0].lower()
            if ACT_SLOT_MAP.get(str(sname).lower()) != base or act_dom not in (dom, "booking"):
                continue
            try:
                start, end = int(start), int(end)
            except (TypeError, ValueError):
                continue
            if 0 <= start < end <= len(text):
                same_value = normalize_value(name, str(sval), lang) == value
                matches.append((not same_value, act_dom != dom, start, end))
        if matches:
            _, _, start, end = min(matches)
            return (start, end), "span_info"
        m = re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text, re.IGNORECASE) \
            if value != DONTCARE else None
        if m:
            return (m.start(), m.end()), "string_match"
        return None, ""

    def load(self, lang: str, split: Split) -> Iterator[Example]:
        yield from self._read(lang, split)

    def ontology(self, lang: str) -> Ontology:
        try:
            desc = json.loads(self._file(lang, "slot_descriptions.json").read_text(encoding="utf-8"))
            desc = {k.lower(): v for k, v in desc.items()}
        except FileNotFoundError:
            desc = {}
        slot_names: set[str] = set()
        train_values: dict[str, set[str]] = defaultdict(set)
        for split in SPLITS:
            for ex in self._read(lang, split):
                names = {s.name for s in ex.slots} | set(ex.state_before)
                slot_names |= names
                if split == "train":
                    for s in ex.slots:
                        train_values[s.name].add(s.value)
        slots = {}
        for n in sorted(slot_names):
            d = desc.get(n)
            closed = slot_base(n) in CATEGORICAL_BASES or n in CATEGORICAL_SLOTS
            slots[n] = SlotSpec(name=n, description=d[0] if d else humanize(n),
                                values=sorted(train_values[n]) if closed and train_values[n] else None)
        intents = {d: IntentSpec(name=d, description=DOMAIN_DESC[d], domain=d,
                                 slots=sorted(n for n in slots if n.startswith(d + "-")))
                   for d in DOMAINS}
        return Ontology(dataset="multi3woz", lang=lang, intents=intents, slots=slots)
