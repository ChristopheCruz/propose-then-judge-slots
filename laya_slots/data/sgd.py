"""Chargeur SGD, Schema-Guided Dialogue (Rastogi et al., 2020), vers le format unifié.

Accès et licence
----------------
Librement téléchargeable, licence CC BY-SA 4.0 : dépôt
https://github.com/google-research-datasets/dstc8-schema-guided-dialogue (archive de la branche
`master`). Décompresser sous `data/raw/sgd/` ; le chargeur cherche `train/`, `dev/`, `test/`, chacun
avec `schema.json` et `dialogues_*.json`. `download()` le fait (réseau requis). Anglais seulement.
Effectifs publiés : 16 142 / 2 482 / 4 201 dialogues ; 20 domaines, 45 services ; le dev et le test
contiennent des services absents du train (généralisation à des schémas inconnus).

Format source
-------------
Chaque dialogue : `dialogue_id`, `services`, `turns` ; chaque tour : `speaker` (USER, SYSTEM),
`utterance`, `frames` (un par service actif) ; une frame : `service`, `slots` (spans `start`,
`exclusive_end` en offsets caractères de l'énoncé du tour), `actions` et, pour les tours USER, `state`
(`active_intent`, `requested_slots`, `slot_values` : slot -> liste de valeurs acceptables, cumulé).
`schema.json` décrit chaque service : intentions (slots requis et optionnels) et slots (description en
langue naturelle, catégoriel ou non, valeurs possibles).

Choix de conversion
-------------------
- Un exemple par (tour USER, frame) : `domain` = service, `intent` = « service.intention active »
  (None si l'intention active est NONE). `history` = tous les tours précédents ; `state_before` = état
  cumulé (tous services) après le tour USER précédent.
- Noms de slots : « service-slot » (ex. « Restaurants_1-city »), car un même nom de slot change de sens
  d'un service à l'autre. Les intentions de l'ontologie portent leurs slots requis et optionnels.
- Slots du tour = DIFFÉRENCE d'état du service (comme pour Multi3WOZ). Valeur gold : la valeur dont la
  forme correspond au span de l'énoncé, sinon la première valeur acceptable ; les autres formes sont
  dans `meta["alternatives"]`. Toutes les valeurs passent par `normalize_value`.
- `origin` : « utterance » si la frame USER porte un span pour ce slot ; « history » si la valeur
  apparaît dans un tour précédent (offre du système acceptée) ; sinon « context ».
- Ontologie : union des schémas des trois splits (les services non vus du test en font partie) ;
  descriptions des slots et intentions tirées des schémas ; valeurs possibles des slots catégoriels.
"""
from __future__ import annotations

import json
import urllib.request
import zipfile
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from ..interfaces import DatasetLoader
from ..normalize import normalize_value
from ..schema import Example, IntentSpec, Ontology, SlotSpec, SlotValue, Split, Turn
from .common import SPLITS, raw_dir_for

URL = "https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/archive/refs/heads/master.zip"
_SPLIT_DIRS = {"train": "train", "dev": "dev", "test": "test"}


def slot_name(service: str, slot: str) -> str:
    return f"{service}-{slot}"


class SgdLoader(DatasetLoader):
    name = "sgd"
    source_version = "SGD (dstc8-schema-guided-dialogue, master)"

    def __init__(self, raw_dir: str | Path | None = None, langs: list[str] | None = None):
        self.raw_dir = raw_dir_for(self.name, raw_dir)
        self._cache: dict[str, list[Example]] = {}
        self.skipped: dict[tuple[str, str], int] = defaultdict(int)

    def languages(self) -> list[str]:
        return ["en"]

    def download(self) -> Path:
        """Télécharge et décompresse l'archive du dépôt (réseau requis)."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        archive = self.raw_dir / "sgd-master.zip"
        if not archive.exists():
            urllib.request.urlretrieve(URL, archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(self.raw_dir)
        return archive

    def _split_dir(self, split: Split) -> Path:
        cands = [p for p in self.raw_dir.rglob(_SPLIT_DIRS[split]) if p.is_dir() and (p / "schema.json").exists()]
        if not cands:
            raise FileNotFoundError(f"SGD : dossier {split}/ avec schema.json introuvable sous {self.raw_dir}")
        return min(cands, key=lambda p: (len(p.parts), str(p)))   # pas sgd_x/ (variantes sans dialogues)

    def source_files(self, lang: str) -> list[Path]:
        return [f for s in SPLITS for f in sorted(self._split_dir(s).glob("*.json"))]

    def _schemas(self) -> list[dict]:
        services: dict[str, dict] = {}
        for split in SPLITS:
            for svc in json.loads((self._split_dir(split) / "schema.json").read_text(encoding="utf-8")):
                services.setdefault(svc["service_name"], svc)
        return [services[k] for k in sorted(services)]

    def _read(self, split: Split) -> list[Example]:
        if split not in self._cache:
            out = []
            for f in sorted(self._split_dir(split).glob("dialogues_*.json")):
                for dialogue in json.loads(f.read_text(encoding="utf-8")):
                    out.extend(self.convert_dialogue(dialogue, split))
            self._cache[split] = out
        return self._cache[split]

    def convert_dialogue(self, dialogue: dict, split: Split) -> list[Example]:
        dial_id = dialogue["dialogue_id"]
        examples, history = [], []
        state: dict[str, str] = {}                       # état cumulé, tous services, noms préfixés
        for i, turn in enumerate(dialogue.get("turns", [])):
            text = turn.get("utterance", "")
            if turn.get("speaker", "").upper() != "USER":
                history.append(Turn("system", text))
                continue
            before = dict(state)
            hist_text = " ".join(t.text for t in history).lower()
            for frame in turn.get("frames", []):
                ex, new_values = self._frame_example(dial_id, i, text, frame, before, history, hist_text, split)
                state.update(new_values)
                examples.append(ex)
            history.append(Turn("user", text))
        return examples

    def _frame_example(self, dial_id, i, text, frame, before, history, hist_text, split):
        service = frame["service"]
        st = frame.get("state") or {}
        spans = {s["slot"]: (int(s["start"]), int(s["exclusive_end"])) for s in frame.get("slots", [])
                 if 0 <= int(s["start"]) < int(s["exclusive_end"]) <= len(text)}
        slots, new_values, alternatives = [], {}, {}
        for slot, values in sorted((st.get("slot_values") or {}).items()):
            if not values:
                continue
            name = slot_name(service, slot)
            span = spans.get(slot)
            surface = text[span[0]:span[1]] if span else None
            norm = [normalize_value(name, v, "en") for v in values]
            chosen = norm[0]
            if surface is not None:
                s_norm = normalize_value(name, surface, "en")
                chosen = s_norm if s_norm in norm else norm[0]
            new_values[name] = chosen
            if before.get(name) in norm:                  # valeur déjà dans l'état : pas un slot du tour
                continue
            if span is not None:
                origin = "utterance"
            elif any(v.lower() in hist_text for v in values):
                origin = "history"
            else:
                origin = "context"
            alternatives[name] = [v for v in norm if v != chosen]
            slots.append(SlotValue(name=name, value=chosen, surface=surface, span=span, origin=origin))
        intent = st.get("active_intent")
        ex = Example(
            id=f"{self.name}:en:{split}:{dial_id}:{i}:{service}", dataset="sgd", lang="en", split=split,
            utterance=text, intent=None if not intent or intent == "NONE" else f"{service}.{intent}",
            domain=service, slots=slots, history=list(history), state_before=dict(before),
            meta={"dialogue_id": dial_id, "turn_id": i, "service": service, "alternatives": alternatives,
                  "requested_slots": st.get("requested_slots", [])},
        )
        return ex, new_values

    def load(self, lang: str, split: Split) -> Iterator[Example]:
        if lang != "en":
            raise FileNotFoundError("SGD n'existe qu'en anglais")
        yield from self._read(split)

    def ontology(self, lang: str) -> Ontology:
        intents, slots = {}, {}
        for svc in self._schemas():
            service = svc["service_name"]
            for s in svc.get("slots", []):
                name = slot_name(service, s["name"])
                values = s.get("possible_values") if s.get("is_categorical") else None
                slots[name] = SlotSpec(name=name, description=s.get("description") or s["name"],
                                       values=sorted({normalize_value(name, v, "en") for v in values}) if values else None)
            for it in svc.get("intents", []):
                iname = f"{service}.{it['name']}"
                allowed = list(it.get("required_slots", [])) + list((it.get("optional_slots") or {}).keys())
                intents[iname] = IntentSpec(name=iname, description=it.get("description") or it["name"],
                                            domain=service, slots=[slot_name(service, s) for s in allowed])
        return Ontology(dataset="sgd", lang="en", intents=intents, slots=slots)
