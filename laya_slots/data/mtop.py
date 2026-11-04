"""Chargeur MTOP (Li et al., 2021) vers le format unifié.

Accès et licence
----------------
Librement téléchargeable, licence CC BY-SA 4.0 : https://dl.fbaipublicfiles.com/mtop/mtop.zip
(8,6 Mo, version v2 du 25/01/2021). Décompresser sous `data/raw/mtop/` ; le chargeur cherche
`<lang>/train.txt`, `<lang>/eval.txt`, `<lang>/test.txt`. `download()` le fait (réseau requis).
Effectifs publiés : EN 15 667 / 2 235 / 4 386, FR 11 814 / 1 577 / 3 193.

Format source (mtop-README.txt)
-------------------------------
TSV sans en-tête, 8 colonnes : id, intention de premier niveau, slots plats
(`début:fin:SL:NOM,...` en offsets d'OCTETS UTF-8), énoncé, domaine, locale, forme découplée
(`[IN:X [SL:Y mots ] ... ]`), jetons JSON.

Choix de conversion
-------------------
- Slots aplatis au premier niveau : on prend la colonne 3, qui ne contient que les slots de premier
  niveau ; les offsets octets sont convertis en offsets caractères. Un slot imbriqué (ex. SL:CONTACT
  contenant [IN:GET_CONTACT [SL:TYPE_RELATION mom ] ]) devient UN slot de premier niveau dont la valeur
  est tout le texte couvert (« mom »). Son sous-arbre est conservé dans `meta["nested"]`
  (liste de {"slot", "surface", "tree"}) et `meta["has_nested"]` permet l'analyse séparée prévue au plan.
  La forme découplée complète est dans `meta["decoupled"]`.
- `Example.intent` = intention de premier niveau (ex. « IN:GET_WEATHER »), `domain` = colonne 5.
  Les noms officiels (préfixes IN:/SL: compris) sont gardés pour que l'exactitude exacte et le F1
  officiels se calculent sans table de correspondance.
- Ontologie : intentions et slots observés dans les trois splits ; `IntentSpec.slots` = slots de premier
  niveau vus avec l'intention. Descriptions de slots rédigées à la main (en anglais, pour EN comme
  pour FR) ; descriptions d'intentions dérivées du nom (« IN:GET_WEATHER » -> « get weather »).
  Valeurs fermées pour quelques slots catégoriels, relevées sur le train uniquement.
- L'analyse de la colonne 3 se fait par expression régulière (une ligne FR contient une virgule
  pleine chasse « ， » comme séparateur).
"""
from __future__ import annotations

import re
import urllib.request
import zipfile
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from ..interfaces import DatasetLoader
from ..normalize import normalize_value
from ..schema import Example, IntentSpec, Ontology, SlotSpec, SlotValue, Split
from .common import SPLITS, byte_to_char_offsets, find_file, humanize, raw_dir_for

URL = "https://dl.fbaipublicfiles.com/mtop/mtop.zip"
_SPLIT_FILES = {"train": "train.txt", "dev": "eval.txt", "test": "test.txt"}
_SLOT_RE = re.compile(r"(\d+):(\d+):(SL:[A-Z_]+)")

SLOT_DESC = {
    "SL:AGE": "age of a person",
    "SL:ALARM_NAME": "name of an alarm",
    "SL:AMOUNT": "quantity or number of items",
    "SL:ATTENDEE": "person attending an event",
    "SL:ATTENDEE_EVENT": "attendee of an event",
    "SL:ATTRIBUTE_EVENT": "attribute of an event (free, kid-friendly...)",
    "SL:CATEGORY_EVENT": "category of event (concert, festival...)",
    "SL:CONTACT": "person to contact",
    "SL:CONTACT_ADDED": "person to add to a call or group",
    "SL:CONTACT_METHOD": "contact method (phone, email...)",
    "SL:CONTACT_RELATED": "person the contact is related to",
    "SL:CONTACT_REMOVED": "person to remove from a call or group",
    "SL:CONTENT_EXACT": "exact content of a message",
    "SL:DATE_TIME": "date or time",
    "SL:EDUCATION_DEGREE": "education degree",
    "SL:EMPLOYER": "employer or company",
    "SL:GENDER": "gender",
    "SL:GROUP": "group of people or chat group",
    "SL:JOB": "job or occupation",
    "SL:LIFE_EVENT": "life event (wedding, birthday...)",
    "SL:LOCATION": "place or location",
    "SL:MAJOR": "field of study",
    "SL:METHOD_RECIPES": "cooking or preparation method",
    "SL:METHOD_RETRIEVAL_REMINDER": "how to retrieve a reminder",
    "SL:METHOD_TIMER": "kind of timer (timer, stopwatch)",
    "SL:MUSIC_ALBUM_MODIFIER": "album modifier (new, latest...)",
    "SL:MUSIC_ALBUM_TITLE": "album title",
    "SL:MUSIC_ARTIST_NAME": "music artist",
    "SL:MUSIC_GENRE": "music genre",
    "SL:MUSIC_PLAYLIST_MODIFIER": "playlist modifier",
    "SL:MUSIC_PLAYLIST_TITLE": "playlist title",
    "SL:MUSIC_PROVIDER_NAME": "music service (Spotify, Pandora...)",
    "SL:MUSIC_RADIO_ID": "radio station",
    "SL:MUSIC_REWIND_TIME": "time to rewind or fast-forward",
    "SL:MUSIC_TRACK_TITLE": "song title",
    "SL:MUSIC_TYPE": "kind of audio (music, song, album, playlist...)",
    "SL:NAME_APP": "name of an application",
    "SL:NEWS_CATEGORY": "news category",
    "SL:NEWS_REFERENCE": "reference to a news item (that, this story...)",
    "SL:NEWS_SOURCE": "news source or outlet",
    "SL:NEWS_TOPIC": "news topic",
    "SL:NEWS_TYPE": "kind of news content (news, article, headlines...)",
    "SL:ORDINAL": "position in a list (next, first, last...)",
    "SL:PERIOD": "time period or duration",
    "SL:PERSON_REMINDED": "person to remind",
    "SL:PHONE_NUMBER": "phone number",
    "SL:RECIPES_ATTRIBUTE": "recipe attribute (calories, time, ingredients...)",
    "SL:RECIPES_COOKING_METHOD": "cooking method (bake, grill...)",
    "SL:RECIPES_CUISINE": "cuisine",
    "SL:RECIPES_DIET": "diet (vegan, gluten free...)",
    "SL:RECIPES_DISH": "dish",
    "SL:RECIPES_EXCLUDED_INGREDIENT": "ingredient to exclude",
    "SL:RECIPES_INCLUDED_INGREDIENT": "ingredient to include",
    "SL:RECIPES_MEAL": "meal (breakfast, dinner...)",
    "SL:RECIPES_QUALIFIER_NUTRITION": "nutrition qualifier (low, high...)",
    "SL:RECIPES_RATING": "recipe rating",
    "SL:RECIPES_SOURCE": "recipe source",
    "SL:RECIPES_TIME_PREPARATION": "preparation time",
    "SL:RECIPES_TYPE": "type of recipe or dish",
    "SL:RECIPES_TYPE_NUTRITION": "nutrient (protein, sugar...)",
    "SL:RECIPES_UNIT_MEASUREMENT": "unit of measurement",
    "SL:RECIPES_UNIT_NUTRITION": "nutrition unit (calories, grams...)",
    "SL:RECIPIENT": "recipient of a message or call",
    "SL:SCHOOL": "school or university",
    "SL:SENDER": "sender of a message",
    "SL:SIMILARITY": "similarity constraint (like, similar to...)",
    "SL:TIMER_NAME": "name of a timer",
    "SL:TITLE_EVENT": "title of an event",
    "SL:TODO": "task to be reminded of",
    "SL:TYPE_CONTACT": "type of contact",
    "SL:TYPE_CONTENT": "type of message content (video, audio, text...)",
    "SL:TYPE_RELATION": "relationship to the user (mom, boss...)",
    "SL:USER_ATTENDEE_EVENT": "the user as attendee of an event",
    "SL:WEATHER_ATTRIBUTE": "weather attribute (rain, sunny, temperature...)",
    "SL:WEATHER_TEMPERATURE_UNIT": "temperature unit (celsius, fahrenheit)",
}

# Slots à valeurs fermées (liste relevée sur le train).
CATEGORICAL_SLOTS = {"SL:METHOD_TIMER", "SL:TYPE_CONTENT", "SL:MUSIC_TYPE", "SL:NEWS_TYPE",
                     "SL:WEATHER_TEMPERATURE_UNIT"}


# ---------------------------------------------------------------------------
# Forme découplée
# ---------------------------------------------------------------------------
def parse_decoupled(form: str) -> dict:
    """Analyse « [IN:X [SL:Y a b ] ] » en arbre {"label", "children"} ; les feuilles sont des chaînes."""
    stack: list[dict] = [{"label": None, "children": []}]
    for tok in form.split():
        if tok.startswith("[") and len(tok) > 1:
            node = {"label": tok[1:], "children": []}
            stack[-1]["children"].append(node)
            stack.append(node)
        elif tok == "]" and len(stack) > 1:
            stack.pop()
        else:
            stack[-1]["children"].append(tok)
    root = stack[0]["children"]
    return root[0] if len(root) == 1 and isinstance(root[0], dict) else {"label": None, "children": root}


def tree_to_str(node) -> str:
    if isinstance(node, str):
        return node
    return "[" + node["label"] + " " + " ".join(tree_to_str(c) for c in node["children"]) + " ]"


def tree_leaves(node) -> list[str]:
    if isinstance(node, str):
        return [node]
    return [w for c in node["children"] for w in tree_leaves(c)]


def _has_intent(node) -> bool:
    return isinstance(node, dict) and any(isinstance(c, dict) and c["label"].startswith("IN:")
                                          for c in node["children"])


def nested_slots(tree: dict) -> list[dict]:
    """Slots de premier niveau qui contiennent une intention imbriquée."""
    return [{"slot": c["label"], "surface": " ".join(tree_leaves(c)), "tree": tree_to_str(c)}
            for c in tree["children"] if _has_intent(c)]


class MtopLoader(DatasetLoader):
    name = "mtop"
    source_version = "MTOP v2 (2021-01-25), mtop.zip"

    def __init__(self, raw_dir: str | Path | None = None, langs: list[str] | None = None):
        self.raw_dir = raw_dir_for(self.name, raw_dir)
        self._langs = langs or ["en", "fr"]
        self._cache: dict[tuple[str, str], list[Example]] = {}
        self.skipped: dict[tuple[str, str], int] = defaultdict(int)

    def languages(self) -> list[str]:
        return list(self._langs)

    def download(self) -> Path:
        """Télécharge et décompresse l'archive officielle (réseau requis)."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        archive = self.raw_dir / "mtop.zip"
        if not archive.exists():
            urllib.request.urlretrieve(URL, archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(self.raw_dir)
        return archive

    def source_files(self, lang: str) -> list[Path]:
        return [self._path(lang, s) for s in SPLITS]

    def _path(self, lang: str, split: Split) -> Path:
        return find_file(self.raw_dir, f"{lang}/{_SPLIT_FILES[split]}")

    def _read(self, lang: str, split: Split) -> list[Example]:
        key = (lang, split)
        if key not in self._cache:
            out = []
            with self._path(lang, split).open(encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n").rstrip("\r")
                    if not line.strip():
                        continue
                    ex = self._convert(line.split("\t"), lang, split)
                    if ex is None:
                        self.skipped[key] += 1
                    else:
                        out.append(ex)
            self._cache[key] = out
        return self._cache[key]

    def _convert(self, cols: list[str], lang: str, split: Split) -> Example | None:
        if len(cols) < 7:
            return None
        src_id, intent, slot_str, utterance, domain, locale, decoupled = (c.strip() for c in cols[:7])
        slots = []
        for b0, b1, name in _SLOT_RE.findall(slot_str):
            start, end = byte_to_char_offsets(utterance, int(b0), int(b1))
            surface = utterance[start:end]
            slots.append(SlotValue(name=name, value=normalize_value(name, surface, lang),
                                   surface=surface, span=(start, end)))
        slots.sort(key=lambda s: s.span)
        nested = nested_slots(parse_decoupled(decoupled))
        return Example(
            id=f"{self.name}:{lang}:{split}:{src_id}", dataset="mtop", lang=lang, split=split,
            utterance=utterance, intent=intent, domain=domain, slots=slots,
            meta={"source_id": src_id, "locale": locale, "decoupled": decoupled,
                  "has_nested": bool(nested), "nested": nested},
        )

    def load(self, lang: str, split: Split) -> Iterator[Example]:
        yield from self._read(lang, split)

    def ontology(self, lang: str) -> Ontology:
        intent_slots: dict[str, set[str]] = defaultdict(set)
        intent_domain: dict[str, str] = {}
        slot_names: set[str] = set()
        train_values: dict[str, set[str]] = defaultdict(set)
        for split in SPLITS:
            for ex in self._read(lang, split):
                names = {s.name for s in ex.slots}
                slot_names |= names
                intent_slots[ex.intent] |= names
                intent_domain.setdefault(ex.intent, ex.domain)
                if split == "train":
                    for s in ex.slots:
                        train_values[s.name].add(s.value)
        intents = {n: IntentSpec(name=n, description=humanize(n), domain=intent_domain[n],
                                 slots=sorted(intent_slots[n])) for n in sorted(intent_slots)}
        slots = {n: SlotSpec(name=n, description=SLOT_DESC.get(n, humanize(n)),
                             values=sorted(train_values[n]) if n in CATEGORICAL_SLOTS and train_values[n]
                             else None)
                 for n in sorted(slot_names)}
        return Ontology(dataset="mtop", lang=lang, intents=intents, slots=slots)
