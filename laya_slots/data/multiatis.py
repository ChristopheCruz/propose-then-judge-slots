"""Chargeur MultiATIS++ (Xu et al., 2020) vers le format unifié.

Accès et licence
----------------
MultiATIS++ n'est PAS librement téléchargeable : il est distribué par le LDC (catalogue LDC2021T04,
« ATIS - Seven Languages », licence LDC pour membres ou non-membres). Les données ne sont jamais
téléchargées par ce module. Une fois la copie LDC obtenue, placer les fichiers TSV (à n'importe quelle
profondeur) sous `data/raw/multiatis/` ; le chargeur cherche `train_EN.tsv`, `dev_EN.tsv`, `test_EN.tsv`,
`train_FR.tsv`, etc.

Split dev de la version LDC
---------------------------
La copie LDC (`atis_seven_languages`) ne fournit que train (4 978) et test (893), plus des tests
localisés `test_XX_loc.tsv` (non utilisés ici). Si `dev_XX.tsv` est absent, le dev est prélevé dans le
train : les DEV_SIZE = 500 identifiants de plus petit sha256 (découpage 4 478 / 500 / 893 usuel sur ATIS).
Les identifiants étant alignés entre langues, EN et FR ont exactement les mêmes énoncés en dev.
Ce n'est PAS le dev de l'article (4 488 / 490) : à mentionner dans le papier.

Format source (README de amazon-science/stil-mbart-multiatis)
-------------------------------------------------------------
TSV UTF-8 avec en-tête `id  utterance  slot_labels  intent`. `utterance` est tokenisé par des espaces ;
`slot_labels` contient une étiquette BIO par jeton (« O », « B-fromloc.city_name »...). Effectifs publiés :
4 488 / 490 / 893 énoncés (train / dev / test) en EN et en FR.

Choix de conversion
-------------------
- Spans : BIO -> spans en jetons (règle tolérante : un I-x orphelin ouvre un span), puis offsets
  caractères retrouvés séquentiellement dans l'énoncé original. `surface` = texte exact du span,
  `value` = `normalize_value(slot, surface, lang)`.
- Intentions multiples (« atis_flight#atis_airfare », ~1-2 % des énoncés) : `Example.intent` est la
  PREMIÈRE intention (intention principale) ; `meta["intents"]` donne la liste complète et
  `meta["intent_raw"]` l'étiquette d'origine, pour calculer l'exactitude d'intention officielle (égalité
  de chaînes) ou une évaluation multi-label. L'ontologie ne contient que des intentions atomiques.
- Ontologie : noms d'intentions et de slots observés dans les trois splits (le schéma est supposé connu) ;
  descriptions rédigées à la main, composées préfixe + attribut (« fromloc » + « city_name » ->
  « departure city »), en anglais pour EN et en français pour FR ; valeurs fermées pour les attributs
  catégoriels seulement (jour, mois, période, classe...), relevées sur le TRAIN uniquement.
- Jetons = découpage sur l'espace ordinaire (les espaces insécables du FR restent dans le jeton).
- Une ligne dont le nombre d'étiquettes diffère du nombre de jetons est ignorée et comptée dans
  `self.skipped` (copie LDC : 13 lignes du train FR, erreurs d'annotation du corpus).
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from ..interfaces import DatasetLoader
from ..normalize import normalize_value
from ..schema import Example, IntentSpec, Ontology, SlotSpec, SlotValue, Split
from .common import SPLITS, bio_to_spans, find_file, humanize, raw_dir_for, token_offsets

INTENT_SEP = "#"
_LANG_CODES = {"en": "EN", "fr": "FR", "de": "DE", "es": "ES", "pt": "PT", "zh": "ZH", "ja": "JA",
               "hi": "HI", "tr": "TR"}

_INTENT_DESC = {
    "atis_abbreviation": ("meaning of an abbreviation or code", "signification d'une abréviation ou d'un code"),
    "atis_aircraft": ("type of aircraft", "type d'avion"),
    "atis_airfare": ("price of a flight", "prix d'un vol"),
    "atis_airline": ("airline information", "informations sur une compagnie aérienne"),
    "atis_airport": ("airport information", "informations sur un aéroport"),
    "atis_capacity": ("seating capacity of an aircraft", "nombre de places d'un avion"),
    "atis_cheapest": ("cheapest fare or flight", "vol ou tarif le moins cher"),
    "atis_city": ("city served by an airline or airport", "ville desservie par une compagnie ou un aéroport"),
    "atis_day_name": ("day of the week of a flight", "jour de la semaine d'un vol"),
    "atis_distance": ("distance between places", "distance entre deux lieux"),
    "atis_flight": ("search for flights", "recherche de vols"),
    "atis_flight_no": ("flight number", "numéro de vol"),
    "atis_flight_time": ("schedule or duration of a flight", "horaire ou durée d'un vol"),
    "atis_ground_fare": ("price of ground transportation", "prix d'un transport terrestre"),
    "atis_ground_service": ("ground transportation", "transport terrestre"),
    "atis_meal": ("meals served on a flight", "repas servis à bord"),
    "atis_quantity": ("number of flights, airlines or seats", "nombre de vols, compagnies ou places"),
    "atis_restriction": ("fare restrictions", "restrictions tarifaires"),
}

_PREFIX_DESC = {
    "fromloc": ("departure", "départ"),
    "toloc": ("arrival", "arrivée"),
    "stoploc": ("stopover", "escale"),
    "depart_date": ("departure date", "date de départ"),
    "arrive_date": ("arrival date", "date d'arrivée"),
    "return_date": ("return date", "date de retour"),
    "depart_time": ("departure time", "heure de départ"),
    "arrive_time": ("arrival time", "heure d'arrivée"),
    "return_time": ("return time", "heure de retour"),
}

_ATTR_DESC = {
    "city_name": ("city", "ville"),
    "airport_code": ("airport code", "code d'aéroport"),
    "airport_name": ("airport name", "nom d'aéroport"),
    "state_code": ("state code", "code d'État"),
    "state_name": ("state name", "nom d'État"),
    "country_name": ("country", "pays"),
    "day_name": ("day of the week", "jour de la semaine"),
    "day_number": ("day of the month", "jour du mois"),
    "month_name": ("month", "mois"),
    "year": ("year", "année"),
    "date_relative": ("relative date (next, following...)", "date relative (prochain, suivant...)"),
    "today_relative": ("date relative to today (today, tomorrow...)",
                       "date relative à aujourd'hui (aujourd'hui, demain...)"),
    "time": ("clock time", "heure"),
    "start_time": ("start of a time range", "début d'une plage horaire"),
    "end_time": ("end of a time range", "fin d'une plage horaire"),
    "period_of_day": ("period of the day (morning, evening...)", "moment de la journée (matin, soir...)"),
    "period_mod": ("modifier of the period of the day (early, late)", "précision du moment (tôt, tard)"),
    "time_relative": ("time comparison (before, after...)", "comparaison horaire (avant, après...)"),
    "aircraft_code": ("aircraft code", "code d'avion"),
    "airline_code": ("airline code", "code de compagnie aérienne"),
    "airline_name": ("airline name", "nom de compagnie aérienne"),
    "booking_class": ("booking class code", "code de classe de réservation"),
    "class_type": ("travel class (first, business, coach...)", "classe de voyage (première, affaires...)"),
    "compartment": ("cabin compartment", "compartiment de cabine"),
    "connect": ("connection requirement", "exigence de correspondance"),
    "cost_relative": ("price comparison (cheapest, less than...)", "comparaison de prix (moins cher...)"),
    "days_code": ("days-of-operation code", "code des jours d'exploitation"),
    "economy": ("economy fare", "tarif économique"),
    "fare_amount": ("fare amount", "montant du tarif"),
    "fare_basis_code": ("fare basis code", "code de base tarifaire"),
    "flight": ("flight type", "type de vol"),
    "flight_days": ("days of operation of the flight", "jours d'exploitation du vol"),
    "flight_mod": ("flight modifier (first, last, earliest...)", "précision sur le vol (premier, dernier...)"),
    "flight_number": ("flight number", "numéro de vol"),
    "flight_stop": ("number of stops (nonstop, direct...)", "nombre d'escales (sans escale, direct...)"),
    "flight_time": ("flight time or duration", "horaire ou durée du vol"),
    "meal": ("meal", "repas"),
    "meal_code": ("meal code", "code de repas"),
    "meal_description": ("meal description (breakfast, dinner...)", "description du repas (petit-déjeuner...)"),
    "mod": ("generic modifier", "modificateur générique"),
    "or": ("alternative (or)", "alternative (ou)"),
    "restriction_code": ("fare restriction code", "code de restriction tarifaire"),
    "round_trip": ("round trip or one way", "aller-retour ou aller simple"),
    "transport_type": ("ground transport type (taxi, limousine...)", "type de transport terrestre (taxi...)"),
}

# Attributs catégoriels : la liste des valeurs (relevée sur le train) est publiée dans l'ontologie.
CATEGORICAL_ATTRS = {"day_name", "month_name", "period_of_day", "period_mod", "today_relative",
                     "date_relative", "time_relative", "class_type", "round_trip", "meal",
                     "meal_description", "cost_relative", "flight_mod", "flight_stop", "connect",
                     "economy", "transport_type", "or", "mod"}


def slot_description(name: str, lang: str) -> str:
    """Description lisible d'un slot ATIS, composée préfixe + attribut ; repli sur le nom humanisé."""
    i = 1 if lang == "fr" else 0
    prefix, _, attr = name.rpartition(".")
    attr_d = _ATTR_DESC.get(attr)
    if attr_d is None:
        return humanize(name)
    if not prefix:
        return attr_d[i]
    pre_d = _PREFIX_DESC.get(prefix)
    if pre_d is None:
        return f"{humanize(prefix)}: {attr_d[i]}"
    # « departure city », « ville de départ » ; pour les dates/heures : « departure date: day of the week »
    if prefix in ("fromloc", "toloc", "stoploc"):
        return f"{pre_d[0]} {attr_d[0]}" if i == 0 else f"{attr_d[1]} ({pre_d[1]})"
    return f"{pre_d[i]}: {attr_d[i]}"


def split_intents(raw: str) -> list[str]:
    return [p for p in raw.strip().split(INTENT_SEP) if p]


DEV_SIZE = 500


def derived_dev_ids(train_ids: list[str], size: int | None = None) -> set[str]:
    """Identifiants du dev prélevé dans le train : tirage déterministe, indépendant de la langue."""
    size = DEV_SIZE if size is None else size
    return set(sorted(train_ids, key=lambda i: hashlib.sha256(i.encode("utf-8")).hexdigest())[:size])


class MultiAtisLoader(DatasetLoader):
    name = "multiatis"
    source_version = "MultiATIS++ (LDC2021T04)"

    def __init__(self, raw_dir: str | Path | None = None, langs: list[str] | None = None):
        self.raw_dir = raw_dir_for(self.name, raw_dir)
        self._langs = langs or ["en", "fr"]
        self._cache: dict[tuple[str, str], list[Example]] = {}
        self.skipped: dict[tuple[str, str], int] = defaultdict(int)

    def languages(self) -> list[str]:
        return list(self._langs)

    def source_files(self, lang: str) -> list[Path]:
        return sorted({self._path(lang, s) for s in SPLITS})

    def _path(self, lang: str, split: Split) -> Path:
        code = _LANG_CODES[lang]
        if split == "dev" and not self._has_dev(lang):
            return self._path(lang, "train")       # dev prélevé dans le train (voir docstring)
        return find_file(self.raw_dir, f"{split}_{code}.tsv")

    def _has_dev(self, lang: str) -> bool:
        try:
            find_file(self.raw_dir, f"dev_{_LANG_CODES[lang]}.tsv")
            return True
        except FileNotFoundError:
            return False

    def _read(self, lang: str, split: Split) -> list[Example]:
        key = (lang, split)
        if key in self._cache:
            return self._cache[key]
        if split in ("train", "dev") and not self._has_dev(lang):
            out = self._read_file(lang, "train", self._path(lang, "train"))
            dev = derived_dev_ids([ex.meta["source_id"] for ex in out])
            self._cache[(lang, "dev")] = [self._resplit(ex, "dev") for ex in out if ex.meta["source_id"] in dev]
            self._cache[(lang, "train")] = [ex for ex in out if ex.meta["source_id"] not in dev]
            return self._cache[key]
        self._cache[key] = self._read_file(lang, split, self._path(lang, split))
        return self._cache[key]

    @staticmethod
    def _resplit(ex: Example, split: Split) -> Example:
        ex.split = split
        ex.id = f"multiatis:{ex.lang}:{split}:{ex.meta['source_id']}"
        return ex

    def _read_file(self, lang: str, split: Split, path: Path) -> list[Example]:
        key = (lang, split)
        out = []
        with path.open(encoding="utf-8-sig") as f:
            for n, line in enumerate(f):
                line = line.rstrip("\n").rstrip("\r")
                if not line.strip():
                    continue
                cols = line.split("\t")
                if n == 0 and cols[0].strip().lower() == "id":
                    continue
                ex = self._convert(cols, lang, split)
                if ex is None:
                    self.skipped[key] += 1
                else:
                    out.append(ex)
        return out

    def _convert(self, cols: list[str], lang: str, split: Split) -> Example | None:
        if len(cols) < 4:
            return None
        src_id, utterance, labels, intent_raw = cols[0].strip(), cols[1], cols[2], cols[3].strip()
        # Jetons séparés par des espaces ORDINAIRES seulement : en FR, « 7\xa0heures » (espace insécable)
        # est un seul jeton annoté ; str.split() le couperait et décalerait les étiquettes.
        tokens, tags = [t for t in utterance.split(" ") if t], labels.split()
        if len(tokens) != len(tags):
            return None
        try:
            offsets = token_offsets(utterance, tokens)
        except ValueError:
            return None
        slots = []
        for name, i, j in bio_to_spans(tags):
            start, end = offsets[i][0], offsets[j - 1][1]
            surface = utterance[start:end]
            slots.append(SlotValue(name=name, value=normalize_value(name, surface, lang),
                                   surface=surface, span=(start, end)))
        intents = split_intents(intent_raw)
        return Example(
            id=f"{self.name}:{lang}:{split}:{src_id}", dataset="multiatis", lang=lang, split=split,
            utterance=utterance, intent=intents[0] if intents else None, domain="flights", slots=slots,
            meta={"source_id": src_id, "intents": intents, "intent_raw": intent_raw},
        )

    def load(self, lang: str, split: Split) -> Iterator[Example]:
        yield from self._read(lang, split)

    def ontology(self, lang: str) -> Ontology:
        intent_slots: dict[str, set[str]] = defaultdict(set)
        slot_names: set[str] = set()
        train_values: dict[str, set[str]] = defaultdict(set)
        for split in SPLITS:
            for ex in self._read(lang, split):
                names = {s.name for s in ex.slots}
                slot_names |= names
                for it in ex.meta["intents"]:
                    intent_slots[it] |= names
                if split == "train":
                    for s in ex.slots:
                        train_values[s.name].add(s.value)
        i = 1 if lang == "fr" else 0
        intents = {
            name: IntentSpec(name=name, description=_INTENT_DESC.get(name, (humanize(name),) * 2)[i],
                             domain="flights", slots=sorted(intent_slots[name]))
            for name in sorted(intent_slots)
        }
        slots = {}
        for name in sorted(slot_names):
            attr = name.rpartition(".")[2]
            values = sorted(train_values[name]) if attr in CATEGORICAL_ATTRS and train_values[name] else None
            slots[name] = SlotSpec(name=name, description=slot_description(name, lang), values=values)
        return Ontology(dataset="multiatis", lang=lang, intents=intents, slots=slots)
