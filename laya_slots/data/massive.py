"""Chargeur MASSIVE (FitzGerald et al., 2023) vers le format unifié.

Accès et licence
----------------
Librement téléchargeable, licence CC BY 4.0 :
https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz (version 1.1).
Décompresser sous `data/raw/massive/` ; le chargeur cherche `data/<locale>.jsonl` (51 locales, ex.
`fr-FR.jsonl`). `download()` le fait (réseau requis). 16 521 énoncés par locale : 11 514 / 2 033 / 2 974
(train / dev / test), parallèles entre locales (traductions de la même base anglaise, SLURP).

Format source
-------------
Une ligne JSON par énoncé : `id`, `locale`, `partition` (train, dev, test), `scenario` (18 domaines),
`intent` (60 intentions), `utt` (texte) et `annot_utt`, le texte annoté en ligne :
« réveille-moi [time : à neuf heures] ».

Choix de conversion
-------------------
- Langue : la locale réduite à la langue (`fr-FR` -> `fr`), sauf le chinois traditionnel (`zh-TW` -> `zh-tw`)
  pour le distinguer de `zh-CN` (`zh`). `meta["locale"]` garde la locale d'origine.
- Spans : relus dans `annot_utt`. Quand le texte reconstruit (annotations retirées) diffère de `utt`
  (espaces ajoutés autour des crochets, ex. en chinois traditionnel), l'énoncé est `utt` et chaque
  valeur y est recherchée dans l'ordre (`realign`).
- `Example.intent` = intention MASSIVE (ex. « alarm_set »), `domain` = scénario. Slots : noms MASSIVE
  (55 types, ex. « time », « place_name ») ; valeurs passées par `normalize_value`.
- Ontologie : intentions et slots observés dans les trois splits de la locale ; descriptions de slots
  rédigées en anglais (toutes langues), descriptions d'intentions dérivées du nom.
"""
from __future__ import annotations

import json
import re
import tarfile
import urllib.request
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from ..interfaces import DatasetLoader
from ..normalize import normalize_value
from ..schema import Example, IntentSpec, Ontology, SlotSpec, SlotValue, Split
from .common import SPLITS, humanize, raw_dir_for

URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
_ANNOT = re.compile(r"\[\s*([^\[\]:]+?)\s*:\s*([^\[\]]*?)\s*\]")

SLOT_DESC = {
    "date": "date", "time": "time of day", "timeofday": "part of the day (morning, evening...)",
    "place_name": "name of a place", "event_name": "name of an event", "person": "name of a person",
    "relation": "relation to the user (mom, boss...)", "business_name": "name of a business",
    "business_type": "type of business", "food_type": "type of food", "meal_type": "type of meal",
    "weather_descriptor": "weather condition", "temperature": "temperature",
    "media_type": "type of media (podcast, audiobook...)", "song_name": "name of a song",
    "artist_name": "name of an artist", "music_genre": "music genre", "playlist_name": "name of a playlist",
    "music_descriptor": "description of the music", "device_type": "type of device",
    "house_place": "room or place in the house", "color_type": "colour", "change_amount": "amount of change",
    "transport_type": "means of transport", "transport_name": "name of a transport service",
    "transport_agency": "transport company", "transport_descriptor": "description of the transport",
    "news_topic": "topic of the news", "currency_name": "name of a currency", "list_name": "name of a list",
    "general_frequency": "frequency (every day, weekly...)", "app_name": "name of an application",
    "email_address": "e-mail address", "email_folder": "e-mail folder", "personal_info": "personal information",
    "definition_word": "word to define", "game_name": "name of a game", "game_type": "type of game",
    "movie_name": "name of a movie", "movie_type": "type of movie", "radio_name": "name of a radio station",
    "audiobook_name": "name of an audiobook", "audiobook_author": "author of an audiobook",
    "podcast_name": "name of a podcast", "podcast_descriptor": "description of a podcast",
    "player_setting": "player setting (shuffle, repeat...)", "order_type": "type of order (delivery, takeaway...)",
    "joke_type": "type of joke", "coffee_type": "type of coffee", "cooking_type": "cooking method",
    "drink_type": "type of drink", "ingredient": "ingredient", "time_zone": "time zone",
    "alarm_type": "type of alarm", "query_detail": "detail asked about",
    "music_album": "name of a music album", "sport_type": "type of sport",
}


def locale_to_lang(locale: str) -> str:
    """`fr-FR` -> `fr` ; `zh-TW` -> `zh-tw` (seul cas où deux locales partagent une langue)."""
    return "zh-tw" if locale == "zh-TW" else locale.split("-")[0].lower()


def parse_annot(annot: str) -> tuple[str, list[tuple[str, int, int]]]:
    """Texte sans annotations et spans (slot, début, fin) en offsets caractères de ce texte."""
    out, spans, pos = [], [], 0
    for m in _ANNOT.finditer(annot):
        out.append(annot[pos:m.start()])
        start = sum(len(x) for x in out)
        value = m.group(2)
        out.append(value)
        spans.append((m.group(1).strip(), start, start + len(value)))
        pos = m.end()
    out.append(annot[pos:])
    return "".join(out), spans


def realign(utt: str, text: str, spans: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """Spans relus dans `utt` quand le texte reconstruit diffère (espaces ajoutés par l'annotation, ex.
    autour des crochets en chinois) : chaque valeur est recherchée dans l'ordre, telle quelle puis sans
    espaces ; une valeur introuvable est abandonnée."""
    out, pos = [], 0
    for name, s, e in spans:
        value = text[s:e]
        for v in (value, value.replace(" ", "")):
            i = utt.find(v, pos) if v else -1
            if i >= 0:
                out.append((name, i, i + len(v)))
                pos = i + len(v)
                break
    return out


class MassiveLoader(DatasetLoader):
    name = "massive"
    source_version = "MASSIVE 1.1 (amazon-massive-dataset-1.1.tar.gz)"

    def __init__(self, raw_dir: str | Path | None = None, langs: list[str] | None = None):
        self.raw_dir = raw_dir_for(self.name, raw_dir)
        self._langs = langs
        self._cache: dict[str, dict[str, list[Example]]] = {}
        self.skipped: dict[tuple[str, str], int] = defaultdict(int)

    def _files(self) -> dict[str, Path]:
        """langue -> fichier `<locale>.jsonl` (cherché récursivement sous raw_dir)."""
        files = {}
        for p in sorted(self.raw_dir.rglob("*.jsonl")):
            if re.fullmatch(r"[a-z]{2}-[A-Z]{2}", p.stem):
                files[locale_to_lang(p.stem)] = p
        return files

    def languages(self) -> list[str]:
        return list(self._langs) if self._langs else sorted(self._files())

    def download(self) -> Path:
        """Télécharge et décompresse l'archive officielle (réseau requis)."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        archive = self.raw_dir / Path(URL).name
        if not archive.exists():
            urllib.request.urlretrieve(URL, archive)
        with tarfile.open(archive) as t:
            t.extractall(self.raw_dir, filter="data")
        return archive

    def source_files(self, lang: str) -> list[Path]:
        return [self._path(lang)]

    def _path(self, lang: str) -> Path:
        files = self._files()
        if lang not in files:
            raise FileNotFoundError(f"MASSIVE : pas de fichier pour la langue {lang!r} sous {self.raw_dir}")
        return files[lang]

    def _read(self, lang: str) -> dict[str, list[Example]]:
        if lang not in self._cache:
            by_split: dict[str, list[Example]] = {s: [] for s in SPLITS}
            with self._path(lang).open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    split = rec.get("partition")
                    if split not in by_split:
                        self.skipped[(lang, str(split))] += 1
                        continue
                    by_split[split].append(self.convert(rec, lang, split))
            self._cache[lang] = by_split
        return self._cache[lang]

    def convert(self, rec: dict, lang: str, split: Split) -> Example:
        text, spans = parse_annot(rec.get("annot_utt") or rec.get("utt", ""))
        utt = rec.get("utt")
        if utt and utt != text:
            spans, text = realign(utt, text, spans), utt
        slots = [SlotValue(name=name, value=normalize_value(name, text[s:e], lang), surface=text[s:e], span=(s, e))
                 for name, s, e in spans if e > s]
        return Example(
            id=f"{self.name}:{lang}:{split}:{rec['id']}", dataset="massive", lang=lang, split=split,
            utterance=text, intent=rec.get("intent"), domain=rec.get("scenario"), slots=slots,
            meta={"source_id": rec["id"], "locale": rec.get("locale"), "utt": rec.get("utt")},
        )

    def load(self, lang: str, split: Split) -> Iterator[Example]:
        yield from self._read(lang)[split]

    def ontology(self, lang: str) -> Ontology:
        intent_slots: dict[str, set[str]] = defaultdict(set)
        intent_domain: dict[str, str] = {}
        slot_names: set[str] = set()
        for split in SPLITS:
            for ex in self._read(lang)[split]:
                names = {s.name for s in ex.slots}
                slot_names |= names
                intent_slots[ex.intent] |= names
                intent_domain.setdefault(ex.intent, ex.domain)
        intents = {n: IntentSpec(name=n, description=humanize(n), domain=intent_domain[n],
                                 slots=sorted(intent_slots[n])) for n in sorted(intent_slots)}
        slots = {n: SlotSpec(name=n, description=SLOT_DESC.get(n, humanize(n))) for n in sorted(slot_names)}
        return Ontology(dataset="massive", lang=lang, intents=intents, slots=slots)
