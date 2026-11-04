"""Normalisation déterministe des valeurs de slots.

Une seule fonction publique : `normalize_value(slot, value, lang, ref_date=None)`. Elle est appliquée
aux valeurs gold à la conversion des jeux, et doit l'être aux valeurs candidates avant comparaison.

Étapes, dans l'ordre :
1. Unicode NFKC, apostrophes typographiques -> « ' », minuscules, espaces compactés,
   ponctuation de bord retirée, espaces autour des apostrophes supprimés (« l' aéroport » -> « l'aéroport »).
2. Dates relatives (si `ref_date` est fourni) : la valeur ENTIÈRE doit être une expression connue
   (today, tomorrow, demain, après-demain...) ; elle devient une date ISO AAAA-MM-JJ.
3. Nombres en toutes lettres -> chiffres, de 0 à 99 (EN et FR). Exceptions : en français « un »/« une »
   ne sont convertis que si la valeur entière est ce mot (sinon ce sont des articles) ; en anglais « a »
   n'est jamais converti. Les ordinaux (first, twenty first, premier) ne sont convertis que pour les
   slots de date (nom contenant « date » ou « day_number »).
4. Heures -> HH:MM : « 5 pm », « 5:30 p.m. », « five thirty pm », « 5 o'clock », « 9:05 », FR « 17h30 »,
   « 17 h », « 8 heures 15 », noon/midi -> 12:00, midnight/minuit -> 00:00. Pour les slots d'heure
   uniquement (nom contenant « time », « leaveat », « arriveby », mais pas « date »), une valeur entière de 3-4 chiffres
   (« 1700 », « 930 ») est lue comme une heure militaire.

La fonction ne lève jamais d'exception sur une entrée textuelle et ne dépend que de ses arguments.
Une valeur vide ou `NONE_VALUE` renvoie `NONE_VALUE`.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime, timedelta

from ..schema import NONE_VALUE

# ---------------------------------------------------------------------------
# Lexiques des nombres (0-99)
# ---------------------------------------------------------------------------
_EN_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
             "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
             "nineteen"]
_EN_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy",
            80: "eighty", 90: "ninety"}
_EN_ORD_UNITS = ["zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
                 "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth",
                 "sixteenth", "seventeenth", "eighteenth", "nineteenth"]
_EN_ORD_TENS = {20: "twentieth", 30: "thirtieth"}

_FR_UNITS = ["zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix",
             "onze", "douze", "treize", "quatorze", "quinze", "seize", "dix-sept", "dix-huit", "dix-neuf"]
_FR_TENS = {20: "vingt", 30: "trente", 40: "quarante", 50: "cinquante", 60: "soixante"}


def _en_cardinals() -> dict[str, int]:
    d = {w: i for i, w in enumerate(_EN_UNITS)}
    for t, w in _EN_TENS.items():
        d[w] = t
        for u in range(1, 10):
            d[f"{w}-{_EN_UNITS[u]}"] = t + u
            d[f"{w} {_EN_UNITS[u]}"] = t + u
    return d


def _en_ordinals() -> dict[str, int]:
    d = {w: i for i, w in enumerate(_EN_ORD_UNITS) if i > 0}
    d.update({w: t for t, w in _EN_ORD_TENS.items()})
    for t in (20, 30):
        for u in range(1, 10):
            if t + u > 31:
                break
            d[f"{_EN_TENS[t]}-{_EN_ORD_UNITS[u]}"] = t + u
            d[f"{_EN_TENS[t]} {_EN_ORD_UNITS[u]}"] = t + u
    return d


def _fr_cardinals() -> dict[str, int]:
    d = {w: i for i, w in enumerate(_FR_UNITS)}
    for t, w in _FR_TENS.items():
        d[w] = t
        for u in range(1, 10):
            d[f"{w} et un" if u == 1 else f"{w}-{_FR_UNITS[u]}"] = t + u
    for u in range(10, 20):                     # 70-79
        d["soixante et onze" if u == 11 else f"soixante-{_FR_UNITS[u]}"] = 60 + u
    d["quatre-vingts"] = d["quatre-vingt"] = 80
    for u in range(1, 20):                      # 81-99
        d[f"quatre-vingt-{_FR_UNITS[u]}"] = 80 + u
    # variantes : traits d'union <-> espaces (« vingt deux », « vingt-et-un »)
    for w, n in list(d.items()):
        d.setdefault(w.replace("-", " "), n)
        d.setdefault(w.replace(" ", "-"), n)
    del d["un"]                                 # article : traité à part
    return d


_EN_CARD = _en_cardinals()
_EN_ORD = _en_ordinals()
_FR_CARD = _fr_cardinals()
_FR_ORD = {"premier": 1, "première": 1, "1er": 1, "1ère": 1, "1re": 1}


def _alternation(words) -> re.Pattern:
    alts = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"(?<![\w-])(?:{alts})(?![\w-])")


_PATTERNS = {
    ("en", "card"): _alternation(_EN_CARD),
    ("en", "ord"): _alternation(_EN_ORD),
    ("fr", "card"): _alternation(_FR_CARD),
    ("fr", "ord"): _alternation(_FR_ORD),
}
_LEX = {("en", "card"): _EN_CARD, ("en", "ord"): _EN_ORD, ("fr", "card"): _FR_CARD, ("fr", "ord"): _FR_ORD}

# ---------------------------------------------------------------------------
# Dates relatives
# ---------------------------------------------------------------------------
_RELATIVE_DAYS = {
    "en": {"today": 0, "tonight": 0, "tomorrow": 1, "yesterday": -1, "day after tomorrow": 2,
           "the day after tomorrow": 2, "day before yesterday": -2, "the day before yesterday": -2},
    "fr": {"aujourd'hui": 0, "ce soir": 0, "demain": 1, "hier": -1, "après-demain": 2, "après demain": 2,
           "avant-hier": -2, "avant hier": -2},
}

_EDGE_PUNCT = " \t\n.,;:!?\"«»()[]{}"


def _lang(lang: str) -> str:
    lang = (lang or "en").lower()[:2]
    return lang if lang in ("en", "fr") else "en"


def _clean(value: str) -> str:
    v = unicodedata.normalize("NFKC", value)
    v = v.replace("’", "'").replace("‘", "'").replace("`", "'").replace("´", "'")
    v = v.lower()
    v = re.sub(r"\s+", " ", v).strip(_EDGE_PUNCT)
    v = re.sub(r"(?<=\w)\s*'\s*(?=\w)", "'", v)
    return v


def _is_time_slot(slot: str) -> bool:
    s = (slot or "").lower()
    if "date" in s:                 # SL:DATE_TIME : « 2020 » est une année, pas 20:20
        return False
    return "time" in s or "leaveat" in s or "arriveby" in s


def _is_date_slot(slot: str) -> bool:
    s = (slot or "").lower()
    return "date" in s or "day_number" in s


def _numbers(v: str, lang: str, slot: str) -> str:
    if lang == "fr" and v in ("un", "une"):
        return "1"
    kinds = ("ord", "card") if _is_date_slot(slot) else ("card",)
    for kind in kinds:
        lex = _LEX[(lang, kind)]
        v = _PATTERNS[(lang, kind)].sub(lambda m, lex=lex: str(lex[m.group(0)]), v)
    return v


def _hhmm(h: int, m: int) -> str | None:
    if 0 <= h <= 24 and 0 <= m < 60:
        return f"{h % 24:02d}:{m:02d}"
    return None


def _times(v: str, lang: str, slot: str) -> str:
    def ampm(mt: re.Match) -> str:
        h, m = int(mt.group(1)), int(mt.group(2) or 0)
        if h > 12:
            return mt.group(0)
        pm = mt.group(3).startswith("p")
        h = (h % 12) + (12 if pm else 0)
        return _hhmm(h, m) or mt.group(0)

    def plain(mt: re.Match) -> str:
        return _hhmm(int(mt.group(1)), int(mt.group(2) or 0)) or mt.group(0)

    v = re.sub(r"(?<![\w:])(\d{1,2})(?:[: ](\d{2}))?\s*([ap])\.?\s?m\b\.?", ampm, v)
    v = re.sub(r"(?<![\w:])(\d{1,2}) o'clock\b", lambda mt: _hhmm(int(mt.group(1)), 0) or mt.group(0), v)
    if lang == "fr":
        v = re.sub(r"(?<![\w:])(\d{1,2})\s*(?:h|heures?)(?:\s*(\d{2}))?(?![\w])", plain, v)
    v = re.sub(r"(?<![\w:])(\d{1,2}):(\d{2})(?![\w:])", plain, v)
    v = re.sub(r"(?<![\w-])(?:noon|midi)(?![\w-])", "12:00", v)
    v = re.sub(r"(?<![\w-])(?:midnight|minuit)(?![\w-])", "00:00", v)
    if _is_time_slot(slot):
        mt = re.fullmatch(r"(\d{1,2})(\d{2})", v)
        if mt:
            v = _hhmm(int(mt.group(1)), int(mt.group(2))) or v
    return v


def _as_date(ref_date) -> date | None:
    if ref_date is None:
        return None
    if isinstance(ref_date, datetime):
        return ref_date.date()
    if isinstance(ref_date, date):
        return ref_date
    return date.fromisoformat(str(ref_date)[:10])


def normalize_value(slot: str, value: str | None, lang: str, ref_date=None) -> str:
    """Normalise une valeur de slot (gold ou candidate). Voir la docstring du module."""
    if value is None:
        return NONE_VALUE
    if value == NONE_VALUE:
        return value
    lang = _lang(lang)
    v = _clean(str(value))
    if not v:
        return NONE_VALUE
    ref = _as_date(ref_date)
    if ref is not None and v in _RELATIVE_DAYS[lang]:
        return (ref + timedelta(days=_RELATIVE_DAYS[lang][v])).isoformat()
    v = _numbers(v, lang, slot)
    v = _times(v, lang, slot)
    return re.sub(r"\s+", " ", v).strip()
