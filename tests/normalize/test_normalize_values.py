from datetime import date

import pytest

from laya_slots.normalize import normalize_value
from laya_slots.schema import NONE_VALUE


@pytest.mark.parametrize("slot,value,expected", [
    ("toloc.city_name", "  New   York. ", "new york"),
    ("x", "Twenty-One", "21"),
    ("x", "twenty one", "21"),
    ("x", "one way", "1 way"),
    ("x", "a table", "a table"),
    ("depart_time.time", "5 pm", "17:00"),
    ("depart_time.time", "five thirty pm", "17:30"),
    ("depart_time.time", "8:05 a.m.", "08:05"),
    ("depart_time.time", "12 am", "00:00"),
    ("depart_time.time", "12 pm", "12:00"),
    ("x", "9:5", "9:5"),
    ("x", "5 o'clock", "05:00"),
    ("x", "noon", "12:00"),
    ("x", "midnight", "00:00"),
    ("x", "afternoon", "afternoon"),
    ("depart_time.time", "1700", "17:00"),
    ("flight_number", "1700", "1700"),
    ("SL:DATE_TIME", "2020", "2020"),
    ("train-leaveat", "0915", "09:15"),
    ("arrive_date.day_number", "twenty first", "21"),
    ("arrive_date.day_number", "first", "1"),
    ("flight_mod", "first", "first"),
    ("x", "Mom’s", "mom's"),
])
def test_normalize_en(slot, value, expected):
    assert normalize_value(slot, value, "en") == expected


@pytest.mark.parametrize("slot,value,expected", [
    ("x", "  Le  Bambou Royal ", "le bambou royal"),
    ("x", "l’ aéroport", "l'aéroport"),
    ("x", "vingt-deux", "22"),
    ("x", "vingt et un", "21"),
    ("x", "soixante-dix-sept", "77"),
    ("x", "quatre-vingts", "80"),
    ("x", "dix sept", "17"),
    ("book people", "une", "1"),
    ("x", "une table", "une table"),
    ("x", "deux personnes", "2 personnes"),
    ("t", "17h30", "17:30"),
    ("t", "17 h", "17:00"),
    ("t", "8 heures 15", "08:15"),
    ("t", "dix-sept heures", "17:00"),
    ("t", "midi", "12:00"),
    ("t", "minuit", "00:00"),
    ("x", "3 hôtels", "3 hôtels"),
    ("depart_date.day_number", "premier", "1"),
])
def test_normalize_fr(slot, value, expected):
    assert normalize_value(slot, value, "fr") == expected


def test_relative_dates_need_ref_date():
    assert normalize_value("date", "tomorrow", "en") == "tomorrow"
    assert normalize_value("date", "Tomorrow", "en", ref_date="2026-10-01") == "2026-10-02"
    assert normalize_value("date", "yesterday", "en", ref_date=date(2026, 10, 1)) == "2026-09-30"
    assert normalize_value("date", "demain", "fr", ref_date="2026-12-31") == "2027-01-01"
    assert normalize_value("date", "après-demain", "fr", ref_date="2026-10-01") == "2026-10-03"
    assert normalize_value("date", "aujourd'hui", "fr", ref_date="2026-10-01") == "2026-10-01"
    # seule la valeur entière est remplacée
    assert normalize_value("date", "for tomorrow", "en", ref_date="2026-10-01") == "for tomorrow"


def test_none_and_empty():
    assert normalize_value("x", None, "en") == NONE_VALUE
    assert normalize_value("x", "   ", "fr") == NONE_VALUE
    assert normalize_value("x", NONE_VALUE, "fr") == NONE_VALUE


def test_idempotent_and_unknown_lang():
    for v, lang in [("five thirty pm", "en"), ("17h30", "fr"), ("Vingt-Deux", "fr"), ("New York", "en")]:
        once = normalize_value("depart_time.time", v, lang)
        assert normalize_value("depart_time.time", once, lang) == once
    assert normalize_value("x", "Five", "de") == "5"      # langue inconnue : règles anglaises
