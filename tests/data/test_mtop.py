from pathlib import Path

import pytest

from laya_slots.data import MtopLoader
from laya_slots.data.mtop import parse_decoupled, tree_to_str

FIX = Path(__file__).parent / "fixtures" / "mtop"


@pytest.fixture
def loader():
    return MtopLoader(raw_dir=FIX)


def test_parse_decoupled_roundtrip():
    form = "[IN:A [SL:B x y ] [SL:C [IN:D [SL:E z ] ] ] ]"
    tree = parse_decoupled(form)
    assert tree["label"] == "IN:A" and tree_to_str(tree) == form


def test_load_en(loader):
    exs = list(loader.load("en", "train"))
    assert [e.id for e in exs] == ["mtop:en:train:101", "mtop:en:train:102", "mtop:en:train:103"]
    ex = exs[0]
    assert ex.intent == "IN:GET_WEATHER" and ex.domain == "weather"
    assert [(s.name, s.surface) for s in ex.slots] == [("SL:LOCATION", "Paris"), ("SL:DATE_TIME", "for tomorrow")]
    assert ex.slots[0].value == "paris"
    assert not ex.meta["has_nested"]


def test_nested_flattened(loader):
    ex = list(loader.load("en", "train"))[1]
    assert [s.name for s in ex.slots] == ["SL:PERSON_REMINDED", "SL:TODO", "SL:DATE_TIME"]
    todo = ex.slots[1]
    assert todo.surface == "call mom" and todo.value == "call mom"
    assert ex.slots[2].value == "tonight at 17:00"
    assert ex.meta["has_nested"]
    (nested,) = ex.meta["nested"]
    assert nested["slot"] == "SL:TODO" and nested["surface"] == "mom"
    assert nested["tree"].startswith("[SL:TODO [IN:CREATE_CALL")


def test_fr_byte_offsets_and_fullwidth_comma(loader):
    exs = list(loader.load("fr", "train"))
    assert exs[0].slots[0].surface == "Orléans" and exs[0].slots[0].span == (8, 15)
    rem = exs[1]
    assert [s.surface for s in rem.slots] == ["moi", "appeler maman", "ce soir à 17h"]
    assert rem.slots[2].value == "ce soir à 17:00"
    odd = exs[2]
    assert [s.name for s in odd.slots] == ["SL:PERSON_REMINDED", "SL:DATE_TIME", "SL:TODO"]
    for ex in exs:
        for s in ex.slots:
            assert ex.utterance[s.span[0]:s.span[1]] == s.surface


def test_no_slots_and_ontology(loader):
    (ex,) = loader.load("en", "dev")
    assert ex.slots == [] and ex.intent == "IN:GET_ALARM"
    onto = loader.ontology("en")
    assert onto.intents["IN:GET_WEATHER"].domain == "weather"
    assert onto.intents["IN:GET_WEATHER"].slots == ["SL:DATE_TIME", "SL:LOCATION", "SL:WEATHER_ATTRIBUTE"]
    assert onto.slots["SL:DATE_TIME"].description == "date or time"
    assert onto.slots["SL:METHOD_TIMER"].values == ["timer"]
    assert onto.slots["SL:LOCATION"].values is None
    assert onto.intents["IN:CREATE_REMINDER"].description == "create reminder"
