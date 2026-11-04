from pathlib import Path

import pytest

from laya_slots.data import Multi3WozLoader
from laya_slots.data.multi3woz import flatten_state, slot_base

FIX = Path(__file__).parent / "fixtures" / "multi3woz"


@pytest.fixture
def loader():
    return Multi3WozLoader(raw_dir=FIX)


def test_helpers():
    assert slot_base("hotel-book stay") == "stay" and slot_base("train-leaveat") == "leaveat"
    st = flatten_state({"hotel": {"semi": {"area": "not mentioned", "name": "Le Lac"},
                                  "book": {"booked": [{"x": 1}], "people": "4"}},
                        "bus": {"semi": {"day": "monday"}}}, "fr")
    assert st == {"hotel-name": "le lac", "hotel-book people": "4"}


@pytest.mark.parametrize("lang", ["fr", "en"])
def test_splits_and_turns(loader, lang):
    train = list(loader.load(lang, "train"))
    dev = list(loader.load(lang, "dev"))
    test = list(loader.load(lang, "test"))
    assert [e.id for e in train] == [f"multi3woz:{lang}:train:FAKE001:{i}" for i in (0, 2, 4)]
    assert len(dev) == 2 and len(test) == 1
    first, second, third = train
    assert first.history == [] and first.state_before == {}
    assert [t.speaker for t in second.history] == ["user", "system"]
    assert second.state_before == {"restaurant-name": "le petit zinc"}
    assert third.slots == [] and third.domain is None
    assert len(third.state_before) == 4


@pytest.mark.parametrize("lang,day_surface", [("fr", "vendredi"), ("en", "friday")])
def test_turn_slots_from_state_diff(loader, lang, day_surface):
    first, second, _ = loader.load(lang, "train")
    (name,) = first.slots
    assert name.name == "restaurant-name" and name.value == "le petit zinc" and name.surface == "Le Petit Zinc"
    assert first.domain == "restaurant" and first.intent is None
    by = {s.name: s for s in second.slots}
    assert set(by) == {"restaurant-book day", "restaurant-book time", "restaurant-book people"}
    assert by["restaurant-book day"].value == "friday"            # valeur canonique
    assert by["restaurant-book day"].surface == day_surface       # texte localisé de l'énoncé
    assert by["restaurant-book people"].surface == "4"            # acte Booking-Inform
    for s in second.slots:
        assert second.utterance[s.span[0]:s.span[1]] == s.surface and s.origin == "utterance"
    assert second.meta["span_source"]["restaurant-book day"] == "span_info"


@pytest.mark.parametrize("lang", ["fr", "en"])
def test_dontcare_history_and_context(loader, lang):
    h0, h2 = loader.load(lang, "dev")
    by = {s.name: s for s in h0.slots}
    assert by["hotel-area"].value == "dontcare" and by["hotel-area"].span is None
    assert by["hotel-area"].origin == "utterance" and h0.meta["span_source"]["hotel-area"] == "act"
    (name,) = h2.slots
    assert name.value == "hôtel du lac" and name.origin == "history" and name.span is None
    (t0,) = loader.load(lang, "test")
    by = {s.name: s for s in t0.slots}
    assert by["train-departure"].surface == "cambridge"
    assert by["train-leaveat"].value == "09:15" and by["train-leaveat"].origin == "context"


def test_ontology(loader):
    onto = loader.ontology("fr")
    assert set(onto.intents) == {"hotel", "train", "restaurant", "attraction", "taxi"}
    assert onto.slots["restaurant-name"].description == "name of the restaurant"
    assert onto.slots["restaurant-book time"].description == "restaurant book time"   # repli
    assert onto.slots["restaurant-book day"].values == ["friday"]
    assert onto.slots["hotel-area"].values is None          # « dontcare » n'est vu qu'en dev
    assert onto.slots["restaurant-name"].values is None
    assert "restaurant-book day" in onto.intents["restaurant"].slots
