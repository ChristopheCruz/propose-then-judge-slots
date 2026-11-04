"""Chargeurs MASSIVE et SGD sur de petites données synthétiques (même format que les sources)."""
import json

import pytest

from laya_slots.data import get_loader
from laya_slots.data.massive import locale_to_lang, parse_annot


def test_parse_annot_offsets():
    text, spans = parse_annot("wake me up at [time : nine am] on [date : friday]")
    assert text == "wake me up at nine am on friday"
    assert [(n, text[s:e]) for n, s, e in spans] == [("time", "nine am"), ("date", "friday")]
    assert parse_annot("no slot here") == ("no slot here", [])


def test_locale_to_lang():
    assert locale_to_lang("fr-FR") == "fr" and locale_to_lang("zh-CN") == "zh" and locale_to_lang("zh-TW") == "zh-tw"


@pytest.fixture
def massive_dir(tmp_path):
    d = tmp_path / "1.1" / "data"
    d.mkdir(parents=True)
    rows = [
        {"id": "1", "locale": "fr-FR", "partition": "train", "scenario": "alarm", "intent": "alarm_set",
         "utt": "réveille-moi à neuf heures", "annot_utt": "réveille-moi à [time : neuf heures]"},
        {"id": "2", "locale": "fr-FR", "partition": "dev", "scenario": "weather", "intent": "weather_query",
         "utt": "quel temps à paris", "annot_utt": "quel temps à [place_name : paris]"},
        {"id": "3", "locale": "fr-FR", "partition": "test", "scenario": "general", "intent": "general_joke",
         "utt": "raconte une blague", "annot_utt": "raconte une blague"},
    ]
    (d / "fr-FR.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return tmp_path


def test_massive_loader(massive_dir):
    loader = get_loader("massive", massive_dir)
    assert loader.languages() == ["fr"]
    train = list(loader.load("fr", "train"))
    assert len(train) == 1 and train[0].intent == "alarm_set" and train[0].domain == "alarm"
    sv = train[0].slots[0]
    assert sv.name == "time" and train[0].utterance[sv.span[0]:sv.span[1]] == "neuf heures"
    assert sv.value == "09:00"                                   # normalisation des heures en français
    assert [e.slots for e in loader.load("fr", "test")] == [[]]
    onto = loader.ontology("fr")
    assert set(onto.intents) == {"alarm_set", "weather_query", "general_joke"}
    assert onto.intents["weather_query"].slots == ["place_name"] and onto.slots["time"].description == "time of day"


@pytest.fixture
def sgd_dir(tmp_path):
    schema = [{"service_name": "Restaurants_1", "description": "find restaurants",
               "slots": [{"name": "city", "description": "city of the restaurant", "is_categorical": False},
                         {"name": "price_range", "description": "price range", "is_categorical": True,
                          "possible_values": ["cheap", "moderate", "expensive"]},
                         {"name": "restaurant_name", "description": "name of the restaurant", "is_categorical": False}],
               "intents": [{"name": "FindRestaurants", "description": "search restaurants",
                            "required_slots": ["city"], "optional_slots": {"price_range": "dontcare"}}]}]
    utt1, utt2 = "I want a cheap place in San Jose", "Yes, book Sushi Ya please"
    dialogue = {"dialogue_id": "1_00000", "services": ["Restaurants_1"], "turns": [
        {"speaker": "USER", "utterance": utt1, "frames": [{
            "service": "Restaurants_1",
            "slots": [{"slot": "city", "start": utt1.index("San Jose"), "exclusive_end": utt1.index("San Jose") + 8}],
            "state": {"active_intent": "FindRestaurants", "requested_slots": [],
                      "slot_values": {"city": ["San Jose"], "price_range": ["cheap"]}}}]},
        {"speaker": "SYSTEM", "utterance": "How about Sushi Ya?", "frames": [{"service": "Restaurants_1", "slots": []}]},
        {"speaker": "USER", "utterance": utt2, "frames": [{
            "service": "Restaurants_1", "slots": [],
            "state": {"active_intent": "FindRestaurants", "requested_slots": [],
                      "slot_values": {"city": ["San Jose"], "price_range": ["cheap"],
                                      "restaurant_name": ["Sushi Ya", "sushi ya restaurant"]}}}]},
    ]}
    for split in ("train", "dev", "test"):
        d = tmp_path / "dstc8-schema-guided-dialogue-master" / split
        d.mkdir(parents=True)
        (d / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
        (d / "dialogues_001.json").write_text(json.dumps([dialogue] if split == "train" else []), encoding="utf-8")
    return tmp_path


def test_sgd_loader(sgd_dir):
    loader = get_loader("sgd", sgd_dir)
    ex = list(loader.load("en", "train"))
    assert len(ex) == 2
    first, second = ex
    assert first.intent == "Restaurants_1.FindRestaurants" and first.domain == "Restaurants_1"
    by = {s.name: s for s in first.slots}
    assert by["Restaurants_1-city"].origin == "utterance" and by["Restaurants_1-city"].surface == "San Jose"
    assert by["Restaurants_1-price_range"].origin == "context"           # pas de span ni d'historique
    # second tour : seuls les slots nouveaux ; la valeur vient de l'offre du système (historique)
    assert [s.name for s in second.slots] == ["Restaurants_1-restaurant_name"]
    assert second.slots[0].origin == "history"
    assert second.state_before["Restaurants_1-city"] == by["Restaurants_1-city"].value
    assert second.history[-1].speaker == "system"
    onto = loader.ontology("en")
    assert onto.intents["Restaurants_1.FindRestaurants"].slots == ["Restaurants_1-city", "Restaurants_1-price_range"]
    assert onto.slots["Restaurants_1-price_range"].values == ["cheap", "expensive", "moderate"]
    assert list(loader.load("en", "dev")) == []


def test_massive_realign_without_spaces():
    from laya_slots.data.massive import realign
    text, spans = parse_annot("叫醒我 [time : 九點] 好嗎")
    assert realign("叫醒我九點好嗎", text, spans) == [("time", 3, 5)]
