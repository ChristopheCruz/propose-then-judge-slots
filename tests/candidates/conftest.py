"""Fixtures construites à la main : style ATIS (mono-tour) et style MultiWOZ (multi-tours)."""
import pytest

from laya_slots.schema import Example, IntentSpec, Ontology, SlotSpec, SlotValue, Turn


class LockedExample(Example):
    """Exemple dont les gold sont inaccessibles : tout accès à `slots` lève une exception."""

    def __getattribute__(self, name):
        if name == "slots":
            raise AssertionError("accès interdit à example.slots (gold)")
        return super().__getattribute__(name)


def lock(ex: Example) -> LockedExample:
    d = {f: object.__getattribute__(ex, f) for f in ex.__dataclass_fields__}
    return LockedExample(**d)


def atis(i, utt, slots, intent="atis_flight", split="test", lang="en"):
    svs = []
    for name, value in slots:
        s = utt.lower().find(value)
        svs.append(SlotValue(name, value, value if s >= 0 else None, (s, s + len(value)) if s >= 0 else None))
    return Example(id=f"multiatis:{lang}:{split}:{i}", dataset="multiatis", lang=lang, split=split,
                   utterance=utt, intent=intent, slots=svs)


@pytest.fixture
def make_atis():
    return atis


@pytest.fixture
def lock_example():
    return lock


@pytest.fixture
def atis_ontology():
    slots = {
        "fromloc.city_name": SlotSpec("fromloc.city_name", "departure city"),
        "toloc.city_name": SlotSpec("toloc.city_name", "arrival city"),
        "depart_date.day_name": SlotSpec("depart_date.day_name", "day of departure",
                                         ["monday", "tuesday", "wednesday", "thursday", "friday",
                                          "saturday", "sunday"]),
        "airline_name": SlotSpec("airline_name", "airline"),
        "class_type": SlotSpec("class_type", "cabin class", ["first class", "business", "economy", "coach"]),
    }
    intents = {"atis_flight": IntentSpec("atis_flight", "find a flight", None, list(slots)),
               "atis_airfare": IntentSpec("atis_airfare", "ask for a fare", None,
                                          ["fromloc.city_name", "toloc.city_name", "class_type"])}
    return Ontology("multiatis", "en", intents, slots)


@pytest.fixture
def atis_train():
    return [
        atis(1, "flights from boston to denver", [("fromloc.city_name", "boston"), ("toloc.city_name", "denver")],
             split="train"),
        atis(2, "show me flights from denver to san francisco on monday",
             [("fromloc.city_name", "denver"), ("toloc.city_name", "san francisco"),
              ("depart_date.day_name", "monday")], split="train"),
        atis(3, "i want to fly from boston to new york with delta",
             [("fromloc.city_name", "boston"), ("toloc.city_name", "new york"), ("airline_name", "delta")],
             split="train"),
        atis(4, "fares from pittsburgh to baltimore in first class",
             [("fromloc.city_name", "pittsburgh"), ("toloc.city_name", "baltimore"),
              ("class_type", "first class")], intent="atis_airfare", split="train"),
        atis(5, "flights to boston from atlanta on united",
             [("toloc.city_name", "boston"), ("fromloc.city_name", "atlanta"), ("airline_name", "united")],
             split="train"),
        atis(6, "flights from dallas to boston", [("fromloc.city_name", "dallas"), ("toloc.city_name", "boston")],
             split="train"),
        atis(7, "flights from philadelphia to dallas",
             [("fromloc.city_name", "philadelphia"), ("toloc.city_name", "dallas")], split="train"),
    ]


@pytest.fixture
def atis_test():
    return [
        atis(10, "Flights from Boston to San Francisco on Thursday",
             [("fromloc.city_name", "boston"), ("toloc.city_name", "san francisco"),
              ("depart_date.day_name", "thursday")]),
        atis(11, "what are the fares from denver to atlanta",
             [("fromloc.city_name", "denver"), ("toloc.city_name", "atlanta")], intent="atis_airfare"),
        atis(12, "delta flights to new york on wenesday",
             [("airline_name", "delta"), ("toloc.city_name", "new york"), ("depart_date.day_name", "wednesday")]),
    ]


@pytest.fixture
def woz_ontology():
    slots = {
        "hotel-area": SlotSpec("hotel-area", "area of the hotel", ["centre", "north", "south", "east", "west"]),
        "hotel-pricerange": SlotSpec("hotel-pricerange", "price budget of the hotel",
                                     ["cheap", "moderate", "expensive", "dontcare"]),
        "hotel-name": SlotSpec("hotel-name", "name of the hotel",
                               ["acorn guest house", "gonville hotel", "hamilton lodge"]),
        "restaurant-area": SlotSpec("restaurant-area", "area of the restaurant",
                                    ["centre", "north", "south", "east", "west"]),
        "restaurant-food": SlotSpec("restaurant-food", "type of food", None),
    }
    intents = {"find_hotel": IntentSpec("find_hotel", "find a hotel", "hotel",
                                        ["hotel-area", "hotel-pricerange", "hotel-name"]),
               "find_restaurant": IntentSpec("find_restaurant", "find a restaurant", "restaurant",
                                             ["restaurant-area", "restaurant-food"])}
    return Ontology("multi3woz", "en", intents, slots)


@pytest.fixture
def woz_train():
    return [
        Example("multi3woz:en:train:d1:0", "multi3woz", "en", "train", "i need a cheap hotel in the north",
                "find_hotel", "hotel", [SlotValue("hotel-pricerange", "cheap"), SlotValue("hotel-area", "north")]),
        Example("multi3woz:en:train:d1:1", "multi3woz", "en", "train", "i want italian food",
                "find_restaurant", "restaurant", [SlotValue("restaurant-food", "italian")],
                [Turn("user", "i need a cheap hotel in the north"), Turn("system", "acorn guest house is cheap")],
                {"hotel-pricerange": "cheap", "hotel-area": "north"}),
        Example("multi3woz:en:train:d2:0", "multi3woz", "en", "train", "chinese food in the centre please",
                "find_restaurant", "restaurant",
                [SlotValue("restaurant-food", "chinese"), SlotValue("restaurant-area", "centre")]),
    ]


@pytest.fixture
def woz_test():
    return [
        Example("multi3woz:en:test:d9:2", "multi3woz", "en", "test",
                "Can you find me a restaurant in the same area? Indian food.",
                "find_restaurant", "restaurant",
                [SlotValue("restaurant-area", "north", None, None, "history"),
                 SlotValue("restaurant-food", "indian", "Indian", (47, 53))],
                [Turn("user", "I am looking for an expensive hotel in the north."),
                 Turn("system", "The Gonville Hotel is a nice place."),
                 Turn("user", "Book the gonvile hotel please."),
                 Turn("system", "Done. Anything else?")],
                {"hotel-pricerange": "expensive", "hotel-area": "north", "hotel-name": "gonville hotel"}),
    ]
