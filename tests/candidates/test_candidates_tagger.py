import pytest

from laya_slots.candidates import TaggerGenerator, build_generator
from laya_slots.candidates.tagger import OUTSIDE, decode, word_tags
from laya_slots.schema import SlotValue


class FakeTagger:
    """Imite TaggerModel.predict : étiquettes fixées par mot, probabilité constante."""

    def __init__(self, tags_by_word, prob=0.9):
        self.tags_by_word = tags_by_word
        self.prob = prob
        self.calls = 0

    def predict(self, words):
        self.calls += 1
        return [(self.tags_by_word.get(w, OUTSIDE), self.prob) for w in words]


def test_word_tags_bio_et_effectifs():
    text = "flights from new york to l'aéroport"
    slots = [SlotValue("fromloc.city_name", "new york", "new york", (13, 21)),
             SlotValue("toloc", "aéroport", "aéroport", (27, 35)),
             SlotValue("ctx", "x", None, None, origin="context")]
    words, tags, st = word_tags(text, slots)
    assert [w for w, _, _ in words] == ["flights", "from", "new", "york", "to", "l'", "aéroport"]
    assert tags == ["O", "O", "B-fromloc.city_name", "I-fromloc.city_name", "O", "O", "B-toloc"]
    assert st == {"spans": 2, "conflicts": 0, "misaligned": 0, "empty": 0}


def test_word_tags_conflit_et_span_mal_aligne():
    text = "boston flights"
    slots = [SlotValue("a", "boston", "boston", (0, 6)), SlotValue("b", "boston", "boston", (0, 6)),
             SlotValue("c", "light", "light", (8, 13))]
    _, tags, st = word_tags(text, slots)
    assert tags == ["B-a", "B-c"]
    assert st["conflicts"] == 1 and st["misaligned"] == 1


def test_decode_tolerant():
    tags = ["B-a", "I-a", "O", "I-b", "I-b", "B-b", "I-a"]
    probs = [1.0, 0.5, 1.0, 0.8, 0.6, 0.9, 0.7]
    assert decode(tags, probs) == [("a", 0, 1, 0.75), ("b", 3, 4, pytest.approx(0.7)), ("b", 5, 5, 0.9),
                                   ("a", 6, 6, 0.7)]
    assert decode([OUTSIDE] * 3) == []


def test_generateur_spans_offsets_et_cache(make_atis, atis_ontology, lock_example):
    ex = lock_example(make_atis(1, "flights from New York to denver", []))
    fake = FakeTagger({"New": "B-fromloc.city_name", "York": "I-fromloc.city_name", "denver": "B-toloc.city_name"})
    gen = TaggerGenerator(model=fake)
    frm = gen.generate(ex, atis_ontology.slots["fromloc.city_name"], atis_ontology)
    to = gen.generate(ex, atis_ontology.slots["toloc.city_name"], atis_ontology)
    assert gen.generate(ex, atis_ontology.slots["airline_name"], atis_ontology) == []
    assert [(c.value, c.surface, c.span, c.source) for c in frm] == [("new york", "New York", (13, 21), "tagger")]
    assert frm[0].score == pytest.approx(0.9)
    assert [c.value for c in to] == ["denver"]
    assert fake.calls == 1                         # une passe par exemple


def test_generateur_top_k_et_seuil(make_atis, atis_ontology):
    ex = make_atis(2, "boston or denver", [])
    fake = FakeTagger({"boston": "B-toloc.city_name", "denver": "B-toloc.city_name"})
    spec = atis_ontology.slots["toloc.city_name"]
    assert len(TaggerGenerator(model=fake).generate(ex, spec, atis_ontology)) == 2
    assert len(TaggerGenerator(model=fake, top_k=1).generate(ex, spec, atis_ontology)) == 1
    assert TaggerGenerator(model=fake, min_score=0.95).generate(ex, spec, atis_ontology) == []


def test_generateur_sans_checkpoint():
    with pytest.raises(ValueError):
        TaggerGenerator()
    assert isinstance(build_generator({"name": "tagger", "model": FakeTagger({})}), TaggerGenerator)
