import numpy as np
import pytest

from laya_slots.candidates import (
    EmbeddingTopK,
    GlinerGenerator,
    HistoryGenerator,
    LexiconGenerator,
    OntologyGenerator,
    OracleGenerator,
    UnionGenerator,
    build_generator,
)
from laya_slots.schema import Candidate, SlotSpec


def _values(cands):
    return [c.value for c in cands]


class FakeGliner:
    """Imite GLiNER.predict_entities : renvoie des entités fixées par étiquette."""

    def __init__(self, entities):
        self.entities = entities
        self.calls = 0

    def predict_entities(self, text, labels, threshold=0.5, **kw):
        self.calls += 1
        return [e for e in self.entities if e["label"] in labels and e["score"] >= threshold]


def _real_generators(train, onto):
    gens = [OntologyGenerator(), OntologyGenerator(mode="all"), HistoryGenerator(all_state=True),
            LexiconGenerator(),
            GlinerGenerator(model=FakeGliner([{"start": 0, "end": 5, "text": "delta", "label": "airline",
                                               "score": 0.9}])),
            EmbeddingTopK(OntologyGenerator(mode="all"), k=2, encoder=lambda xs: np.ones((len(xs), 3)))]
    gens.append(UnionGenerator([LexiconGenerator(), OntologyGenerator(), HistoryGenerator()], k=5))
    for g in gens:
        g.fit(train, onto)
    return gens


# --- règle « pas de fuite » ------------------------------------------------
@pytest.mark.parametrize("data", ["atis", "woz"])
def test_real_generators_never_read_gold(request, data, lock_example):
    train = request.getfixturevalue(f"{data}_train")
    onto = request.getfixturevalue(f"{data}_ontology")
    test = request.getfixturevalue(f"{data}_test")
    for g in _real_generators(train, onto):
        for ex in test:
            locked = lock_example(ex)
            for slot in onto.slots.values():
                g.generate(locked, slot, onto)          # lèverait AssertionError


def test_lock_detects_oracle(atis_train, atis_ontology, atis_test, lock_example):
    g = OracleGenerator()
    g.fit(atis_train, atis_ontology)
    with pytest.raises(AssertionError):
        g.generate(lock_example(atis_test[0]), atis_ontology.slots["toloc.city_name"], atis_ontology)


def test_union_refuses_oracle():
    with pytest.raises(ValueError):
        UnionGenerator([OracleGenerator()])


# --- ontology ----------------------------------------------------------------
def test_ontology_exact_and_fuzzy(atis_ontology, atis_test):
    g = OntologyGenerator(threshold=0.85)
    slot = atis_ontology.slots["depart_date.day_name"]
    exact = g.generate(atis_test[0], slot, atis_ontology)
    assert _values(exact) == ["thursday"] and exact[0].score == 1.0
    assert atis_test[0].utterance[slice(*exact[0].span)] == "Thursday"
    fuzzy = g.generate(atis_test[2], slot, atis_ontology)       # « wenesday » mal orthographié
    assert _values(fuzzy) == ["wednesday"] and 0.85 <= fuzzy[0].score < 1.0
    assert OntologyGenerator(threshold=0.99).generate(atis_test[2], slot, atis_ontology) == []


def test_ontology_multiword_aliases_and_open_slot(woz_ontology, woz_test, atis_ontology, atis_test):
    ex = woz_test[0]
    g = OntologyGenerator(search_history=True)
    names = g.generate(ex, woz_ontology.slots["hotel-name"], woz_ontology)
    assert _values(names) == ["gonville hotel"] and names[0].span is None   # trouvé dans l'historique
    assert g.generate(ex, woz_ontology.slots["restaurant-food"], woz_ontology) == []   # slot ouvert
    al = OntologyGenerator(aliases={"class_type": {"first class": ["1st class"]}})
    from laya_slots.schema import Example
    e = Example("multiatis:en:test:x", "multiatis", "en", "test", "a 1st class seat", "atis_flight")
    assert _values(al.generate(e, atis_ontology.slots["class_type"], atis_ontology)) == ["first class"]
    assert len(OntologyGenerator(mode="all").generate(ex, woz_ontology.slots["hotel-area"], woz_ontology)) == 5


# --- lexicon -----------------------------------------------------------------
def test_lexicon_matches_train_values_with_affinity(atis_train, atis_ontology, atis_test):
    g = LexiconGenerator()
    g.fit(atis_train, atis_ontology)
    ex = atis_test[0]
    to = g.generate(ex, atis_ontology.slots["toloc.city_name"], atis_ontology)
    assert set(_values(to)) == {"boston", "san francisco"}
    by_v = {c.value: c for c in to}
    assert by_v["san francisco"].score == 1.0 > by_v["boston"].score         # boston surtout un départ
    assert ex.utterance[slice(*by_v["boston"].span)] == "Boston"
    assert _values(g.generate(ex, atis_ontology.slots["airline_name"], atis_ontology)) == []
    assert _values(g.generate(ex, SlotSpec("unknown", "x"), atis_ontology)) == []


# --- history -----------------------------------------------------------------
def test_history_state_cross_domain_and_turns(woz_train, woz_ontology, woz_test):
    g = HistoryGenerator()
    g.fit(woz_train, woz_ontology)
    ex = woz_test[0]
    area = g.generate(ex, woz_ontology.slots["restaurant-area"], woz_ontology)
    assert _values(area)[0] == "north" and area[0].score == 0.8              # hotel-area -> restaurant-area
    name = g.generate(ex, woz_ontology.slots["hotel-name"], woz_ontology)
    assert _values(name) == ["gonville hotel"] and name[0].score == 1.0
    price = g.generate(ex, woz_ontology.slots["hotel-pricerange"], woz_ontology)
    assert _values(price) == ["expensive"]
    assert "expensive" not in _values(g.generate(ex, woz_ontology.slots["hotel-area"], woz_ontology))
    assert "expensive" in _values(HistoryGenerator(all_state=True).generate(
        ex, woz_ontology.slots["hotel-area"], woz_ontology))


# --- union -------------------------------------------------------------------
def test_union_dedupes_merges_sources_and_truncates(woz_train, woz_ontology, woz_test):
    u = UnionGenerator([OntologyGenerator(search_history=True), HistoryGenerator()], k=None)
    u.fit(woz_train, woz_ontology)
    out = u.generate(woz_test[0], woz_ontology.slots["hotel-name"], woz_ontology)
    assert _values(out) == ["gonville hotel"] and out[0].source == "ontology+history"
    u3 = UnionGenerator([OntologyGenerator(mode="all")], k=3)
    assert len(u3.generate(woz_test[0], woz_ontology.slots["hotel-area"], woz_ontology)) == 3


def test_union_round_robin_order(atis_ontology, atis_test):
    class Fixed:
        name = "fixed"

        def __init__(self, vals, src):
            self.vals, self.src = vals, src

        def fit(self, *a):
            pass

        def generate(self, *a):
            return [Candidate(v, self.src) for v in self.vals]

    a, b = Fixed(["a1", "a2", "x"], "A"), Fixed(["b1", "x", "b2"], "B")
    slot = atis_ontology.slots["toloc.city_name"]
    ex = atis_test[0]
    assert _values(UnionGenerator([a, b]).generate(ex, slot, atis_ontology)) == ["a1", "b1", "a2", "x", "b2"]
    assert _values(UnionGenerator([a, b], order="priority", k=4).generate(ex, slot, atis_ontology)) == \
        ["a1", "a2", "x", "b1"]


# --- embeddings, gliner (factices) -------------------------------------------
def test_embedding_topk_with_fake_encoder(woz_ontology, woz_test):
    vecs = {"north": [1, 0], "south": [0.8, 0.2], "centre": [0, 1], "east": [-1, 0], "west": [0.1, 0.9]}

    def enc(texts):
        return np.array([vecs.get(t, [1, 0.05]) for t in texts], dtype=float)   # requête ≈ « north »
    g = EmbeddingTopK(OntologyGenerator(mode="all"), k=2, encoder=enc)
    out = g.generate(woz_test[0], woz_ontology.slots["hotel-area"], woz_ontology)
    assert _values(out) == ["north", "south"] and out[0].score > out[1].score


def test_gliner_with_fake_model_caches_per_example(atis_ontology, atis_test):
    ex = atis_test[0]
    fake = FakeGliner([
        {"start": 13, "end": 19, "text": "Boston", "label": "departure city", "score": 0.9},
        {"start": 13, "end": 19, "text": "Boston", "label": "arrival city", "score": 0.4},
        {"start": 23, "end": 36, "text": "San Francisco", "label": "arrival city", "score": 0.8},
        {"start": 0, "end": 7, "text": "Flights", "label": "airline", "score": 0.1},
    ])
    g = GlinerGenerator(model=fake, threshold=0.3)
    to = g.generate(ex, atis_ontology.slots["toloc.city_name"], atis_ontology)
    assert _values(to) == ["san francisco", "boston"] and to[0].span == (23, 36)
    assert _values(g.generate(ex, atis_ontology.slots["fromloc.city_name"], atis_ontology)) == ["boston"]
    assert g.generate(ex, atis_ontology.slots["airline_name"], atis_ontology) == []
    assert fake.calls == 1                                   # une seule prédiction par exemple


def test_build_generator_from_config(atis_train, atis_ontology, atis_test):
    calls = []

    def norm(v, slot=None, lang=None):
        calls.append(slot)
        return " ".join(v.lower().split())
    g = build_generator({"name": "union", "k": 4, "generators": [
        {"name": "lexicon"}, {"name": "ontology", "threshold": 0.9}]}, normalize=norm)
    g.fit(atis_train, atis_ontology)
    assert len(g.generate(atis_test[0], atis_ontology.slots["toloc.city_name"], atis_ontology)) <= 4
    assert calls                                            # la normalisation injectée est utilisée
    o = build_generator({"name": "oracle", "k": 3, "p_absent": 0.2, "seed": 1,
                         "distractors": {"preset": "hard", "temperature": 0.2}})
    assert o.sampler.config.selection == "similar" and o.sampler.config.temperature == 0.2
    with pytest.raises(ValueError):
        build_generator({"name": "nope"})


# --- intégration (modèles réels) ---------------------------------------------
@pytest.mark.integration
def test_gliner_real_model(atis_ontology, atis_test):
    g = GlinerGenerator()
    out = g.generate(atis_test[0], atis_ontology.slots["toloc.city_name"], atis_ontology)
    assert all(isinstance(c, Candidate) for c in out)


@pytest.mark.integration
def test_embedding_topk_real_model(woz_ontology, woz_test):
    g = EmbeddingTopK(OntologyGenerator(mode="all"), k=3)
    assert len(g.generate(woz_test[0], woz_ontology.slots["hotel-area"], woz_ontology)) == 3
