import random

import pytest

from laya_slots.candidates import DistractorSampler, OracleGenerator, minimal_normalize
from laya_slots.candidates.base import Normalizer, char_similarity, ngrams, stable_seed, tokens
from laya_slots.schema import NONE_VALUE


def _gen(train, onto, **kw):
    g = OracleGenerator(**kw)
    g.fit(train, onto)
    return g


def _values(cands):
    return [c.value for c in cands]


# --- base ------------------------------------------------------------------
def test_minimal_normalize_and_injection():
    assert minimal_normalize("  San   Francisco\n") == "san francisco"
    calls = []
    n = Normalizer(lambda v, slot=None, lang=None: calls.append((slot, lang)) or v.upper())
    assert n("x", slot="s", lang="fr") == "X" and calls == [("s", "fr")]
    assert Normalizer(str.strip)(" a ", slot="s") == "a"     # signature sans slot/lang


def test_stable_seed_is_stable():
    assert stable_seed(1, "a", "b") == stable_seed(1, "a", "b") != stable_seed(1, "b", "a")
    assert stable_seed(0, "x") == 5043433155909824765  # fixe : ne dépend pas de PYTHONHASHSEED


def test_tokens_and_ngrams_fr():
    toks = [t for t, _, _ in tokens("vols d'Atlanta à l'aéroport")]
    assert toks == ["vols", "d'", "Atlanta", "à", "l'", "aéroport"]
    sp = next(s for s in ngrams("vols d'Atlanta", 2) if s.text == "Atlanta")
    assert (sp.start, sp.end) == (7, 14)
    assert char_similarity("boston", "boston") == 1.0 > char_similarity("boston", "austin")


# --- oracle ----------------------------------------------------------------
def test_oracle_contains_gold_and_has_size_k(atis_train, atis_ontology, atis_test):
    g = _gen(atis_train, atis_ontology, k=4, seed=1)
    ex = atis_test[0]
    cands = g.generate(ex, atis_ontology.slots["toloc.city_name"], atis_ontology)
    vals = _values(cands)
    assert "san francisco" in vals and len(vals) == 4 == len(set(vals))
    assert all(c.source == "oracle" and c.span is None and c.score is None for c in cands)  # blind


def test_oracle_deterministic_and_order_independent(atis_train, atis_ontology, atis_test):
    slot = atis_ontology.slots["fromloc.city_name"]
    a = _gen(atis_train, atis_ontology, k=5, seed=3)
    b = _gen(atis_train, atis_ontology, k=5, seed=3)
    first = _values(a.generate(atis_test[1], slot, atis_ontology))
    for ex in atis_test:                    # d'autres appels entre-temps ne changent rien
        b.generate(ex, slot, atis_ontology)
    assert _values(b.generate(atis_test[1], slot, atis_ontology)) == first
    c = _gen(atis_train, atis_ontology, k=5, seed=4)
    seeds_differ = any(_values(c.generate(ex, slot, atis_ontology)) != _values(a.generate(ex, slot, atis_ontology))
                       for ex in atis_test)
    assert seeds_differ


def test_oracle_unexpressed_slot_only_distractors(atis_train, atis_ontology, atis_test):
    g = _gen(atis_train, atis_ontology, k=3, seed=0, blind=False)
    ex = atis_test[1]                       # pas de compagnie aérienne
    cands = g.generate(ex, atis_ontology.slots["airline_name"], atis_ontology)
    assert cands and all(c.source.startswith("distractor:") for c in cands)
    assert set(_values(cands)) <= {"delta", "united"}     # réservoirs épuisés : liste plus courte


def test_oracle_p_absent_extremes(atis_train, atis_ontology, atis_test):
    slot = atis_ontology.slots["toloc.city_name"]
    always = _gen(atis_train, atis_ontology, k=5, p_absent=1.0, seed=0)
    never = _gen(atis_train, atis_ontology, k=5, p_absent=0.0, seed=0)
    for ex in atis_test:
        gold = [s.value for s in ex.slots if s.name == slot.name]
        vals = _values(always.generate(ex, slot, atis_ontology))
        assert not set(gold) & set(vals) and len(vals) == 5
        assert set(gold) <= set(_values(never.generate(ex, slot, atis_ontology)))


def test_oracle_p_absent_rate_and_nesting(atis_train, atis_ontology, make_atis):
    exs = [make_atis(100 + i, "flights from boston to denver", [("toloc.city_name", "denver")]) for i in range(400)]
    slot = atis_ontology.slots["toloc.city_name"]
    absent = {}
    for p in (0.2, 0.5):
        g = _gen(atis_train, atis_ontology, k=3, p_absent=p, seed=7)
        absent[p] = {ex.id for ex in exs if "denver" not in _values(g.generate(ex, slot, atis_ontology))}
        assert abs(len(absent[p]) / len(exs) - p) < 0.07
    assert absent[0.2] <= absent[0.5]       # emboîtement quand p_absent augmente


def test_oracle_lists_nested_across_k(atis_train, atis_ontology, atis_test):
    slot = atis_ontology.slots["fromloc.city_name"]
    small = _gen(atis_train, atis_ontology, k=3, seed=5, blind=False)
    big = _gen(atis_train, atis_ontology, k=6, seed=5, blind=False)
    for ex in atis_test:
        assert set(_values(small.generate(ex, slot, atis_ontology))) <= set(_values(big.generate(ex, slot, atis_ontology)))


def test_oracle_rejects_bad_params():
    with pytest.raises(ValueError):
        OracleGenerator(k=0)
    with pytest.raises(ValueError):
        OracleGenerator(p_absent=1.5)


# --- distracteurs ----------------------------------------------------------
def test_distractors_no_duplicates_after_normalization(atis_ontology, atis_test, make_atis):
    train = [make_atis(1, "x", [("toloc.city_name", "Boston")], split="train"),
             make_atis(2, "x", [("toloc.city_name", " boston ")], split="train"),
             make_atis(3, "x", [("toloc.city_name", "BOSTON")], split="train"),
             make_atis(4, "x", [("toloc.city_name", "Denver")], split="train")]
    s = DistractorSampler({"strategies": {"train_values": 1, "utterance": 1, "other_slots": 1}}).fit(train)
    ex = atis_test[0]
    out = s.sample(50, ex, atis_ontology.slots["toloc.city_name"], random.Random(0),
                   exclude={"san francisco"}, other_values=["Boston", "thursday"])
    vals = _values(out)
    assert len(vals) == len(set(vals))
    assert "san francisco" not in vals and NONE_VALUE not in vals
    assert {"boston", "denver", "thursday"} <= set(vals)
    assert not {"from", "to", "on"} & set(vals)          # pas de n-grammes de mots vides


@pytest.mark.parametrize("strategy,expected", [
    ("train_values", {"boston", "dallas", "denver", "atlanta", "philadelphia", "pittsburgh"}),
    ("ontology", {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}),
])
def test_distractor_strategies(atis_train, atis_ontology, atis_test, strategy, expected):
    s = DistractorSampler({"strategies": {strategy: 1.0}}).fit(atis_train, atis_ontology)
    slot_name = "fromloc.city_name" if strategy == "train_values" else "depart_date.day_name"
    out = s.sample(20, atis_test[0], atis_ontology.slots[slot_name], random.Random(0))
    assert set(_values(out)) == expected and all(c.source == f"distractor:{strategy}" for c in out)


def test_history_strategy(woz_train, woz_ontology, woz_test):
    s = DistractorSampler({"strategies": {"history": 1.0}}).fit(woz_train, woz_ontology)
    out = s.sample(20, woz_test[0], woz_ontology.slots["restaurant-area"], random.Random(0), exclude={"north"})
    vals = set(_values(out))
    assert {"expensive", "gonville hotel"} <= vals and "north" not in vals


def test_hard_distractors_are_lexically_closer(atis_ontology, atis_test, make_atis):
    cities = ["san diego", "san jose", "san antonio", "santa fe", "oakland", "miami", "tampa", "detroit",
              "chicago", "seattle", "houston", "memphis", "orlando", "phoenix", "toronto", "montreal"]
    train = [make_atis(i, "x", [("toloc.city_name", c)], split="train") for i, c in enumerate(cities)]
    slot = atis_ontology.slots["toloc.city_name"]
    sims = {}
    for sel in ("uniform", "similar"):
        s = DistractorSampler({"strategies": {"train_values": 1}, "selection": sel}).fit(train)
        tot = 0.0
        for seed in range(30):
            out = s.sample(3, atis_test[0], slot, random.Random(seed), exclude={"san francisco"},
                           reference=["san francisco"])
            tot += sum(char_similarity(c.value, "san francisco") for c in out)
        sims[sel] = tot
    assert sims["similar"] > 1.5 * sims["uniform"]


def test_easy_preset_draws_across_slots(atis_train, atis_ontology, atis_test):
    s = DistractorSampler("easy").fit(atis_train, atis_ontology)
    out = s.sample(30, atis_test[0], atis_ontology.slots["toloc.city_name"], random.Random(0))
    assert {"monday", "delta", "economy"} & set(_values(out))


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        DistractorSampler({"strategies": {"nope": 1.0}})


def test_overlaps():
    from laya_slots.candidates.distractors import overlaps
    assert overlaps("on april", "april") and overlaps("las vegas", "to las vegas that")
    assert overlaps("salt lake", "salt lake city") and not overlaps("april", "april")
    assert not overlaps("aprils", "april") and not overlaps("las vegas", "vegas strip")


def test_hard_no_overlap_drops_boundary_variants(atis_train, atis_ontology, make_atis):
    ex = make_atis(90, "i want a flight on april 1 from charlotte to las vegas",
                   [("depart_date.month_name", "april"), ("fromloc.city_name", "charlotte"),
                    ("toloc.city_name", "las vegas")])
    spec = atis_ontology.slots["toloc.city_name"]
    hard = _gen(atis_train, atis_ontology, k=20, distractors="hard", seed=3)
    strict = _gen(atis_train, atis_ontology, k=20, distractors="hard_no_overlap", seed=3)
    from laya_slots.candidates.distractors import overlaps
    variants = lambda g: [v for v in _values(g.generate(ex, spec, atis_ontology)) if overlaps(v, "las vegas")]
    assert variants(hard)                      # « to las vegas », « vegas »… présents en hard
    assert variants(strict) == []
    assert "las vegas" in _values(strict.generate(ex, spec, atis_ontology))
