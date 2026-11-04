import json

import pytest

from laya_slots.candidates import (
    LexiconGenerator,
    OracleGenerator,
    candidate_recall,
    candidate_records,
    query_slots,
    report_to_frame,
    summarize,
)
from laya_slots.schema import Candidate


class Fixed:
    """Générateur factice : renvoie une liste fixée par slot."""
    name = "fixed"

    def __init__(self, lists):
        self.lists = lists

    def fit(self, *a):
        pass

    def generate(self, example, slot, ontology):
        return [Candidate(v, s) for v, s in self.lists.get(slot.name, [])]


def test_query_slots_modes(atis_ontology, atis_test):
    ex = atis_test[1]                                   # atis_airfare : 3 slots autorisés
    assert [s.name for s in query_slots(ex, atis_ontology, "gold")] == ["fromloc.city_name", "toloc.city_name"]
    assert len(query_slots(ex, atis_ontology, "intent")) == 3
    assert len(query_slots(ex, atis_ontology, "all")) == 5


def test_recall_counts_by_slot_and_source(atis_ontology, atis_test):
    ex = atis_test[1]                                   # gold : from=denver, to=atlanta
    gen = Fixed({"fromloc.city_name": [("boston", "lexicon"), ("denver", "lexicon+gliner")],
                 "toloc.city_name": [("boston", "gliner")],
                 "class_type": []})
    rep = candidate_recall([ex], gen, atis_ontology, slots="intent", ks=(1, 2))
    o = rep["overall"]
    assert (o["queries"], o["expressed"], o["gold"]) == (3, 2, 2)
    assert o["recall"] == 0.5 and o["gold_absent_rate"] == 0.5
    assert o["recall@1"] == 0.0 and o["recall@2"] == 0.5
    assert o["mean_size"] == 1.0 and o["mean_size_expressed"] == 1.5 and o["mean_size_unexpressed"] == 0.0
    assert o["empty_rate"] == pytest.approx(1 / 3)
    assert rep["by_slot"]["toloc.city_name"]["recall"] == 0.0
    assert rep["by_slot"]["class_type"]["recall"] is None
    src = rep["by_source"]
    assert src["lexicon"] == {"candidates": 2, "hits": 1, "recall": 0.5, "precision": 0.5}
    assert src["gliner"]["hits"] == 1 and src["gliner"]["candidates"] == 2
    df = report_to_frame(rep)
    assert df["slot"].iloc[0] == "__all__" and len(df) == 4


def test_records_are_json_and_summarize_roundtrip(atis_train, atis_ontology, atis_test):
    g = LexiconGenerator()
    g.fit(atis_train, atis_ontology)
    recs = list(candidate_records(atis_test, g, atis_ontology, slots="gold"))
    back = [json.loads(json.dumps(r)) for r in recs]
    assert summarize(back) == summarize(recs)
    r = next(r for r in recs if r["slot"] == "toloc.city_name" and r["example_id"].endswith(":10"))
    assert r["gold"] == ["san francisco"] and r["gold_ranks"][0] is not None


def test_oracle_recall_matches_p_absent(atis_train, atis_ontology, make_atis):
    exs = [make_atis(i, "flights from boston to denver",
                     [("fromloc.city_name", "boston"), ("toloc.city_name", "denver")]) for i in range(300)]
    for p in (0.0, 0.5):
        g = OracleGenerator(k=4, p_absent=p, seed=2)
        g.fit(atis_train, atis_ontology)
        o = candidate_recall(exs, g, atis_ontology, slots="gold")["overall"]
        assert abs(o["gold_absent_rate"] - p) < 0.08 and abs(o["recall"] - (1 - p)) < 0.08
        assert o["mean_size"] == 4
