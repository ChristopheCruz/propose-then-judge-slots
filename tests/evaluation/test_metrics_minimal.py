import pytest

from laya_slots.evaluation import metrics as M
from laya_slots.schema import NONE_VALUE


def test_slot_prf_micro_and_none_excluded():
    gold = {"a": [("from", "boston"), ("to", "denver")], "b": [("from", "dallas")]}
    pred = {"a": [("from", "boston"), ("to", NONE_VALUE)], "b": [("from", "miami")], "c": [("x", "y")]}
    m = M.slot_prf(gold, pred)
    assert (m["slot_tp"], m["slot_fp"], m["slot_fn"]) == (1, 2, 2)
    assert m["slot_precision"] == pytest.approx(1 / 3) and m["slot_recall"] == pytest.approx(1 / 3)
    assert m["slot_f1"] == pytest.approx(1 / 3)


def test_slot_prf_multiset_and_edge_cases():
    gold = {"a": [("from", "boston"), ("from", "boston")]}
    assert M.slot_prf(gold, {"a": [("from", "boston")]})["slot_recall"] == 0.5
    assert M.slot_prf({}, {})["slot_f1"] == 1.0
    assert M.slot_prf(gold, {})["slot_f1"] == 0.0


def test_intent_and_frame_accuracy():
    assert M.intent_accuracy({"a": "x", "b": "y", "c": None}, {"a": "x", "b": "z"}) == 0.5
    assert M.intent_accuracy({}, {}) is None
    gold = {"a": [("s", "1")], "b": []}
    assert M.frame_accuracy(gold, {"a": [("s", "1")], "b": [("s", "2")]}) == 0.5
    assert M.frame_accuracy(gold, {"a": [("s", "1")]}, {"a": "x", "b": "y"}, {"a": "z", "b": "y"}) == 0.5


def _rec(ex, slot, cands, choice, gold):
    return {"example_id": ex, "slot": slot, "candidates": [{"value": v} for v in cands],
            "choice_value": choice, "gold_values": gold}


def test_decomposition():
    gold = {"e1": [("from", "boston"), ("to", "denver")], "e2": [("from", "dallas"), ("date", "monday")]}
    recs = [
        _rec("e1", "from", ["boston", "miami"], "boston", ["boston"]),        # correct
        _rec("e1", "to", ["denver", "miami"], "miami", ["denver"]),           # erreur de choix
        _rec("e2", "from", ["miami"], NONE_VALUE, ["dallas"]),                # erreur de rappel
        _rec("e2", "to", ["miami"], "miami", []),                             # faux positif
        _rec("e1", "cost", ["cheap"], NONE_VALUE, []),                        # abstention correcte
    ]                                                                         # e2/date : non interrogé
    d = M.error_decomposition(recs, gold)
    assert d["gold_pairs"] == 4 and d["correct"] == 1 and d["choice_error"] == 1
    assert d["recall_error"] == 2 and d["not_queried"] == 1
    assert d["false_positive"] == 1 and d["correct_abstention"] == 1
    assert d["choice_accuracy_given_present"] == 0.5 and d["candidate_recall_all"] == 0.5
    # e2/from : gold absente de la liste, le juge s'abstient (bonne réaction) ; e2/date : non interrogé
    assert d["recall_error_abstain"] == 1 and d["absent_abstention_rate"] == 1.0
    cr = M.candidate_recall(recs)
    assert cr["candidate_recall"] == pytest.approx(2 / 3) and cr["n_expressed_requests"] == 3
    a = M.abstention_stats(recs)
    assert a["abstention_precision"] == 0.5 and a["abstention_recall"] == 0.5


def test_gold_value_fallback_and_latency():
    assert M.gold_values_of({"gold_value": NONE_VALUE}) == []
    assert M.gold_values_of({"gold_value": "x"}) == ["x"]
    s = M.latency_stats([1.0, 2.0, 3.0, None])
    assert s["latency_ms_mean"] == 2.0 and s["latency_ms_p50"] == 2.0
    assert M.latency_stats([])["latency_ms_p95"] is None
