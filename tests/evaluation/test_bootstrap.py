"""Bootstrap par grappes, test de permutation apparié, Holm, et la commande experiments.intervalles."""
import json
import random

import numpy as np
import pytest

from experiments import intervalles as I
from experiments import tables as T
from laya_slots.evaluation import bootstrap as B
from laya_slots.evaluation import metrics as M
from laya_slots.schema import NONE_VALUE


def _rec(ex, slot, cands, choice, gold):
    return {"example_id": ex, "slot": slot, "candidates": [{"value": v} for v in cands],
            "choice_value": choice, "gold_values": gold}


def _system(n, acc, seed, fp_rate=0.0, prefix="multiatis:en:test"):
    """n énoncés, 2 slots chacun : « to » exprimé (choix juste avec proba acc), « cost » non exprimé."""
    rng = random.Random(seed)
    recs = []
    for i in range(n):
        ex = f"{prefix}:{i}"
        recs.append(_rec(ex, "to", ["denver", "miami"], "denver" if rng.random() < acc else "miami", ["denver"]))
        recs.append(_rec(ex, "cost", ["cheap"], "cheap" if rng.random() < fp_rate else NONE_VALUE, []))
    return recs


def _gold(recs):
    gold = {}
    for r in recs:
        gold.setdefault(r["example_id"], []).extend((r["slot"], v) for v in M.gold_values_of(r))
    return gold


def test_point_estimates_match_metrics():
    recs = _system(50, 0.8, 1, fp_rate=0.1) + [
        _rec("multiatis:en:test:x", "to", ["miami"], NONE_VALUE, ["denver"]),   # gold absente, abstention
        _rec("multiatis:en:test:y", "to", ["miami"], "miami", ["denver"])]       # gold absente, erreur
    ids, st = B.cluster_stats(recs)
    pt = B.point_estimates(st)
    gold, pred = _gold(recs), {}
    for r in recs:
        if r["choice_value"] != NONE_VALUE:
            pred.setdefault(r["example_id"], []).append((r["slot"], r["choice_value"]))
    prf, dec = M.slot_prf(gold, pred), M.error_decomposition(recs, gold)
    assert pt["slot_f1"] == pytest.approx(prf["slot_f1"])
    assert pt["slot_precision"] == pytest.approx(prf["slot_precision"])
    assert pt["frame_accuracy"] == pytest.approx(M.frame_accuracy(gold, pred))
    assert pt["choice_accuracy_given_present"] == pytest.approx(dec["choice_accuracy_given_present"])
    assert pt["false_positive_rate"] == pytest.approx(dec["false_positive_rate"])
    assert pt["absent_abstention_rate"] == pytest.approx(dec["absent_abstention_rate"]) == 0.5
    assert len(ids) == 52


def test_clusters_are_dialogues_for_multi3woz():
    assert B.cluster_of("multi3woz:fr:test:MUL0004:3") == "multi3woz:fr:test:MUL0004"
    assert B.cluster_of("multiatis:en:test:12") == "multiatis:en:test:12"
    recs = [_rec(f"multi3woz:en:test:D{d}:{t}", "to", ["a", "b"], "a", ["a"]) for d in range(3) for t in range(4)]
    ids, st = B.cluster_stats(recs)
    assert ids == [f"multi3woz:en:test:D{d}" for d in range(3)] and st[:, B._I["frames"]].tolist() == [4, 4, 4]


def test_bootstrap_ci_contains_point_and_is_deterministic():
    _, st = B.cluster_stats(_system(300, 0.8, 2, fp_rate=0.05))
    ci = B.bootstrap_ci(st, n_boot=500, seed=3)
    for m in ("slot_f1", "choice_accuracy_given_present", "false_positive_rate"):
        assert ci[m]["lo"] <= ci[m]["value"] <= ci[m]["hi"]
    assert 0.03 < ci["choice_accuracy_given_present"]["hi"] - ci["choice_accuracy_given_present"]["lo"] < 0.15
    assert B.bootstrap_ci(st, n_boot=500, seed=3) == ci
    assert ci["absent_abstention_rate"]["value"] is None                         # aucune gold absente


def test_paired_test_detects_difference_and_not_noise():
    a, b, a2 = _system(400, 0.95, 4), _system(400, 0.70, 5), _system(400, 0.95, 4)
    sa, sb = B.align(*B.cluster_stats(a), *B.cluster_stats(b))
    res = B.paired_test(sa, sb, ["choice_accuracy_given_present"], n_perm=2000, n_boot=500, seed=0)
    r = res["choice_accuracy_given_present"]
    assert r["diff"] > 0.15 and r["lo"] > 0 and r["p"] < 0.001
    same = B.paired_test(*B.align(*B.cluster_stats(a), *B.cluster_stats(a2)), ["slot_f1"], n_perm=500, seed=0)
    assert same["slot_f1"]["diff"] == 0 and same["slot_f1"]["p"] == 1.0


def test_align_requires_common_clusters():
    ia, sa = B.cluster_stats(_system(5, 1.0, 0, prefix="multiatis:en:test"))
    ib, sb = B.cluster_stats(_system(5, 1.0, 0, prefix="mtop:en:test"))
    with pytest.raises(ValueError, match="aucune grappe commune"):
        B.align(ia, sa, ib, sb)


def test_holm():
    adj = B.holm([0.01, 0.04, None, 0.03])
    assert adj[0] == pytest.approx(0.03) and adj[3] == pytest.approx(0.06) and adj[1] == pytest.approx(0.06)
    assert adj[2] is None
    assert B.holm([0.5, 0.6]) == [1.0, 1.0]


def _write_run(d, recs, status="complete"):
    d.mkdir(parents=True)
    (d / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    (d / "metrics.json").write_text(json.dumps({"status": status, "name": d.name, "condition": "C2",
                                                "dataset": "multiatis", "lang": "en", "seed": 13, "slot_f1": 0.5}))


def test_cli_writes_ci_and_compares(tmp_path, capsys):
    _write_run(tmp_path / "a", _system(200, 0.95, 1))
    _write_run(tmp_path / "b", _system(200, 0.70, 2))
    _write_run(tmp_path / "c", _system(200, 0.70, 2), status="error")
    assert I.main([str(tmp_path), "--n-boot", "200"]) == 0
    assert (tmp_path / "a" / "ci.json").exists() and not (tmp_path / "c" / "ci.json").exists()
    row = next(r for r in T.load_rows([tmp_path]) if r["name"] == "a")
    assert row["ci.slot_f1.lo"] <= row["ci.slot_f1.hi"]
    out = tmp_path / "cmp.csv"
    assert I.main(["--comparer", str(tmp_path / "a"), str(tmp_path / "b"), "--metrics", "slot_f1,slot_recall",
                   "--n-perm", "500", "--n-boot", "200", "--csv", str(out)]) == 0
    lines = out.read_text().splitlines()
    assert lines[0].startswith("run_a,run_b,metric,a,b,diff") and len(lines) == 3
    assert "p_holm" in lines[0] and "slot_recall" in capsys.readouterr().out


def test_unknown_metric_rejected():
    with pytest.raises(SystemExit):
        I.main(["--comparer", "x", "y", "--metrics", "bleu"])
    assert np.isnan(B._ratio(np.array(1.0), np.array(0.0)))
