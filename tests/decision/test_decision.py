"""Calibration (température, isotone), risque-couverture et commande experiments.decision."""
import json
import random

import numpy as np
import pytest

from experiments import decision as D
from laya_slots.decision import calibration as C
from laya_slots.decision import risk as K
from laya_slots.schema import NONE_VALUE


def _rec(ex, cands, probs, gold):
    return {"example_id": ex, "slot": "s", "candidates": [{"value": v} for v in cands],
            "choice_value": NONE_VALUE, "probs": probs, "gold_values": gold}


def _overconfident(n, seed=0, acc=0.8):
    """Juge surconfiant : il annonce 0,99 mais n'a raison que dans `acc` des cas."""
    rng = random.Random(seed)
    recs = []
    for i in range(n):
        right = rng.random() < acc
        p = [0.99, 0.005, 0.005] if right else [0.005, 0.99, 0.005]
        recs.append(_rec(f"multiatis:en:dev:{i}", ["a", "b"], p, ["a"]))
    return recs


def test_requests_targets_and_exclusions():
    recs = [_rec("e1", ["a", "b"], [0.7, 0.2, 0.1], ["b"]),        # cible = position de la gold
            _rec("e2", ["a"], [0.3, 0.7], []),                     # non exprimé : « non précisé »
            _rec("e3", ["a"], [0.4, 0.6], ["z"]),                  # gold absente : « non précisé »
            _rec("e4", [], [1.0], []),                             # liste vide : exclue
            {**_rec("e5", ["a"], None, ["a"])}]                    # sans probs : exclue
    r = C.requests_from_records(recs)
    assert len(r) == 3 and r.target.tolist() == [1, 1, 1] and r.expressed.tolist() == [True, False, True]


def test_temperature_fixes_overconfidence():
    reqs = C.requests_from_records(_overconfident(400))
    raw = C.calibration_report(reqs)
    ts = C.TemperatureScaling().fit(reqs)
    cal = C.calibration_report(reqs, ts.temperature)
    assert ts.temperature > 1.5 and cal["nll"] < raw["nll"] and cal["ece"] < raw["ece"]
    assert cal["accuracy"] == raw["accuracy"]                        # la décision ne change pas
    conf, ok = C.IsotonicConfidence().fit(reqs).confidence(reqs)
    assert C.ece(conf, ok) < 0.05


def test_ece_and_brier_perfect():
    reqs = C.requests_from_records([_rec(f"e{i}", ["a"], [1.0, 0.0], ["a"]) for i in range(10)])
    assert C.calibration_report(reqs)["ece"] == pytest.approx(0.0, abs=1e-5)
    assert C.brier(reqs) == pytest.approx(0.0, abs=1e-6)


def test_risk_coverage_and_threshold():
    conf = np.array([0.9, 0.9, 0.8, 0.6, 0.5])
    ok = np.array([1, 1, 1, 0, 1], dtype=float)
    rc = K.risk_coverage(conf, ok)
    assert rc["coverage"] == [0.4, 0.6, 0.8, 1.0]                   # ex aequo à 0,9 groupés
    assert rc["risk"] == pytest.approx([0.0, 0.0, 0.25, 0.2])
    assert rc["aurc"] == pytest.approx(0.2 * 0.25 + 0.2 * 0.2)
    assert K.threshold_for_risk(conf, ok, 0.0) == 0.8
    assert K.threshold_for_risk(conf, ok, 0.2) == 0.5
    assert K.threshold_for_risk(np.array([0.9]), np.array([0.0]), 0.01) is None
    pol = K.apply_threshold(conf, ok, np.array([True, False, True, True, False]), 0.8)
    assert pol["coverage"] == 0.6 and pol["risk"] == 0.0 and pol["ask"] == pytest.approx(0.4)
    assert pol["act"] == pytest.approx(0.4) and pol["ignore"] == pytest.approx(0.2)


def test_cli_cross_fit_and_dev_to_test(tmp_path, capsys):
    for name, seed in (("dev", 1), ("test", 2)):
        d = tmp_path / name
        d.mkdir()
        (d / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in _overconfident(300, seed)))
    assert D.main(["--dev", str(tmp_path / "dev"), "--risques", "0.05,0.3"]) == 0
    res = json.loads((tmp_path / "dev" / "decision.json").read_text())
    assert res["mode"].startswith("2 plis") and res["temperature"] > 1.5
    assert set(res["policies"]) == {"0.05", "0.3"}
    assert D.main(["--dev", str(tmp_path / "dev"), "--test", str(tmp_path / "test")]) == 0
    res = json.loads((tmp_path / "test" / "decision.json").read_text())
    assert res["mode"] == "dev -> test" and res["n_eval"] == 300 and "AURC" in capsys.readouterr().out
