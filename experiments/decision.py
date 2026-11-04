"""Calibration et décision à trois zones d'un juge (QR4), réglées sur le dev.

    uv run python -m experiments.decision --dev results/c2-dev-… --test results/c2-…
    uv run python -m experiments.decision --dev results/c2-dev-…           # sans test : 2 plis sur le dev

Avec --test : température, régression isotone et seuils (pour chaque risque cible) sont ajustés
sur le dev, puis appliqués au test ; decision.json est écrit dans le run de test.
Sans --test : le dev est coupé en deux moitiés de grappes (dialogues ou énoncés), on ajuste sur
l'une, on évalue sur l'autre, dans les deux sens, et on moyenne ; decision.json va dans le run dev.
Le test n'est donc jamais utilisé pour régler quoi que ce soit.

Contenu de decision.json : calibration (NLL, Brier, ECE) brute / température / isotone ; courbe
risque-couverture et AURC ; pour chaque risque cible, seuil choisi sur le dev et bilan obtenu
(risque, couverture, parts agir / ignorer / redemander).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from experiments.intervalles import read_records
from laya_slots.decision import calibration as C
from laya_slots.decision import risk as K
from laya_slots.evaluation.bootstrap import cluster_of


def _decisions(reqs: C.Requests, temperature: float = 1.0, iso: C.IsotonicConfidence | None = None):
    conf, ok = (iso.confidence(reqs) if iso else C.confidence_and_correct(reqs, temperature))
    is_value = np.asarray([int(np.argmax(p)) != len(p) - 1 for p in reqs.probs], dtype=bool)
    return conf, ok, is_value


def fit_and_evaluate(fit: C.Requests, ev: C.Requests, risks: list[float]) -> dict:
    ts = C.TemperatureScaling().fit(fit)
    iso = C.IsotonicConfidence().fit(fit)
    out = {"n_fit": len(fit), "n_eval": len(ev), "temperature": ts.temperature,
           "calibration": {"brute": C.calibration_report(ev), "temperature": C.calibration_report(ev, ts.temperature)}}
    conf_iso, ok, _ = _decisions(ev, iso=iso)
    out["calibration"]["isotone"] = {"n": len(ev), "ece": C.ece(conf_iso, ok)}
    conf, ok, is_value = _decisions(ev, ts.temperature)
    rc = K.risk_coverage(conf, ok)
    out["aurc"] = rc["aurc"]
    out["risk_full_coverage"] = rc.get("full_coverage_risk")
    out["curve"] = {"coverage": rc["coverage"][::max(1, len(rc["coverage"]) // 200)],
                    "risk": rc["risk"][::max(1, len(rc["risk"]) // 200)]}
    fconf, fok, _ = _decisions(fit, ts.temperature)
    out["policies"] = {}
    for r in risks:
        thr = K.threshold_for_risk(fconf, fok, r)
        out["policies"][f"{r:g}"] = {"target_risk": r, **K.apply_threshold(conf, ok, is_value, thr)}
    return out


def _half(example_id: str) -> int:
    return int(hashlib.sha256(cluster_of(example_id).encode()).hexdigest(), 16) % 2


def _mean(a, b):
    if isinstance(a, dict):
        return {k: _mean(a[k], b[k]) for k in a if k in b}
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return (a + b) / 2
    return a if a == b else None


def cross_fit(reqs: C.Requests, risks: list[float]) -> dict:
    half = np.asarray([_half(i) for i in reqs.example_ids])
    a, b = reqs.subset(half == 0), reqs.subset(half == 1)
    ab, ba = fit_and_evaluate(a, b, risks), fit_and_evaluate(b, a, risks)
    res = _mean({k: v for k, v in ab.items() if k != "curve"}, {k: v for k, v in ba.items() if k != "curve"})
    res["curve"] = ab["curve"]
    res["mode"] = "2 plis sur le dev (grappes)"
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dev", required=True, type=Path, help="run sur le dev (réglage)")
    ap.add_argument("--test", type=Path, help="run sur le test (évaluation) ; absent : 2 plis sur le dev")
    ap.add_argument("--risques", default="0.005,0.01,0.02,0.05", help="risques cibles, séparés par des virgules")
    args = ap.parse_args(argv)
    risks = [float(x) for x in args.risques.split(",") if x]
    dev = C.requests_from_records(read_records(args.dev))
    if not len(dev):
        ap.error(f"{args.dev} : aucune requête avec probabilités (output.save_probs) et candidats")
    if args.test:
        res = {"mode": "dev -> test", "dev": str(args.dev), "test": str(args.test),
               **fit_and_evaluate(dev, C.requests_from_records(read_records(args.test)), risks)}
        target = args.test / "decision.json"
    else:
        res = {"dev": str(args.dev), **cross_fit(dev, risks)}
        target = args.dev / "decision.json"
    target.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    cal = res["calibration"]
    print(f"{res['mode']} : {res['n_eval']:.0f} requêtes jugées, T = {res['temperature']:.3f}")
    for k in ("brute", "temperature"):
        print(f"  {k:12} NLL {cal[k]['nll']:.4f}  Brier {cal[k]['brier']:.4f}  ECE {cal[k]['ece']:.4f}  "
              f"exact. {cal[k]['accuracy']:.4f}")
    print(f"  {'isotone':12} ECE {cal['isotone']['ece']:.4f}")
    print(f"  AURC {res['aurc']:.4f}  (risque à couverture totale {res['risk_full_coverage']:.4f})")
    for p in res["policies"].values():
        risk = "-" if p["risk"] is None else f"{p['risk']:.4f}"
        print(f"  risque cible {p['target_risk']:<6g} -> seuil {p['threshold'] if p['threshold'] is None else round(p['threshold'], 4)}"
              f"  risque obtenu {risk}  couverture {p['coverage']:.3f}  (agir {p['act']:.3f}, ignorer {p['ignore']:.3f},"
              f" redemander {p['ask']:.3f})")
    print(f"-> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
