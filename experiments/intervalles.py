"""Intervalles de confiance bootstrap des runs, et comparaisons appariées avec correction de Holm.

    uv run python -m experiments.intervalles results/                  # ci.json dans chaque run complet
    uv run python -m experiments.intervalles results/ --force          # recalcule les ci.json existants
    uv run python -m experiments.intervalles --comparer results/c3-… results/c2-… \\
        --comparer results/c3-… results/c4-dev-… --csv comparaisons.csv

- ci.json (à côté de metrics.json) : pour chaque métrique, valeur, bornes à 95 % (percentiles),
  nombre de tirages et de grappes. `experiments.tables` le lit : colonnes `ci.<métrique>.lo|hi`.
- --comparer A B (répétable) : différence A − B sur les grappes communes, intervalle bootstrap
  apparié, p-valeur d'un test de permutation, et p-valeur ajustée par Holm sur l'ensemble des
  comparaisons × métriques demandées.

Grappes : dialogues pour Multi3WOZ, énoncés sinon (laya_slots.evaluation.bootstrap.cluster_of).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from laya_slots.evaluation import bootstrap as B


def read_records(run_dir: Path) -> list[dict]:
    with open(run_dir / "predictions.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_dirs(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if (p / "metrics.json").exists():
            out.append(p)
        elif p.is_dir():
            out += sorted(f.parent for f in p.rglob("metrics.json"))
    return out


def write_ci(run_dir: Path, n_boot: int, seed: int, force: bool) -> str:
    target = run_dir / "ci.json"
    if target.exists() and not force:
        return "existe"
    m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if m.get("status") != "complete" or not (run_dir / "predictions.jsonl").exists():
        return "ignoré"
    _, stats = B.cluster_stats(read_records(run_dir))
    ci = B.bootstrap_ci(stats, n_boot=n_boot, seed=seed)
    target.write_text(json.dumps({"method": "bootstrap percentile 95 %, grappes = dialogues (Multi3WOZ) "
                                            "ou énoncés", "seed": seed, "metrics": ci},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    return "écrit"


def compare(pairs: list[tuple[str, str]], metrics: list[str], n_perm: int, n_boot: int, seed: int) -> list[dict]:
    rows = []
    for a, b in pairs:
        sa, sb = B.align(*B.cluster_stats(read_records(Path(a))), *B.cluster_stats(read_records(Path(b))))
        res = B.paired_test(sa, sb, metrics, n_perm=n_perm, n_boot=n_boot, seed=seed)
        rows += [{"run_a": a, "run_b": b, "metric": m, **res[m]} for m in metrics]
    for r, p in zip(rows, B.holm([r["p"] for r in rows])):
        r["p_holm"] = p
    return rows


def _fmt(x, d=4):
    return "-" if x is None else f"{x:.{d}f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="*", help="dossiers de runs (ou parents) pour écrire ci.json")
    ap.add_argument("--comparer", nargs=2, action="append", metavar=("A", "B"), default=[],
                    help="comparaison appariée A − B (répétable)")
    ap.add_argument("--metrics", default=",".join(B.DEFAULT_METRICS))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="recalculer les ci.json existants")
    ap.add_argument("--csv", type=Path, help="écrire les comparaisons en CSV")
    args = ap.parse_args(argv)
    metrics = [m for m in args.metrics.split(",") if m]
    unknown = set(metrics) - set(B.METRICS)
    if unknown:
        ap.error(f"métriques inconnues : {sorted(unknown)} (connues : {sorted(B.METRICS)})")
    if not args.runs and not args.comparer:
        ap.error("donner des runs, ou --comparer A B")

    if args.runs:
        from collections import Counter
        done = Counter(write_ci(d, args.n_boot, args.seed, args.force) for d in run_dirs(args.runs))
        print("ci.json :", ", ".join(f"{v} {k}" for k, v in sorted(done.items())) or "aucun run")

    if args.comparer:
        rows = compare(args.comparer, metrics, args.n_perm, args.n_boot, args.seed)
        print(f"{'métrique':32} {'A':>8} {'B':>8} {'A−B':>8} {'IC 95 %':>20} {'p':>8} {'p Holm':>8}  A | B")
        for r in rows:
            ci = f"[{_fmt(r['lo'])}, {_fmt(r['hi'])}]"
            print(f"{r['metric']:32} {_fmt(r['a']):>8} {_fmt(r['b']):>8} {_fmt(r['diff']):>8} {ci:>20} "
                  f"{_fmt(r['p']):>8} {_fmt(r['p_holm']):>8}  {Path(r['run_a']).name} | {Path(r['run_b']).name}")
        if args.csv:
            with open(args.csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
