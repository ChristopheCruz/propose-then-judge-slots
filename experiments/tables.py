"""Agrège les metrics.json de plusieurs runs en un tableau (CSV et LaTeX booktabs).

    uv run python -m experiments.tables results/ --csv tab.csv --latex tab.tex
    uv run python -m experiments.tables results/2026* --group-by condition,dataset,lang \\
        --columns slot_f1,candidate_recall,latency_ms_p95

Chaque argument est un dossier de run ou un dossier qui en contient (recherche récursive de
metrics.json). Les clés imbriquées se désignent avec un point : decomposition.choice_error_rate.
Avec --group-by, chaque groupe (typiquement les graines d'une condition) donne moyenne ± écart-type.
Un run remplacé (checkpoint refait, bug…) s'écarte des tableaux par un fichier OBSOLETE dans son
dossier, qui en donne la raison ; --all-status le réintègre.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import sys
from pathlib import Path

import yaml

from experiments.config import run_settings

DEFAULT_KEYS = ["name", "condition", "dataset", "lang", "seed"]
DEFAULT_COLUMNS = ["slot_f1", "slot_precision", "slot_recall", "candidate_recall",
                   "decomposition.choice_accuracy_given_present", "latency_ms_p95", "throughput_req_per_s"]


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def find_runs(paths: list[str | Path]) -> list[Path]:
    found: list[Path] = []
    for p in map(Path, paths):
        if (p / "metrics.json").is_file():
            found.append(p / "metrics.json")
        elif p.is_file() and p.name == "metrics.json":
            found.append(p)
        elif p.is_dir():
            found.extend(sorted(p.rglob("metrics.json")))
    return sorted(set(found))


def load_rows(paths: list[str | Path], status: str | None = "complete") -> list[dict]:
    rows = []
    for f in find_runs(paths):
        if status is not None and (f.parent / "OBSOLETE").exists():
            continue
        m = json.loads(f.read_text(encoding="utf-8"))
        cfg_file = f.parent / "config.yaml"
        if "settings" not in m and cfg_file.exists():   # runs antérieurs à settings
            m["settings"] = run_settings(yaml.safe_load(cfg_file.read_text(encoding="utf-8")))
        ci_file = f.parent / "ci.json"
        if ci_file.exists():                              # écrit par experiments.intervalles
            m["ci"] = {k: {"lo": v["lo"], "hi": v["hi"]}
                       for k, v in json.loads(ci_file.read_text(encoding="utf-8"))["metrics"].items()}
        row = flatten(m)
        row["run_dir"] = str(f.parent)
        if status is None or row.get("status") == status:
            rows.append(row)
    return rows


def _num(x) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def aggregate(rows: list[dict], keys: list[str], columns: list[str]) -> list[dict]:
    """Une ligne par groupe : pour chaque colonne, moyenne, écart-type (échantillon) et n."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r.get(k) for k in keys), []).append(r)
    out = []
    for gkey, rs in groups.items():
        row = dict(zip(keys, gkey))
        row["n_runs"] = len(rs)
        for c in columns:
            vals = [v for v in (_num(r.get(c)) for r in rs) if v is not None]
            row[c] = statistics.fmean(vals) if vals else None
            row[c + ".std"] = statistics.stdev(vals) if len(vals) > 1 else None
        out.append(row)
    return out


def _fmt(v, digits: int) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def to_csv(rows: list[dict], columns: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c) for c in columns})
    return buf.getvalue()


_LATEX_ESC = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
              "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}


def latex_escape(s: str) -> str:
    return "".join(_LATEX_ESC.get(ch, ch) for ch in s)


def to_latex(rows: list[dict], keys: list[str], columns: list[str], digits: int = 3,
             with_std: bool = False) -> str:
    """Tableau booktabs (paquet LaTeX booktabs requis)."""
    head = keys + columns
    lines = [r"\begin{tabular}{" + "l" * len(keys) + "r" * len(columns) + "}", r"\toprule",
             " & ".join(latex_escape(h) for h in head) + r" \\", r"\midrule"]
    for r in rows:
        cells = [latex_escape(_fmt(r.get(k), digits)) for k in keys]
        for c in columns:
            v, sd = r.get(c), r.get(c + ".std")
            cell = _fmt(v, digits)
            if with_std and sd is not None:
                cell = f"${cell} \\pm {_fmt(sd, digits)}$"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Agrège les metrics.json de plusieurs runs.")
    ap.add_argument("runs", nargs="+", help="dossiers de runs (ou parents)")
    ap.add_argument("--columns", default=",".join(DEFAULT_COLUMNS))
    ap.add_argument("--keys", default=",".join(DEFAULT_KEYS), help="colonnes d'identification")
    ap.add_argument("--group-by", default=None, help="ex. condition,dataset,lang (moyenne ± écart-type)")
    ap.add_argument("--all-status", action="store_true", help="inclure les runs interrompus ou en échec")
    ap.add_argument("--digits", type=int, default=3)
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--latex", type=Path)
    args = ap.parse_args(argv)

    columns = [c for c in args.columns.split(",") if c]
    rows = load_rows(args.runs, status=None if args.all_status else "complete")
    if not rows:
        print("aucun run trouvé", file=sys.stderr)
        return 1
    if args.group_by:
        keys = [k for k in args.group_by.split(",") if k]
        rows = aggregate(rows, keys, columns)
        keys = keys + ["n_runs"]
        csv_cols = keys + [x for c in columns for x in (c, c + ".std")]
    else:
        keys = [k for k in args.keys.split(",") if k]
        csv_cols = keys + columns + ["run_dir"]
    rows.sort(key=lambda r: tuple(str(r.get(k)) for k in keys))
    csv_text = to_csv(rows, csv_cols)
    latex_text = to_latex(rows, keys, columns, args.digits, with_std=bool(args.group_by))
    if args.csv:
        args.csv.write_text(csv_text, encoding="utf-8")
    if args.latex:
        args.latex.write_text(latex_text, encoding="utf-8")
    if not (args.csv or args.latex):
        sys.stdout.write(csv_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
