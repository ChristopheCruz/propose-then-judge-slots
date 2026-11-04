"""Figures de l'article générées depuis results/ (pgfplots : .dat + .tex, sans dépendance Python).

    uv run python -m experiments.figures --split dev --out figures/dev
    uv run python -m experiments.figures --split test --out figures/test      # runs finaux

Figures produites (une par jeu × langue quand les runs existent) :
- balayage-<jeu>-<lang>.tex   F1 des slots selon k, pour p_absent = 0 et 0,5 (QR2, H3) ;
                               moyenne ± écart-type sur les graines, une courbe par juge ;
- risque-<jeu>-<lang>.tex     courbes risque-couverture, candidats réels (QR4) : decision.json
                               (experiments.decision) de la graine 13 de chaque juge ;
- cout-<jeu>-<lang>.tex       F1 (point de grille) selon la latence p50 par requête, échelle log
                               (H1) : runs de latence (settings.mesure = latence) ;
- apercu.tex                  document LaTeX autonome qui inclut toutes les figures produites.
Chaque .tex lit ses .dat par chemin relatif : compiler depuis le dossier de sortie, ou \\input avec
\\pgfplotsset{table/search path=…}. Les étiquettes sont en anglais (langue de l'article).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from experiments.tables import load_rows

GRID = {"k": 5, "p_absent": 0.0, "distractors": "hard_no_overlap"}
JUDGE_LABELS = {"C1": "Laya zero-shot", "C2": "Laya fine-tuned (C2)", "C3v2": "Laya FT + augmentation (C3)",
                "C3": "Laya FT + aug. v1"}
STYLES = ["blue,mark=*", "red,mark=square*", "black,mark=triangle*", "orange,mark=diamond*",
          "teal,mark=pentagon*", "violet,mark=x", "brown,mark=o"]


def judge_name(r: dict) -> str:
    if r["condition"] == "C4":
        return (r.get("settings.judge_model") or "LLM").split("/")[-1]
    return r["condition"]


def label(name: str) -> str:
    return JUDGE_LABELS.get(name, f"{name} (LLM judge)")


def is_real(r: dict) -> bool:
    return "Gliner" in (r.get("settings.generators") or "")


def mean_sd(vals: list[float]) -> tuple[float, float]:
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _axis(opts: str, plots: list[str]) -> str:
    return ("\\begin{tikzpicture}\n\\begin{axis}[" + opts + "]\n" + "\n".join(plots)
            + "\n\\end{axis}\n\\end{tikzpicture}\n")


def fig_balayage(rows: list[dict], ds: str, lang: str, out: Path) -> Path | None:
    by = defaultdict(list)                     # (juge, p_absent, k) -> F1 des graines
    for r in rows:
        if (r["dataset"], r["lang"]) != (ds, lang) or is_real(r) or r.get("settings.distractors") != GRID["distractors"]:
            continue
        if r.get("settings.k") is None or r.get("settings.p_absent") not in (0.0, 0.5) or r.get("settings.mesure"):
            continue
        if r.get("settings.train_data") and not r["settings.train_data"].endswith(f"-{lang}"):
            continue                           # le balayage porte sur les modèles monolingues
        by[(judge_name(r), r["settings.p_absent"], r["settings.k"])].append(r["slot_f1"])
    judges = sorted({j for j, _, _ in by})
    series = {(j, p): sorted(k for jj, pp, k in by if (jj, pp) == (j, p)) for j in judges for p in (0.0, 0.5)}
    if not any(len(v) >= 2 for v in series.values()):
        return None
    plots = []
    for i, j in enumerate(judges):
        for p, dash in ((0.0, ""), (0.5, ",dashed")):
            ks = series[(j, p)]
            if len(ks) < 2:
                continue
            dat = out / f"balayage-{ds}-{lang}-{j}-p{int(p * 100)}.dat"
            lines = ["k f1 sd"] + [f"{k} {mean_sd(by[(j, p, k)])[0]:.4f} {mean_sd(by[(j, p, k)])[1]:.4f}" for k in ks]
            _write(dat, "\n".join(lines) + "\n")
            plots.append(f"\\addplot+[{STYLES[i % len(STYLES)]}{dash},error bars/.cd,y dir=both,y explicit] "
                         f"table[x=k,y=f1,y error=sd]{{{dat.name}}};\n"
                         f"\\addlegendentry{{{label(j)}, $p_{{absent}}={p:g}$}}")
    tex = _axis("xlabel={Number of candidates $k$},ylabel={Slot F1},xmode=log,log basis x=2,"
                "xtick={2,5,10,20},xticklabels={2,5,10,20},legend pos=outer north east,"
                f"title={{{ds} {lang.upper()}: robustness to list size and missing gold}},width=9cm,height=6cm", plots)
    return _write(out / f"balayage-{ds}-{lang}.tex", tex)


def fig_risque(rows: list[dict], ds: str, lang: str, out: Path) -> Path | None:
    plots, i = [], 0
    for r in sorted(rows, key=judge_name):
        if (r["dataset"], r["lang"]) != (ds, lang) or not is_real(r) or r["seed"] != 13 or r.get("settings.mesure"):
            continue
        dec = Path(r["run_dir"]) / "decision.json"
        if not dec.exists():
            continue
        d = json.loads(dec.read_text(encoding="utf-8"))
        cov, risk = d["curve"]["coverage"], d["curve"]["risk"]
        name = f"{judge_name(r)}{'-' + r['settings.train_data'] if r.get('settings.train_data') else ''}"
        dat = out / f"risque-{ds}-{lang}-{name}.dat"
        _write(dat, "coverage risk\n" + "\n".join(f"{c:.4f} {k:.4f}" for c, k in zip(cov, risk)) + "\n")
        plots.append(f"\\addplot+[{STYLES[i % len(STYLES)]},mark=none,thick] table[x=coverage,y=risk]{{{dat.name}}};\n"
                     f"\\addlegendentry{{{label(judge_name(r))} (AURC {d['aurc']:.3f})}}")
        i += 1
    if not plots:
        return None
    tex = _axis("xlabel={Coverage (decided without asking)},ylabel={Risk (error rate)},ymin=0,xmin=0,xmax=1,"
                f"legend pos=north west,title={{{ds} {lang.upper()}, real candidates: risk--coverage}},"
                "width=9cm,height=6cm", plots)
    return _write(out / f"risque-{ds}-{lang}.tex", tex)


def fig_cout(rows: list[dict], ds: str, lang: str, out: Path) -> Path | None:
    lat = {}
    for r in rows:
        if (r["dataset"], r["lang"]) == (ds, lang) and r.get("settings.mesure") == "latence" and r.get("latency_ms_p50"):
            lat[judge_name(r)] = r["latency_ms_p50"]
    f1 = defaultdict(list)
    for r in rows:
        if (r["dataset"], r["lang"]) != (ds, lang) or is_real(r) or r.get("settings.mesure"):
            continue
        if (r.get("settings.k"), r.get("settings.p_absent"), r.get("settings.distractors")) != tuple(GRID.values()):
            continue
        if r.get("settings.train_data") and not r["settings.train_data"].endswith(f"-{lang}"):
            continue
        f1[judge_name(r)].append(r["slot_f1"])
    pts = [(j, lat[j], mean_sd(f1[j])) for j in sorted(lat) if f1.get(j)]
    if len(pts) < 2:
        return None
    dat = out / f"cout-{ds}-{lang}.dat"
    _write(dat, "latency f1 sd label\n" + "\n".join(f"{x:.3f} {m:.4f} {s:.4f} {{{label(j)}}}" for j, x, (m, s) in pts) + "\n")
    plots = [(f"\\addplot+[only marks,mark=*,nodes near coords,point meta=explicit symbolic,"
              f"error bars/.cd,y dir=both,y explicit] table[x=latency,y=f1,y error=sd,meta=label]{{{dat.name}}};")]
    tex = _axis("xlabel={Latency per request, p50 (ms, log scale)},ylabel={Slot F1 (grid setting)},xmode=log,"
                f"title={{{ds} {lang.upper()}: accuracy vs. latency}},width=9cm,height=6cm,"
                "nodes near coords style={font=\\scriptsize,anchor=west}", plots)
    return _write(out / f"cout-{ds}-{lang}.tex", tex)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--split", default="dev", help="dev (mise au point) ou test (runs finaux)")
    ap.add_argument("--out", type=Path, default=Path("figures/dev"))
    ap.add_argument("--exclure", default="Qwen/Qwen3-32B",
                    help="modèles LLM à écarter, séparés par des virgules (remplacés le 27/10)")
    args = ap.parse_args(argv)
    exclus = {m for m in args.exclure.split(",") if m}
    rows = [r for r in load_rows([args.results])
            if r.get("split") == args.split and (r.get("settings.limit") is None or r.get("settings.mesure") == "latence")
            and r.get("settings.judge_model") not in exclus]
    args.out.mkdir(parents=True, exist_ok=True)
    made = []
    for ds, lang in sorted({(r["dataset"], r["lang"]) for r in rows}):
        for fig in (fig_balayage, fig_risque, fig_cout):
            path = fig(rows, ds, lang, args.out)
            if path:
                made.append(path)
    body = "\n\\clearpage\n".join(f"\\input{{{p.name}}}" for p in made)
    _write(args.out / "apercu.tex", "\\documentclass{article}\n\\usepackage{pgfplots}\n\\pgfplotsset{compat=1.18}\n"
           "\\begin{document}\n" + body + "\n\\end{document}\n")
    print(f"{len(made)} figures dans {args.out} :", ", ".join(p.name for p in made) or "aucune")
    return 0


if __name__ == "__main__":
    sys.exit(main())
