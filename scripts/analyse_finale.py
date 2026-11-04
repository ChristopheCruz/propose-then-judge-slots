"""Analyse des runs finaux du test, selon docs/protocole_test.md (§ 3 et § 4).

    .venv/bin/python scripts/analyse_finale.py familles 2 3        # comparaisons appariées + Holm
    .venv/bin/python scripts/analyse_finale.py familles 1          # C2 contre les 3 LLM (après Flash-Next)
    .venv/bin/python scripts/analyse_finale.py decisions           # decision.json (dev -> test), réels, graine 13
    .venv/bin/python scripts/analyse_finale.py tableau             # F1 test moyen ± é.-t. par condition

Familles de Holm (métriques : F1 des slots, taux de faux positifs, précision du choix ; graine 13 contre
graine 13) :
  1. C2 contre chacun des 3 LLM : 6 couples × 2 candidats × 3 LLM × 3 métriques = 108 tests ;
  2. C3 (v2) contre C2 : 6 × 2 × 3 = 36 tests ;
  3. C0 contre le meilleur juge Laya (C2 ou C3, choisi sur le F1 dev, candidats réels) : 6 × 3 = 18 tests.
Sorties : results/analyse/famille-<n>.csv, decisions.csv, tableau-test.csv.
Hors du code_sha256 (scripts/) : n'influence aucun run.
"""
from __future__ import annotations

import csv
import json
import statistics as st
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.intervalles import compare  # noqa: E402

RES = Path("results")
OUT = RES / "analyse"
JEUX = [(d, l) for d in ("multiatis", "mtop", "multi3woz") for l in ("en", "fr")]
METRIQUES = ["slot_f1", "false_positive_rate", "choice_accuracy_given_present"]
LLM = {"Qwen/Qwen3.8-27B": "Qwen3.8-27B", "google/gemma-4-31B-it": "Gemma-4-31B-it",
       "Qwen/Qwen3.8-Flash-Next-FP8": "Qwen3.8-Flash-Next"}


def runs():
    out = []
    for m in RES.glob("*/metrics.json"):
        d = m.parent
        if (d / "OBSOLETE").exists():
            continue
        r = json.loads(m.read_text(encoding="utf-8"))
        if r.get("status") != "complete":
            continue
        s = r.get("settings") or {}
        r.update(dir=d, real="Gliner" in str(s.get("generators")), s=s)
        out.append(r)
    return out


def pick(rows, cond, ds, lang, split, real, seed, model=None):
    """Run de grille unique (k=5, p_absent=0, hard_no_overlap pour l'oracle ; monolingue ; pas de latence)."""
    c = []
    for r in rows:
        s = r["s"]
        if (r["condition"], r["dataset"], r["lang"], r["split"], r["real"], r["seed"]) != (cond, ds, lang, split, real, seed):
            continue
        if s.get("mesure") or s.get("limit") is not None or "-entr-" in r["dir"].name:
            continue
        if split == "test" and not r["dir"].name.startswith("final-"):
            continue
        if grid_point_of_sweep(r["dir"].name):
            continue
        if not real and cond != "C0" and (s.get("k"), s.get("p_absent"), s.get("distractors")) != (5, 0.0, "hard_no_overlap"):
            continue
        if model and s.get("judge_model") != model:
            continue
        c.append(r)
    if not c:
        return None
    return sorted(c, key=lambda r: r["dir"].name)[-1]          # le plus récent


def grid_point_of_sweep(name):
    """final-<cond>-<jeu>-<lang>-s<g>-k5-p0.0-… : point de grille d'un balayage (même config que la grille)."""
    return "-k" in name and "-p" in name


def ecrire(rows, path):
    OUT.mkdir(exist_ok=True)
    if not rows:
        print("rien à écrire :", path)
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("écrit", path, f"({len(rows)} lignes)")


def famille(n, rows):
    paires, labels = [], []
    for ds, lang in JEUX:
        if n == 1:
            for real in (False, True):
                a = pick(rows, "C2", ds, lang, "test", real, 13)
                for model, nom in LLM.items():
                    b = pick(rows, "C4", ds, lang, "test", real, 13, model)
                    if a and b:
                        paires.append((str(a["dir"]), str(b["dir"])))
                        labels.append((ds, lang, "real" if real else "oracle", "C2", nom))
                    else:
                        print("manquant :", ds, lang, real, nom)
        elif n == 2:
            for real in (False, True):
                a, b = pick(rows, "C3v2", ds, lang, "test", real, 13), pick(rows, "C2", ds, lang, "test", real, 13)
                if a and b:
                    paires.append((str(a["dir"]), str(b["dir"])))
                    labels.append((ds, lang, "real" if real else "oracle", "C3", "C2"))
                else:
                    print("manquant :", ds, lang, real)
        elif n == 3:
            # meilleur juge Laya choisi sur le DEV (F1 moyen des graines, candidats réels)
            dev = {c: [r["slot_f1"] for r in rows if (r["condition"], r["dataset"], r["lang"], r["split"], r["real"]) ==
                       (c, ds, lang, "dev", True) and "-entr-" not in r["dir"].name and not r["s"].get("mesure")
                       and r["s"].get("limit") is None] for c in ("C2", "C3v2")}
            best = max(dev, key=lambda c: st.mean(dev[c]) if dev[c] else -1)
            a, b = pick(rows, "C0", ds, lang, "test", False, 13), pick(rows, best, ds, lang, "test", True, 13)
            if a and b:
                paires.append((str(a["dir"]), str(b["dir"])))
                labels.append((ds, lang, "real", "C0", best.replace("v2", "")))
            else:
                print("manquant :", ds, lang, best)
    expected = {1: 36, 2: 12, 3: 6}[n]
    print(f"famille {n} : {len(paires)} paires sur {expected} attendues, {len(paires) * 3} tests")
    if len(paires) < expected:
        print("famille incomplète : la correction de Holm porterait sur une famille tronquée ; arrêt")
        return
    res = compare(paires, METRIQUES, 10000, 1000, 0)
    out = []
    for i, r in enumerate(res):
        ds, lang, cand, a, b = labels[i // len(METRIQUES)]
        out.append({"dataset": ds, "lang": lang, "candidates": cand, "A": a, "B": b, "metric": r["metric"],
                    "value_A": r["a"], "value_B": r["b"], "diff": r["diff"], "ci_lo": r["lo"], "ci_hi": r["hi"],
                    "p": r["p"], "p_holm": r["p_holm"], "run_A": Path(r["run_a"]).name, "run_B": Path(r["run_b"]).name})
    ecrire(out, OUT / f"famille-{n}.csv")


def decisions(rows):
    out = []
    juges = [("C2", None), ("C3v2", None)] + [("C4", m) for m in LLM]
    for ds, lang in JEUX:
        for cond, model in juges:
            dev, test = pick(rows, cond, ds, lang, "dev", True, 13, model), pick(rows, cond, ds, lang, "test", True, 13, model)
            nom = LLM.get(model, cond)
            if not (dev and test):
                print("décision : manquant", ds, lang, nom, "dev" if not dev else "test")
                continue
            if not (test["dir"] / "decision.json").exists():
                subprocess.run([sys.executable, "-m", "experiments.decision", "--dev", str(dev["dir"]),
                                "--test", str(test["dir"])], check=True, capture_output=True)
            d = json.loads((test["dir"] / "decision.json").read_text(encoding="utf-8"))
            row = {"dataset": ds, "lang": lang, "judge": nom, "temperature": d.get("temperature"), "aurc": d.get("aurc"),
                   "ece_raw": (d.get("calibration") or {}).get("brute", {}).get("ece"),
                   "ece_temp": (d.get("calibration") or {}).get("temperature", {}).get("ece"), "run": test["dir"].name}
            for target, pol in sorted((d.get("policies") or {}).items()):
                row[f"coverage@{target}"] = pol.get("coverage")
                row[f"risk@{target}"] = pol.get("risk")
            out.append(row)
    ecrire(out, OUT / "decisions.csv")


def tableau(rows):
    g = defaultdict(list)
    for r in rows:
        if r["split"] != "test" or not r["dir"].name.startswith("final-") or "-entr-" in r["dir"].name:
            continue
        if grid_point_of_sweep(r["dir"].name):
            continue
        s = r["s"]
        if s.get("mesure") or (not r["real"] and r["condition"] != "C0" and
                               (s.get("k"), s.get("p_absent"), s.get("distractors")) != (5, 0.0, "hard_no_overlap")):
            continue
        nom = LLM.get(s.get("judge_model"), r["condition"])
        g[(r["dataset"], r["lang"], "real" if (r["real"] or r["condition"] == "C0") else "oracle", nom)].append(r)
    out = []
    for (ds, lang, cand, nom), rs in sorted(g.items()):
        f1 = [r["slot_f1"] for r in rs]
        dec = [r.get("decomposition") or {} for r in rs]
        mean = lambda k: st.mean([x[k] for x in dec if x.get(k) is not None]) if any(x.get(k) is not None for x in dec) else None  # noqa: E731
        out.append({"dataset": ds, "lang": lang, "candidates": cand, "judge": nom, "n_seeds": len(rs),
                    "slot_f1": st.mean(f1), "slot_f1_sd": st.stdev(f1) if len(f1) > 1 else 0.0,
                    "precision": st.mean([r["slot_precision"] for r in rs]), "recall": st.mean([r["slot_recall"] for r in rs]),
                    "frame_accuracy": st.mean([r["frame_accuracy"] for r in rs]),
                    "candidate_recall": st.mean([r["candidate_recall"] for r in rs if r.get("candidate_recall") is not None] or [float("nan")]),
                    "choice_acc": mean("choice_accuracy_given_present"), "fp_rate": mean("false_positive_rate")})
    ecrire(out, OUT / "tableau-test.csv")


if __name__ == "__main__":
    rows = runs()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tableau"
    if cmd == "familles":
        for n in map(int, sys.argv[2:] or ["2", "3"]):
            famille(n, rows)
    elif cmd == "decisions":
        decisions(rows)
    else:
        tableau(rows)
