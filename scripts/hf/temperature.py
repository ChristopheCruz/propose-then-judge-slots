"""Usage (machine GPU, racine du dépôt) : python3 scripts/hf/temperature.py <graine>

Température du modèle publié : ajustée sur les prédictions dev (graine donnée), listes oracle et réelles.
Cible = position de la valeur gold parmi les candidats, sinon « non précisé » (dernière option)."""
import json, glob, math, random, re, sys
seed = sys.argv[1]
random.seed(0)
rows = []                                    # (log-probs à T = 1, cible, jeu, réel)
for d in sorted(glob.glob(f"results/hf-dev-*-s{seed}-*")):
    name = d.split("/")[-1]
    jeu = re.match(r"hf-dev-(?:reels-)?([a-z0-9]+)-", name).group(1)
    reel = "-reels-" in name
    lines = open(d + "/predictions.jsonl").read().splitlines()
    if len(lines) > 6000:
        lines = random.sample(lines, 6000)  # SGD et Multi3WOZ : échantillon
    for line in lines:
        r = json.loads(line)
        probs = r.get("probs")
        if not probs or len(probs) < 2:
            continue
        cands = r["candidates"]
        vals = re.findall(r"'value': '((?:[^'\\]|\\.)*)'", cands) if isinstance(cands, str) else [c["value"] for c in cands]
        gold = set(r.get("gold_values") or [])
        tgt = next((i for i, v in enumerate(vals) if v in gold), len(probs) - 1)
        if tgt >= len(probs):
            continue
        rows.append(([math.log(max(p, 1e-4)) for p in probs], tgt, jeu, reel))
def scaled(lp, T):
    z = [x / T for x in lp]; m = max(z); e = [math.exp(x - m) for x in z]; s = sum(e)
    return [x / s for x in e]
def nll(T, sel):
    return sum(-math.log(max(scaled(lp, T)[t], 1e-12)) for lp, t, _, _ in sel) / len(sel)
def ece(T, sel, bins=15):
    b = [[0, 0.0, 0] for _ in range(bins)]
    for lp, t, _, _ in sel:
        p = scaled(lp, T); k = max(range(len(p)), key=p.__getitem__)
        i = min(int(p[k] * bins), bins - 1); b[i][0] += 1; b[i][1] += p[k]; b[i][2] += (k == t)
    n = len(sel)
    return sum(abs(c / n - (a / n)) * 0 + abs(a - ok) / n for c, a, ok in b if c)
grid = [round(0.6 + 0.05 * i, 2) for i in range(49)]
best = min(grid, key=lambda T: nll(T, rows))
print(f"n={len(rows)}  T*={best}  NLL {nll(1.0, rows):.4f} -> {nll(best, rows):.4f}  ECE {ece(1.0, rows):.4f} -> {ece(best, rows):.4f}")
for jeu in ["mtop", "multi3woz", "massive", "sgd"]:
    for reel in [False, True]:
        sel = [r for r in rows if r[2] == jeu and r[3] == reel]
        if sel:
            bj = min(grid, key=lambda T: nll(T, sel))
            print(f"{jeu:9} {'real  ' if reel else 'oracle'} n={len(sel):6} ECE T=1 {ece(1.0, sel):.4f}  T*={best} {ece(best, sel):.4f}   (own T*={bj})")
json.dump({"temperature_choice": best, "fitted_on": f"dev, seed {seed}, oracle and real lists, {len(rows)} questions"},
          open(f"results/hf/temperature-s{seed}.json", "w"), indent=2)
