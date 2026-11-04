"""Usage (machine GPU, racine du dépôt) : python3 scripts/hf/seuils.py <graine> <T>

Seuils de décision indicatifs (dev, graine donnée, T appliquée) : on décide quand max p ≥ τ (choix d'une valeur
ou de « non précisé ») ; risque = part d'erreurs parmi les décisions ; couverture = part des requêtes décidées."""
import json, glob, math, random, re, sys
seed, T = sys.argv[1], float(sys.argv[2])
random.seed(0)
data = {False: [], True: []}
for d in sorted(glob.glob(f"results/hf-dev-*-s{seed}-*")):
    reel = "-reels-" in d
    lines = open(d + "/predictions.jsonl").read().splitlines()
    if len(lines) > 6000: lines = random.sample(lines, 6000)
    for line in lines:
        r = json.loads(line); probs = r.get("probs")
        if not probs: continue
        cands = r["candidates"]
        vals = re.findall(r"'value': '((?:[^'\\]|\\.)*)'", cands) if isinstance(cands, str) else [c["value"] for c in cands]
        gold = set(r.get("gold_values") or [])
        tgt = next((i for i, v in enumerate(vals) if v in gold), len(probs) - 1)
        z = [math.log(max(p, 1e-4)) / T for p in probs]; m = max(z); e = [math.exp(x - m) for x in z]; s = sum(e)
        p = [x / s for x in e]; k = max(range(len(p)), key=p.__getitem__)
        if k != len(p) - 1:                      # le juge propose une valeur (zone « agir »)
            data[reel].append((p[k], k == tgt))
for reel, rows in data.items():
    rows.sort(key=lambda x: -x[0]); n = len(rows); n_ok = sum(ok for _, ok in rows)
    out = []
    for target in (0.01, 0.05, 0.10):
        err = 0; best = None
        for i, (conf, ok) in enumerate(rows, 1):
            err += (not ok)
            if err / i <= target: best = (conf, (i - err) / n_ok)
        out.append(f"risk {target:.0%}: tau={best[0]:.3f} correct_fills_kept={best[1]:.1%}" if best else f"risk {target:.0%}: none")
    print("real  " if reel else "oracle", "fills", n, f"wrong at tau=0: {1 - n_ok / n:.1%}", " | ".join(out))
