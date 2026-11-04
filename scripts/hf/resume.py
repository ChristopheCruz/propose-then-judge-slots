"""Résumé des runs dev du modèle publié par jeu (MASSIVE : langues vues / tenues à l'écart).
Usage (machine GPU, racine du dépôt) : python3 scripts/hf/resume.py <graine>"""
import json, glob, re, statistics as st, sys
seed = sys.argv[1]
rows = {}
for d in glob.glob(f"results/hf-dev-*-s{seed}-*"):
    try:
        m = json.load(open(d + "/metrics.json"))
    except Exception:
        continue
    if m.get("status") != "complete":
        continue
    name = re.sub(r"-\d{8}-\d{6}$", "", d.split("/")[-1])
    reel = "-reels-" in name
    jeu, lang = re.match(rf"hf-dev-(?:reels-)?([a-z0-9]+)-(.+)-s{seed}$", name).groups()
    rows[(jeu, lang, reel)] = m
held = {"sw", "cy", "is", "km", "am"}
def agg(jeu, reel, f=lambda l: True):
    v = [m for (j, l, r), m in rows.items() if j == jeu and r == reel and f(l)]
    if not v:
        return 0, float("nan"), float("nan")
    return len(v), st.mean(m["slot_f1"] for m in v), st.mean(m.get("candidate_recall", float("nan")) for m in v)
for jeu in ["mtop", "multi3woz", "massive", "sgd"]:
    for reel in [False, True]:
        groupes = [("seen", lambda l: l not in held), ("held-out", lambda l: l in held)] if jeu == "massive" else [("", lambda l: True)]
        for lab, f in groupes:
            n, f1, cr = agg(jeu, reel, f)
            print(f"{jeu:9} {lab:9} {'real  ' if reel else 'oracle'} n={n:2} F1={f1:.3f} cand_recall={cr:.3f}")
print("--")
for (j, l, r), m in sorted(rows.items()):
    if j in ("mtop", "multi3woz", "sgd") or (j == "massive" and l in held | {"en", "fr"}):
        print(f"{j:9} {l:6} {'real  ' if r else 'oracle'} F1={m['slot_f1']:.3f} FP={m.get('slot_fp')} abst={m.get('abstention_recall', float('nan')):.3f}")
