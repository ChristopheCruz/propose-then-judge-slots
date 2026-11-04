"""experiments.figures : fichiers pgfplots produits depuis des runs factices."""
import json

from experiments import figures as F


def _run(root, name, cond, k, p, seed, f1, real=False, model=None, lat=None, decision=False):
    d = root / name
    d.mkdir(parents=True)
    settings = {"k": None if real else k, "p_absent": None if real else p, "distractors": None if real else "hard_no_overlap",
                "generators": "LexiconGenerator+GlinerGenerator" if real else "OracleGenerator", "limit": 100 if lat else None}
    if model:
        settings["judge_model"] = model
    if lat:
        settings["mesure"] = "latence"
    m = {"status": "complete", "name": name, "condition": cond, "dataset": "mtop", "lang": "en", "split": "dev",
         "seed": seed, "slot_f1": f1, "latency_ms_p50": lat, "settings": settings}
    (d / "metrics.json").write_text(json.dumps(m))
    if decision:
        (d / "decision.json").write_text(json.dumps({"aurc": 0.05, "curve": {"coverage": [0.5, 1.0], "risk": [0.0, 0.1]}}))


def test_figures(tmp_path):
    res = tmp_path / "results"
    for i, (k, p) in enumerate([(2, 0.0), (5, 0.0), (2, 0.5), (5, 0.5)]):
        for s in (13, 14):
            _run(res, f"c2-{i}-{s}", "C2", k, p, s, 0.9 - 0.01 * k - p / 10)
    _run(res, "c4-grille", "C4", 5, 0.0, 13, 0.6, model="google/gemma-4-31B-it")
    _run(res, "c4-vieux", "C4", 5, 0.0, 13, 0.3, model="Qwen/Qwen3-32B")
    _run(res, "c2-reels", "C2", None, None, 13, 0.8, real=True, decision=True)
    _run(res, "lat-c2", "C2", 5, 0.0, 13, 0.0, lat=1.2)
    _run(res, "lat-c4", "C4", 5, 0.0, 13, 0.0, model="google/gemma-4-31B-it", lat=10.0)
    out = tmp_path / "fig"
    assert F.main(["--results", str(res), "--out", str(out)]) == 0
    names = {p.name for p in out.iterdir()}
    assert {"balayage-mtop-en.tex", "risque-mtop-en.tex", "cout-mtop-en.tex", "apercu.tex"} <= names
    cout = (out / "cout-mtop-en.dat").read_text()
    assert "gemma-4-31B-it" in cout and "Qwen3-32B" not in cout and "1.200" in cout
    assert "\\input{balayage-mtop-en.tex}" in (out / "apercu.tex").read_text()
