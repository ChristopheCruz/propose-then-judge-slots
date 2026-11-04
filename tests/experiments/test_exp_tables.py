import json

import pytest

from experiments import tables as T


def _run(root, name, cond, seed, f1, status="complete"):
    d = root / name
    d.mkdir(parents=True)
    m = {"status": status, "name": name, "condition": cond, "dataset": "multiatis", "lang": "en",
         "seed": seed, "slot_f1": f1, "candidate_recall": 1.0,
         "decomposition": {"choice_accuracy_given_present": f1}, "latency_ms_p95": 2.5}
    (d / "metrics.json").write_text(json.dumps(m))


@pytest.fixture
def runs(tmp_path):
    _run(tmp_path, "a", "C1", 13, 0.80)
    _run(tmp_path, "b", "C1", 14, 0.90)
    _run(tmp_path, "c", "C4_x", 13, 0.95)
    _run(tmp_path, "d", "C4_x", 14, 0.10, status="interrupted")
    return tmp_path


def test_rows_and_csv(runs, tmp_path):
    rows = T.load_rows([runs])
    assert len(rows) == 3 and rows[0]["decomposition.choice_accuracy_given_present"] == 0.80
    assert len(T.load_rows([runs], status=None)) == 4
    out = tmp_path / "t.csv"
    assert T.main([str(runs), "--csv", str(out)]) == 0
    lines = out.read_text().splitlines()
    assert lines[0].startswith("name,condition,dataset,lang,seed,slot_f1") and len(lines) == 4


def test_group_by_and_latex(runs, tmp_path):
    rows = T.aggregate(T.load_rows([runs]), ["condition"], ["slot_f1"])
    c1 = next(r for r in rows if r["condition"] == "C1")
    assert c1["n_runs"] == 2 and c1["slot_f1"] == pytest.approx(0.85)
    assert c1["slot_f1.std"] == pytest.approx(0.0707, abs=1e-3)
    tex = tmp_path / "t.tex"
    T.main([str(runs), "--group-by", "condition", "--columns", "slot_f1", "--latex", str(tex)])
    text = tex.read_text()
    assert r"\toprule" in text and r"\bottomrule" in text and r"C4\_x" in text
    assert r"$0.850 \pm 0.071$" in text


def test_settings_backfilled_from_config(tmp_path):
    """Un run sans `settings` dans metrics.json les retrouve dans son config.yaml."""
    _run(tmp_path, "old", "C1", 13, 0.5)
    (tmp_path / "old" / "config.yaml").write_text(
        "data: {limit: 200}\n"
        "candidates: {generators: [{class: laya_slots.candidates.oracle.OracleGenerator,\n"
        "  params: {k: 5, p_absent: 0.2, distractors: easy}}]}\n"
        "judge: {class: laya_slots.judges.llm_judge.LLMJudge, params: {model: Qwen/Qwen3-32B}}\n")
    row = T.load_rows([tmp_path])[0]
    assert (row["settings.k"], row["settings.p_absent"], row["settings.distractors"]) == (5, 0.2, "easy")
    assert row["settings.limit"] == 200 and row["settings.generators"] == "OracleGenerator"
    assert (row["settings.judge"], row["settings.judge_model"]) == ("LLMJudge", "Qwen/Qwen3-32B")


def test_old_runs_without_fit_are_marked_unfitted(tmp_path):
    """Un run antérieur à `settings` (config.yaml avec fit: false) ressort avec settings.fit = False,
    et ne se regroupe pas avec un run identique ajusté sur le train."""
    cand = "candidates: {generators: [{class: laya_slots.candidates.oracle.OracleGenerator, params: {k: 5}}], "
    for name, fit in (("avant", "false"), ("apres", "true")):
        _run(tmp_path, name, "C1", 13, 0.1 if fit == "false" else 0.6)
        (tmp_path / name / "config.yaml").write_text(cand + f"fit: {fit}}}\njudge: {{class: x.y.LayaJudge}}\n")
    rows = {r["name"]: r for r in T.load_rows([tmp_path])}
    assert rows["avant"]["settings.fit"] is False and rows["apres"]["settings.fit"] is True
    groups = T.aggregate(list(rows.values()), ["condition", "settings.fit"], ["slot_f1"])
    assert len(groups) == 2 and all(g["n_runs"] == 1 for g in groups)


def test_obsolete_runs_are_skipped(runs):
    (runs / "a" / "OBSOLETE").write_text("checkpoint remplacé\n")
    assert {r["name"] for r in T.load_rows([runs])} == {"b", "c"}
    assert "a" in {r["name"] for r in T.load_rows([runs], status=None)}


def test_judge_models_are_separate_groups(tmp_path):
    """make tableaux : deux LLM-juges (C4) au même réglage donnent deux lignes, graines regroupées."""
    import re
    from pathlib import Path
    mk = (Path(__file__).parents[2] / "Makefile").read_text(encoding="utf-8")
    groups = re.search(r"^TABLEAU_GROUPES\s*:=\s*(\S+)", mk, re.MULTILINE).group(1).split(",")
    assert "settings.judge_model" in groups
    for i, (model, seed) in enumerate([("Qwen/Qwen3.8-27B", 13), ("Qwen/Qwen3.8-27B", 14),
                                       ("mistralai/Mistral-Small-4-119B-2603", 13),
                                       ("google/gemma-4-31B-it", 13)]):
        _run(tmp_path, f"r{i}", "C4", seed, 0.5 + i / 10)
        m = json.loads((tmp_path / f"r{i}" / "metrics.json").read_text())
        m["settings"] = {"judge": "LLMJudge", "judge_model": model, "k": 5, "p_absent": 0.0}
        (tmp_path / f"r{i}" / "metrics.json").write_text(json.dumps(m))
    rows = T.load_rows([tmp_path])
    out = {r["settings.judge_model"]: r["n_runs"] for r in T.aggregate(rows, groups, ["slot_f1"])}
    assert out == {"Qwen/Qwen3.8-27B": 2, "mistralai/Mistral-Small-4-119B-2603": 1,
                   "google/gemma-4-31B-it": 1}
