"""Tests de bout en bout de experiments.run avec de faux composants (CPU, sans réseau)."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import exp_fakes
import pytest
import yaml

from experiments import run as R
from experiments.config import instantiate, load_config
from laya_slots.schema import NONE_VALUE

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


@pytest.fixture
def data_root(tmp_path):
    exp_fakes.write_fixture(tmp_path / "processed")
    return tmp_path / "processed"


def _read_preds(out):
    return [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]


def test_end_to_end(tmp_path, data_root):
    cfg = exp_fakes.write_config(tmp_path / "c.yaml", data_root)
    out = tmp_path / "run"
    assert R.main(["--config", str(cfg), "--output", str(out)]) == 0
    for f in ("config.yaml", "predictions.jsonl", "metrics.json", "run.log"):
        assert (out / f).exists(), f
    m = json.loads((out / "metrics.json").read_text())
    preds = _read_preds(out)
    # slots de l'intention gold : 2 pour atis_flight, 3 pour atis_airfare (4 exemples sur 12)
    assert m["status"] == "complete" and m["n_examples"] == 12
    assert len(preds) == m["n_requests"] == 8 * 2 + 4 * 3
    assert m["slot_f1"] == pytest.approx(1.0)
    assert m["candidate_recall"] == pytest.approx(1.0)
    assert m["decomposition"]["recall_error"] == 0
    assert m["latency_ms_p95"] is not None and m["throughput_req_per_s"] > 0
    assert len(m["data_sha256"]) == 64 and "git_commit" in m
    assert m["settings"]["limit"] is None and m["settings"]["judge"]
    assert len(m["code_sha256"]) == 64 and m["code_sha256"] == R.code_sha256()
    p = preds[0]
    for k in ("example_id", "slot", "candidates", "choice_value", "proba", "gold_value"):
        assert k in p
    resolved = yaml.safe_load((out / "config.yaml").read_text())
    assert resolved["judge"]["batch_size"] == 4 and resolved["requests"]["slots"] == "intent"


def test_overrides_limit_and_ontology_scope(tmp_path, data_root):
    cfg = exp_fakes.write_config(tmp_path / "c.yaml", data_root)
    out = tmp_path / "run"
    R.main(["--config", str(cfg), "--output", str(out), "data.limit=3", "requests.slots=ontology",
            "candidates.generators.0.params.p_absent=1.0"])
    m = json.loads((out / "metrics.json").read_text())
    assert m["n_examples"] == 3 and m["n_requests"] == 9
    # toutes les valeurs gold retirées : erreurs de rappel uniquement
    d = m["decomposition"]
    assert m["candidate_recall"] == 0.0 and d["recall_error"] == d["gold_pairs"] > 0
    assert d["choice_error"] == 0
    assert yaml.safe_load((out / "config.yaml").read_text())["data"]["limit"] == 3


def test_choice_errors_and_false_positives(tmp_path, data_root):
    cfg = exp_fakes.write_config(tmp_path / "c.yaml", data_root,
                                 judge={"class": "exp_fakes.FirstJudge", "batch_size": 5})
    out = tmp_path / "run"
    R.main(["--config", str(cfg), "--output", str(out)])
    m = json.loads((out / "metrics.json").read_text())
    d = m["decomposition"]
    assert d["recall_error"] == 0 and d["choice_error"] > 0
    assert d["false_positive"] == d["unexpressed_requests"] > 0   # FirstJudge ne s'abstient jamais
    assert m["latency_ms_mean"] == pytest.approx(1.0)              # latence fournie par le juge


def test_loader_fit_intent_predictor_and_seed_injection(tmp_path):
    cfg = exp_fakes.write_config(
        tmp_path / "c.yaml", tmp_path,
        data={"loader": {"class": "exp_fakes.FakeLoader", "params": {"n": 6}}, "dataset": "multiatis"},
        candidates={"generators": ["exp_fakes.FakeOracle", "exp_fakes.EmptyGenerator"], "fit": True,
                    "shuffle": True},
        requests={"intent": "predicted"},
        intent_predictor={"class": "exp_fakes.FakeIntentPredictor"})
    out = tmp_path / "run"
    R.main(["--config", str(cfg), "--output", str(out)])
    m = json.loads((out / "metrics.json").read_text())
    assert m["n_examples"] == 6
    assert m["intent_accuracy"] == pytest.approx(4 / 6)   # 2 exemples atis_airfare sur 6
    assert all(p["intent"] == "atis_flight" for p in _read_preds(out))
    assert instantiate({"class": "exp_fakes.FakeOracle"}, seed=99).seed == 99
    assert instantiate({"class": "exp_fakes.FakeOracle", "params": {"seed": 1}}, seed=99).seed == 1


def test_reproducible(tmp_path, data_root):
    cfg = exp_fakes.write_config(tmp_path / "c.yaml", data_root,
                                 candidates={"generators": [{"class": "exp_fakes.FakeOracle",
                                                             "params": {"k": 4, "p_absent": 0.3}}],
                                             "shuffle": True})
    runs = []
    for i in range(2):
        R.main(["--config", str(cfg), "--output", str(tmp_path / f"r{i}")])
        runs.append([(p["candidates"], p["choice_value"]) for p in _read_preds(tmp_path / f"r{i}")])
    assert runs[0] == runs[1]


def test_inherit(tmp_path, data_root):
    base = exp_fakes.write_config(tmp_path / "base.yaml", data_root)
    child = tmp_path / "sub" / "child.yaml"
    child.parent.mkdir()
    child.write_text("inherit: ../base.yaml\nname: enfant\njudge:\n  batch_size: 2\n")
    cfg = load_config(child, ["seed=3"])
    assert cfg.name == "enfant" and cfg.judge.batch_size == 2 and cfg.seed == 3
    assert cfg.judge["class"] == "exp_fakes.LexicalJudge" and base.exists()


def test_slots_for_multi_intent():
    onto = exp_fakes.make_ontology()
    specs, fallback = R.slots_for("atis_flight#atis_airfare", onto, "intent")
    assert [s.name for s in specs] == ["fromloc.city_name", "toloc.city_name", "cost_relative"]
    assert not fallback
    assert R.slots_for("inconnue", onto, "intent")[1]


def test_sigterm_writes_partial_results(tmp_path, data_root):
    """Un SIGTERM pendant le run : le lot en cours finit, metrics.json est écrit, code 143."""
    cfg = exp_fakes.write_config(tmp_path / "c.yaml", data_root,
                                 judge={"class": "exp_fakes.LexicalJudge", "params": {"sleep_s": 0.4},
                                        "batch_size": 2})
    out = tmp_path / "run"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(HERE), str(REPO)])}
    proc = subprocess.Popen([sys.executable, "-m", "experiments.run", "--config", str(cfg),
                             "--output", str(out)], cwd=REPO, env=env)
    preds = out / "predictions.jsonl"
    deadline = time.time() + 60
    while time.time() < deadline and not (preds.exists() and preds.read_text().count("\n") >= 4):
        time.sleep(0.1)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == 128 + signal.SIGTERM
    m = json.loads((out / "metrics.json").read_text())
    assert m["status"] == "interrupted"
    assert 0 < m["n_requests_judged"] < 28 and m["n_examples"] < 12
    # les métriques ne portent que sur les exemples entièrement jugés
    assert m["slot_f1"] == pytest.approx(1.0)
    assert all(p["choice_value"] != NONE_VALUE or p["gold_value"] == NONE_VALUE for p in _read_preds(out))


def test_domain_intent_mode():
    """requests.intent=domain (Multi3WOZ) : le domaine gold sert d'intention, donc de filtre des slots."""
    from omegaconf import OmegaConf

    from laya_slots.schema import Example, IntentSpec, Ontology, SlotSpec
    onto = Ontology("multi3woz", "en", {"hotel": IntentSpec("hotel", "find a hotel", slots=["hotel-area"], domain="hotel")},
                    {"hotel-area": SlotSpec("hotel-area", "area"), "taxi-leaveat": SlotSpec("taxi-leaveat", "leave at")})
    cfg = OmegaConf.create({"requests": {"intent": "domain"}})
    b = R.RequestBuilder(cfg, onto, [])
    exs = [Example(id="multi3woz:en:test:D1:0", dataset="multi3woz", lang="en", split="test", utterance="x",
                   intent=None, domain=d) for d in ("hotel", None)]
    assert b.predict_intents(exs) == ["hotel", None]
    assert [s.name for s in R.slots_for("hotel", onto, "intent")[0]] == ["hotel-area"]
    assert R.slots_for(None, onto, "intent")[1] is True          # domaine inconnu : toute l'ontologie
