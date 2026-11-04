"""Test réel (modèle Laya en cache, GPU conseillé) : quelques pas de fine-tuning, sauvegarde, rechargement.

    HF_HUB_OFFLINE=1 uv run pytest -m integration tests/train -s
"""
import json

import pytest

from laya_slots.judges import LayaJudge
from laya_slots.schema import Candidate, Example, JudgeRequest, SlotSpec
from laya_slots.train.items import items_from_request

pytestmark = pytest.mark.integration

CITIES = ["boston", "denver", "dallas", "atlanta", "seattle", "miami"]


def _items(judge, n=24):
    items = []
    for i in range(n):
        a, b = CITIES[i % 6], CITIES[(i + 2) % 6]
        ex = Example(id=f"t:{i}", dataset="multiatis", lang="en", split="train",
                     utterance=f"flights from {a} to {b}", intent="atis_flight")
        cands = [Candidate(v, "oracle") for v in (a, b, CITIES[(i + 4) % 6])]
        for slot, desc, gold in (("toloc.city_name", "arrival city", [b]),
                                 ("fromloc.city_name", "departure city", [a]),
                                 ("depart_date.day_name", "departure day of the week", [])):
            req = JudgeRequest(ex, SlotSpec(slot, desc), cands, intent="atis_flight")
            items += items_from_request(judge, req, gold)
    return items


def test_finetune_save_and_reload(tmp_path):
    pytest.importorskip("laya")
    torch = pytest.importorskip("torch")
    from laya_slots.train.trainer import TrainConfig, load_base_agent, train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    judge = LayaJudge(mode="single", seed=1)
    items = _items(judge)
    agent = load_base_agent("multilingual", device=device)
    cfg = TrainConfig(epochs=1, micro_batch=8, lr_encoder=1e-5, lr_head=5e-5, bf16=device == "cuda",
                      grad_checkpointing=False, log_every=1)
    out = tmp_path / "ckpt"
    res = train(agent, items[:48], items[48:], cfg, out, {"test": True})
    assert res["steps"] >= 1 and len(res["history"]) == 2
    for f in ("model.safetensors", "rl_agent_config.json", "tokenizer", "encoder"):
        assert (out / f).exists(), f
    acfg = json.loads((out / "rl_agent_config.json").read_text())
    assert acfg["fine_tuned"] is True and acfg["temperature"] == [1.0, 1.0, 1.0]

    # le checkpoint se recharge par le chemin de l'évaluation et répond
    ft = LayaJudge(mode="single", model_path=str(out), device=device, candidate_order="given")
    reqs = [JudgeRequest(Example(id="e", dataset="multiatis", lang="en", split="test",
                                 utterance="flights from boston to denver"),
                         SlotSpec("toloc.city_name", "arrival city"),
                         [Candidate("boston", "o"), Candidate("denver", "o")], intent="atis_flight")]
    [j] = ft.judge_batch(reqs)
    assert len(j.probs) == 3 and abs(sum(j.probs) - 1) < 1e-3
    assert ft.predictor.loaded() == ["multilingual"]
