"""Registre et garantie de non-fuite : aucun juge ne lit example.slots (ni intent, state_before, meta)."""
import pytest

from laya_slots.judges import JUDGES, LayaJudge, build_judge
from laya_slots.schema import Turn

from .fake_laya import FakeRouter
from .helpers import GuardedExample, example, request
from .test_llm_judge import FakeBackend

CONFIGS = [
    ({"type": "first"}, {}),
    ({"type": "random", "seed": 3}, {}),
    ({"type": "generator_score"}, {}),
    ({"type": "string_match"}, {}),
    ({"type": "majority"}, {}),
    ({"type": "laya", "mode": "single"}, {"predictor": FakeRouter()}),
    ({"type": "laya", "mode": "two_step", "expressed_qtype": "choice", "rotations": 2,
      "overflow": "tournament", "max_options": 2}, {"predictor": FakeRouter()}),
    ({"type": "laya", "mode": "two_step"}, {"predictor": FakeRouter()}),
    ({"type": "llm", "few_shot": []}, {"backend": FakeBackend()}),
]


def test_guard_works():
    ex = example()
    for name in ("slots", "intent", "state_before", "meta"):
        with pytest.raises(AssertionError):
            getattr(ex, name)


@pytest.mark.parametrize("config,objects", CONFIGS, ids=lambda c: str(c.get("type", "")) if isinstance(c, dict) else "")
def test_no_judge_reads_gold(config, objects):
    judge = build_judge(config, **objects)
    ex = example(utterance="from boston to denver",
                 history=[Turn("user", "hi"), Turn("system", "where to?")])
    assert isinstance(ex, GuardedExample)
    reqs = [request(ex=ex), request(ex=ex, values=()), request(ex=ex, values=("a", "b", "c", "denver"),
                                                               scores=[0.1, 0.2, 0.3, 0.4])]
    reqs[0].filled = {"fromloc.city_name": "boston"}
    outs = judge.judge_batch(reqs)
    assert len(outs) == 3
    for req, out in zip(reqs, outs):
        assert out.choice is None or 0 <= out.choice < len(req.candidates)
        assert out.probs is None or len(out.probs) == len(req.candidates) + 1


def test_registry():
    assert set(JUDGES) == {"laya", "llm", "first", "random", "generator_score", "string_match",
                           "majority"}
    j = build_judge({"type": "laya", "mode": "two_step", "checkpoint": "lang"}, predictor=FakeRouter())
    assert isinstance(j, LayaJudge) and j.mode == "two_step" and j.name == "laya-two_step-lang"
    with pytest.raises(ValueError):
        build_judge({"type": "oracle"})


def test_registry_accepts_omegaconf():
    from omegaconf import OmegaConf
    j = build_judge(OmegaConf.create({"type": "random", "seed": 7}))
    assert j.seed == 7
