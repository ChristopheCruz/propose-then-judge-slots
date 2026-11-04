"""Items d'entraînement : mêmes questions qu'à l'inférence, cibles correctes, pas de fuite des gold."""
import pytest

from laya_slots.judges import LayaJudge
from laya_slots.judges.laya_judge import EXPRESSED_TRUE, NONE_DESCRIPTION
from laya_slots.schema import Candidate, Example, JudgeRequest, SlotSpec, SlotValue
from laya_slots.train.items import items_from_request

FORBIDDEN = {"slots", "intent", "state_before", "meta"}


class GuardedExample(Example):
    """Example dont les champs gold sont illisibles : le juge ne doit jamais y accéder."""

    def __getattribute__(self, name):
        if name in FORBIDDEN:
            raise AssertionError(f"lecture interdite de example.{name}")
        return super().__getattribute__(name)


def _request(values=("boston", "denver", "dallas"), utterance="flights from boston to denver"):
    ex = GuardedExample(id="multiatis:en:train:1", dataset="multiatis", lang="en", split="train",
                        utterance=utterance, intent="atis_flight",
                        slots=[SlotValue("toloc.city_name", "denver", "denver")])
    return JudgeRequest(ex, SlotSpec("toloc.city_name", "arrival city"),
                        [Candidate(v, "oracle") for v in values], intent="atis_flight")


def _option(item, pos):
    return list(item.question["criteria"].values())[pos]


def test_single_expressed_targets_gold_position():
    judge = LayaJudge(mode="single", seed=3)
    [it] = items_from_request(judge, _request(), ["denver"])
    assert it.kind == "choice" and it.expressed
    assert len(it.question["criteria"]) == 4                    # 3 candidats + « non précisé »
    assert _option(it, it.target) == "denver"
    assert _option(it, -1) == NONE_DESCRIPTION
    assert it.meta == {"example_id": "multiatis:en:train:1", "slot": "toloc.city_name"}


def test_single_question_is_the_inference_question():
    judge = LayaJudge(mode="single", seed=3)
    req = _request()
    [it] = items_from_request(judge, req, ["denver"])
    plan = judge._plan(req)
    assert it.question == plan.round1["v"] and it.state == plan.state


@pytest.mark.parametrize("gold", [[], ["seattle"]])
def test_single_unexpressed_or_absent_gold_targets_none(gold):
    [it] = items_from_request(LayaJudge(mode="single"), _request(), gold)
    assert it.target == 3 and _option(it, it.target) == NONE_DESCRIPTION
    assert it.expressed == bool(gold)


def test_two_step_noul():
    judge = LayaJudge(mode="two_step", expressed_qtype="noul", seed=1)
    x, v = items_from_request(judge, _request(), ["denver"])
    assert (x.kind, x.target, x.question["type"]) == ("expressed", 1, "noul")   # options noul : [false, true]
    assert x.question["criteria"]["true"] == EXPRESSED_TRUE
    assert v.kind == "choice_given" and len(v.question["criteria"]) == 3 and _option(v, v.target) == "denver"
    [x0] = items_from_request(judge, _request(), [])             # non exprimé : pas de question de choix
    assert (x0.kind, x0.target) == ("expressed", 0)


def test_two_step_choice_expressed_question():
    judge = LayaJudge(mode="two_step", expressed_qtype="choice")
    x, _ = items_from_request(judge, _request(), ["denver"])
    assert (x.question["type"], x.target) == ("choice", 0)       # A = exprimé
    [x0] = items_from_request(judge, _request(), [])
    assert x0.target == 1


def test_two_step_absent_gold_keeps_only_expressed_question():
    judge = LayaJudge(mode="two_step")
    [x] = items_from_request(judge, _request(), ["seattle"])
    assert x.kind == "expressed" and x.expressed


def test_rotations_and_tournament_are_inference_only():
    judge = LayaJudge(mode="single", rotations=4, overflow="tournament", max_options=2)
    items = items_from_request(judge, _request(), ["denver"])
    assert len(items) == 1 and "option_order" not in items[0].question
    assert (judge.rotations, judge.overflow) == (4, "tournament")   # le juge d'origine n'est pas modifié


def test_no_candidates_no_item():
    assert items_from_request(LayaJudge(), _request(values=()), ["denver"]) == []


def test_order_follows_candidate_shuffle():
    """Deux graines de juge -> deux ordres : la cible suit la gold."""
    vals = [f"city{i}" for i in range(8)] + ["denver"]
    targets = set()
    for seed in range(6):
        [it] = items_from_request(LayaJudge(seed=seed), _request(values=vals), ["denver"])
        assert _option(it, it.target) == "denver"
        targets.add(it.target)
    assert len(targets) > 1


def test_selection_loss_criteria():
    from laya_slots.train.trainer import selection_loss
    dev = {"all": {"loss": 0.3}, "expressed": {"loss": 0.5}, "unexpressed": {"loss": 0.1},
           "slot_f1": {"f1": 0.9, "tp": 9, "fp": 1, "fn": 1}}
    assert selection_loss(dev, "f1") == pytest.approx(0.1)
    assert selection_loss(dev, "balanced") == pytest.approx(0.3)
    assert selection_loss(dev, "all") == pytest.approx(0.3)
    del dev["slot_f1"]                                   # two_step : pas de question "choice"
    assert selection_loss(dev, "f1") == pytest.approx(0.3)
    assert selection_loss({}, "f1") is None
    with pytest.raises(ValueError):
        selection_loss(dev, "bleu")


def test_confusion_preset_favours_other_slots():
    from laya_slots.candidates.distractors import PRESETS
    c = PRESETS["confusion"]
    assert c.exclude_overlapping and max(c.strategies, key=c.strategies.get) == "other_slots"
    assert "random" not in c.strategies
