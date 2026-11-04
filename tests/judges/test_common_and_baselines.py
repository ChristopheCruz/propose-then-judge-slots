import pytest

from laya_slots.judges import FirstJudge, GeneratorScoreJudge, MajorityJudge, RandomJudge, StringMatchJudge
from laya_slots.judges._common import judgment_from_probs, option_keys, order_candidates
from laya_slots.schema import Example, SlotValue, write_jsonl

from .helpers import request


def test_option_keys():
    assert option_keys(3) == ["A", "B", "C"]
    with pytest.raises(ValueError):
        option_keys(27)


def test_order_given_score_shuffle():
    r = request(values=("a", "b", "c", "d"), scores=[0.1, None, 0.9, 0.5])
    assert order_candidates(r, "given") == ([0, 1, 2, 3], 0)
    assert order_candidates(r, "score") == ([2, 3, 0, 1], 0)
    idx, _ = order_candidates(r, "shuffle", seed=3)
    assert sorted(idx) == [0, 1, 2, 3]
    assert order_candidates(r, "shuffle", seed=3) == (idx, 0)       # déterministe


def test_truncation_keeps_best_scores_then_applies_order():
    r = request(values=("a", "b", "c", "d"), scores=[0.1, None, 0.9, 0.5])
    assert order_candidates(r, "given", max_options=2) == ([2, 3], 2)
    r2 = request(values=("a", "b", "c", "d"))
    assert order_candidates(r2, "given", max_options=3) == ([0, 1, 2], 1)


def test_judgment_from_probs_none_is_last():
    j = judgment_from_probs([0.2, 0.1, 0.7])
    assert j.choice is None and j.proba == 0.7
    assert judgment_from_probs([0.5, 0.2, 0.3]).choice == 0


def test_first_and_empty():
    assert FirstJudge().judge(request()).choice == 0
    j = FirstJudge().judge(request(values=()))
    assert j.choice is None and j.probs == [1.0]


def test_random_deterministic_and_uniform():
    reqs = [request(id=f"x:{i}") for i in range(20)]
    a = [j.choice for j in RandomJudge(seed=1).judge_batch(reqs)]
    b = [j.choice for j in RandomJudge(seed=1).judge_batch(list(reversed(reqs)))][::-1]
    assert a == b                       # indépendant de l'ordre du lot
    assert len(set(a)) > 1
    j = RandomJudge(seed=1).judge(reqs[0])
    assert j.probs == [0.25] * 4
    assert all(j.choice is not None for j in RandomJudge(seed=1, include_none=False).judge_batch(reqs))


def test_generator_score():
    r = request(scores=[0.2, 0.8, 0.8])
    j = GeneratorScoreJudge().judge(r)
    assert j.choice == 1 and j.proba == 0.8
    assert GeneratorScoreJudge(min_score=0.9).judge(r).choice is None
    assert GeneratorScoreJudge().judge(request()).choice == 0
    assert GeneratorScoreJudge(fallback_first=False).judge(request()).choice is None


def test_string_match_accents_and_words():
    r = request(values=("lyon", "orléans", "paris nord"), utterance="Un vol pour Orleans demain",
                lang="fr")
    assert StringMatchJudge().judge(r).choice == 1
    r = request(values=("paris", "paris nord"), utterance="depart de paris nord")
    assert StringMatchJudge().judge(r).choice == 1            # le plus long
    r = request(values=("den",), utterance="to denver")
    assert StringMatchJudge().judge(r).choice is None         # mots entiers


def _train():
    def ex(i, vals):
        return Example(id=f"t:{i}", dataset="multiatis", lang="en", split="train", utterance="u",
                       intent=None, slots=[SlotValue("toloc.city_name", v) for v in vals])
    return [ex(0, ["denver"]), ex(1, ["denver"]), ex(2, ["dallas"])]


def test_majority(tmp_path):
    j = MajorityJudge().fit(_train())
    out = j.judge(request(values=("boston", "dallas", "denver")))
    assert out.choice == 2 and abs(out.proba - 2 / 3) < 1e-9
    assert j.judge(request(values=("boston",))).choice is None
    j2 = MajorityJudge(abstain_if_unseen=False).fit(_train())
    assert j2.judge(request(values=("boston",))).choice == 0
    write_jsonl(tmp_path / "train.jsonl", _train())
    assert MajorityJudge(train_path=tmp_path / "train.jsonl").judge(request()).choice == 1
