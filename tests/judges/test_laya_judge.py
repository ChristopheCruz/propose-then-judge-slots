import pytest

from laya_slots.judges import LayaJudge
from laya_slots.schema import Turn

from .fake_laya import FakeRouter
from .helpers import example, request

UTT = "i want to fly to denver tomorrow"


def judge(router=None, **kw):
    kw.setdefault("candidate_order", "given")
    return LayaJudge(predictor=router or FakeRouter(), **kw)


def test_single_picks_candidate_and_full_probs():
    j = judge()
    out = j.judge(request(utterance=UTT))
    assert out.choice == 1
    assert len(out.probs) == 4 and abs(sum(out.probs) - 1) < 1e-6
    assert out.proba == max(out.probs)
    assert out.latency_ms is not None and out.latency_ms >= 0
    assert out.raw["usage"][0]["model"] == "multilingual"


def test_single_none_when_nothing_matches():
    out = judge().judge(request(utterance="show me the fares"))
    assert out.choice is None and out.probs[-1] == out.proba


def test_question_format_neutral_keys_and_none_last():
    router = FakeRouter()
    judge(router).judge(request(utterance=UTT))
    (call,) = router.calls
    (r,) = call
    (q,) = r["questions"].values()
    assert q["type"] == "choice"
    assert list(q["criteria"]) == ["A", "B", "C", "D"]
    assert q["criteria"]["B"] == "denver"
    assert q["criteria"]["D"].startswith("not specified")
    assert "toloc.city_name" in q["instructions"] and "arrival city" in q["instructions"]
    assert r["model"] == "multilingual"


def test_state_uses_request_context_not_gold():
    ex = example(utterance=UTT, history=[Turn("user", "hello"), Turn("system", "where to?")])
    req = request(ex=ex)
    req.intent, req.filled = "book_flight", {"fromloc.city_name": "boston"}
    state = judge().build_state(req)
    assert next(iter(state)) == "last_user_message"
    assert state["intent"] == "book_flight"
    assert "boston" in state["slots_already_filled"]
    assert state["history"] == "user: hello\nsystem: where to?"
    assert "history" not in judge().build_state(request(utterance=UTT))


def test_history_truncated_from_left():
    turns = [Turn("user", f"turn {i}") for i in range(10)]
    state = judge(history_turns=2).build_state(request(ex=example(history=turns)))
    assert state["history"] == "user: turn 8\nuser: turn 9"


def test_empty_candidates_no_call():
    router = FakeRouter()
    out = judge(router).judge(request(values=()))
    assert out.choice is None and out.probs == [1.0] and router.calls == []


def test_grouping_one_laya_request_per_state():
    router = FakeRouter()
    ex = example(utterance=UTT)
    reqs = [request(ex=ex), request(ex=ex, slot="fromloc.city_name", description="departure city"),
            request(ex=example(utterance="to dallas", id="b"))]
    outs = judge(router).judge_batch(reqs)
    (call,) = router.calls
    assert len(call) == 2 and len(call[0]["questions"]) == 2
    assert [o.choice for o in outs] == [1, 1, 2]


def test_two_step_noul():
    router = FakeRouter(noul_fn=lambda state, q: 0.9 if "denver" in state["last_user_message"] else 0.1)
    j = judge(router, mode="two_step")
    out = j.judge(request(utterance=UTT))
    assert out.choice == 1 and out.p_expressed == pytest.approx(0.9)
    assert out.probs[-1] == pytest.approx(0.1) and abs(sum(out.probs) - 1) < 1e-6
    qs = router.calls[0][0]["questions"]
    assert {q["type"] for q in qs.values()} == {"choice", "noul"}
    choice_q = next(q for q in qs.values() if q["type"] == "choice")
    assert len(choice_q["criteria"]) == 3          # pas d'option « non précisé »
    out = j.judge(request(utterance="show me the fares"))
    assert out.choice is None and out.proba == pytest.approx(0.9)


def test_two_step_choice_fallback_and_labels():
    router = FakeRouter()
    j = judge(router, mode="two_step", expressed_qtype="choice")
    out = j.judge(request(utterance=UTT))
    x = next(q for k, q in router.calls[0][0]["questions"].items() if k.endswith("|x"))
    assert x["type"] == "choice" and list(x["criteria"]) == ["A", "B"]
    assert 0 <= out.p_expressed <= 1
    router = FakeRouter()
    judge(router, mode="two_step", noul_labels={"true": "A", "false": "B"}).judge(request(utterance=UTT))
    x = next(q for k, q in router.calls[0][0]["questions"].items() if k.endswith("|x"))
    assert x["labels"] == {"true": "A", "false": "B"}


def test_truncation_documented_in_raw():
    vals = [f"city{i}" for i in range(30)] + ["denver"]
    out = judge(max_options=10).judge(request(values=vals, utterance=UTT))
    assert out.raw["dropped"] == 21 and out.choice is None       # denver tronqué
    scores = [0.0] * 30 + [1.0]
    out = judge(max_options=10).judge(request(values=vals, scores=scores, utterance=UTT))
    assert out.choice == 30 and len(out.probs) == 32


def test_tournament_two_rounds():
    router = FakeRouter()
    vals = [f"city{i}" for i in range(30)] + ["denver"]
    out = judge(router, max_options=10, overflow="tournament").judge(request(values=vals, utterance=UTT))
    assert len(router.calls) == 2
    assert len(router.calls[0][0]["questions"]) == 4                  # 4 paquets de ≤ 10
    assert out.choice == 30 and len(out.probs) == 32 and abs(sum(out.probs) - 1) < 1e-6
    assert set(out.raw["tournament"]) == {"t0", "t1", "t2", "t3"}


def test_rotations_cancel_position_bias():
    bias = [3.0, 0.0, 0.0, 0.0]
    req = request(utterance="show me the fares")
    biased = judge(FakeRouter(position_bias=bias)).judge(req)
    assert biased.choice == 0                                         # le biais l'emporte
    router = FakeRouter(position_bias=bias)
    fair = judge(router, rotations=4).judge(req)
    assert fair.choice is None
    qs = router.calls[0][0]["questions"]
    assert len(qs) == 4 and all("option_order" in q for q in qs.values())


def test_routing_modes_and_token_budget():
    router = FakeRouter()
    judge(router, checkpoint="lang", max_len=2048, head_max_len=384).judge(
        request(utterance="vol pour denver", lang="fr"))
    r = router.calls[0][0]
    assert r["lang"] == "fr" and "model" not in r
    assert r["max_len"] == 2048 and r["head_max_len"] == 384
    router = FakeRouter()
    judge(router, checkpoint="auto").judge(request(utterance=UTT))
    assert "lang" not in router.calls[0][0] and "model" not in router.calls[0][0]


def test_shuffle_order_is_mapped_back():
    router = FakeRouter()
    out = judge(router, candidate_order="shuffle", seed=5).judge(request(utterance=UTT))
    assert out.choice == 1
    (q,) = router.calls[0][0]["questions"].values()
    assert list(q["criteria"].values())[out.raw["order"].index(1)] == "denver"


def test_bad_config():
    with pytest.raises(ValueError):
        LayaJudge(mode="three_step")
    with pytest.raises(ValueError):
        LayaJudge(max_options=40)
