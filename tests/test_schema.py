import json

from laya_slots.schema import (NONE_VALUE, Candidate, Example, JudgeRequest, Judgment, SlotSpec,
                               SlotValue, Turn, write_jsonl, read_examples)


def _ex():
    return Example(id="multiatis:en:test:1", dataset="multiatis", lang="en", split="test",
                   utterance="flights from boston to denver", intent="atis_flight",
                   slots=[SlotValue("fromloc.city_name", "boston", "boston", (13, 19))],
                   history=[Turn("user", "hi")])


def test_roundtrip_jsonl(tmp_path):
    p = tmp_path / "x.jsonl"
    assert write_jsonl(p, [_ex()]) == 1
    (back,) = list(read_examples(p))
    assert back == _ex()
    json.loads(p.read_text())


def test_judgment_value():
    req = JudgeRequest(_ex(), SlotSpec("toloc.city_name", "arrival city"),
                       [Candidate("denver", "oracle"), Candidate("boston", "distractor")])
    assert Judgment(choice=0, proba=0.9).value(req) == "denver"
    assert Judgment(choice=None, proba=0.6).value(req) == NONE_VALUE
