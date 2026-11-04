"""Exemples et requêtes de test, dont un Example qui interdit la lecture des champs gold."""
from __future__ import annotations

from laya_slots.schema import Candidate, Example, JudgeRequest, SlotSpec, SlotValue, Turn

FORBIDDEN = {"slots", "intent", "state_before", "meta"}


class GuardedExample(Example):
    """Example dont la lecture de slots / intent / state_before / meta lève une exception."""

    def __getattribute__(self, name):
        if name in FORBIDDEN:
            raise AssertionError(f"un juge a lu example.{name}")
        return super().__getattribute__(name)


def example(utterance="flights from boston to denver", lang="en", id="multiatis:en:test:1",
            history=None, guarded=True):
    cls = GuardedExample if guarded else Example
    return cls(id=id, dataset="multiatis", lang=lang, split="test", utterance=utterance,
               intent="atis_flight",
               slots=[SlotValue("toloc.city_name", "denver", "denver", (23, 29))],
               history=history or [], state_before={"x": "y"}, meta={"gold": True})


def request(values=("boston", "denver", "dallas"), slot="toloc.city_name",
            description="arrival city", scores=None, **kw):
    ex = kw.pop("ex", None) or example(**kw)
    cands = [Candidate(v, "test", score=None if scores is None else scores[i])
             for i, v in enumerate(values)]
    return JudgeRequest(ex, SlotSpec(slot, description), cands, intent="atis_flight")


__all__ = ["GuardedExample", "Turn", "example", "request"]
