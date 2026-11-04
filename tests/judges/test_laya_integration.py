"""Test réel : charge le checkpoint Laya multilingue (~650 Mo) et juge 4 requêtes EN/FR sur CPU.

    uv sync --extra laya && uv run pytest -m integration tests/judges -s
"""
import time

import pytest

from laya_slots.judges import LayaJudge
from laya_slots.schema import Candidate, Example, JudgeRequest, SlotSpec, Turn

pytestmark = pytest.mark.integration


def _req(id, lang, utt, slot, desc, values, history=(), intent=None, filled=None):
    ex = Example(id=id, dataset="multiatis", lang=lang, split="test", utterance=utt, intent=None,
                 history=[Turn(*t) for t in history])
    return JudgeRequest(ex, SlotSpec(slot, desc), [Candidate(v, "manual") for v in values],
                        intent=intent, filled=filled or {})


REQUESTS = [
    _req("en:1", "en", "show me flights from boston to denver on monday", "toloc.city_name",
         "arrival city", ["boston", "denver", "dallas"], intent="atis_flight"),
    _req("en:2", "en", "what is the cheapest fare to dallas", "depart_date.day_name",
         "departure day of the week", ["monday", "friday"], intent="atis_airfare"),
    _req("fr:1", "fr", "je voudrais un vol de lyon à paris jeudi matin", "fromloc.city_name",
         "ville de départ", ["paris", "lyon", "marseille"], intent="atis_flight"),
    _req("fr:2", "fr", "plutôt le 50 Go alors", "forfait_cible", "forfait souhaité",
         ["Mobile 20 Go à 9,99 €", "Mobile 50 Go à 14,99 €", "Fibre 1 Gb/s à 29,99 €"],
         history=[("user", "je veux un forfait moins cher pour mon portable"),
                  ("system", "vers quel forfait voulez-vous passer ?")],
         intent="changer_forfait"),
]


def test_real_laya_multilingual_cpu():
    pytest.importorskip("laya")
    judge = LayaJudge(checkpoint="multilingual", candidate_order="given", device="cpu")
    t0 = time.perf_counter()
    judge._get_predictor()
    print(f"\nchargement : {time.perf_counter() - t0:.1f} s")
    for mode in ("single", "two_step"):
        judge.mode = mode
        for expressed in (("noul", "choice") if mode == "two_step" else ("-",)):
            judge.expressed_qtype = expressed if expressed != "-" else "noul"
            judge.judge_batch(REQUESTS[:1])                     # échauffement
            t0 = time.perf_counter()
            outs = judge.judge_batch(REQUESTS)
            wall = (time.perf_counter() - t0) * 1000
            print(f"--- {mode} {expressed} : {wall:.0f} ms pour {len(REQUESTS)} requêtes")
            for r, o in zip(REQUESTS, outs):
                value = o.value(r)
                print(f"{r.example.id:5s} {r.slot.name:22s} -> {value!r:28s} proba={o.proba:.3f} "
                      f"p_expr={o.p_expressed} probs={[round(p, 3) for p in o.probs]} "
                      f"lat={o.latency_ms:.0f} ms")
                assert abs(sum(o.probs) - 1) < 1e-3
                assert o.raw["usage"][0]["model"] == "multilingual"
            # un appel unitaire, pour la latence d'une requête isolée
            t0 = time.perf_counter()
            judge.judge(REQUESTS[2])
            print(f"requête isolée : {(time.perf_counter() - t0) * 1000:.0f} ms")
