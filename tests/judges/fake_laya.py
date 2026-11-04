"""Faux `laya.Router` qui imite le format de réponse de laya 0.3.22 (vérifié dans le code source).

- `predict_batch(requests, batch_size=None, ...)` : requests = [{"state", "questions", "model"?,
  "lang"?, "max_len"?, "head_max_len"?}] -> une réponse par requête, dans l'ordre ;
- réponse : {"model": "laya-rl-agent", "answers": {qid: ...}, "usage": {...}, "routing": {...}} ;
- choice : {"type", "choice", "probabilities" (arrondies à 4 décimales, indexées par clé),
  "confidence" (1 - entropie normalisée), "answer_confidence" (max p), "action"} ;
- noul : {"type": "noul", "noul": P(vrai), "confidence", "answer_confidence", "action"} ;
- option rendue « clé: description », `option_order` = permutation des positions.

Comportement simulé : une option dont la description apparaît dans le dernier message reçoit un
logit élevé ; l'option dont la description commence par « not specified » un logit moyen ;
`position_bias[s]` est ajouté au logit de la position s (biais de position de Laya).
"""
from __future__ import annotations

import math

QTYPES = ("choice", "score", "noul")


def _check(qid, q):
    # Sous-ensemble des validations de Agent._check_question.
    assert isinstance(q, dict) and q.get("type") in QTYPES, qid
    assert isinstance(q.get("instructions"), str) and q["instructions"].strip(), qid
    crit = q.get("criteria")
    if q["type"] == "choice":
        assert isinstance(crit, (dict, list)) and crit, f"{qid}: choice sans critères"
    if q["type"] == "noul" and crit is not None:
        assert {str(k).lower() for k in crit} <= {"true", "false"}, f"{qid}: clés noul invalides"
    if "labels" in q:
        assert q["type"] == "noul", f"{qid}: labels réservé à noul"
    if "option_order" in q:
        n = 2 if q["type"] == "noul" else len(crit)
        assert sorted(q["option_order"]) == list(range(n)), f"{qid}: option_order invalide"


class FakeRouter:
    def __init__(self, position_bias=None, noul_fn=None, hit_logit=4.0, none_logit=1.5):
        self.position_bias = list(position_bias or [])
        self.noul_fn = noul_fn or (lambda state, q: 0.5)
        self.hit_logit, self.none_logit = hit_logit, none_logit
        self.calls: list[list[dict]] = []

    def _route(self, r):
        if "model" in r:
            return r["model"]
        if "lang" in r:
            return "english" if str(r["lang"]).lower().startswith("en") else "multilingual"
        return "english"

    def _choice(self, state, q):
        keys = list(q["criteria"])
        texts = [str(q["criteria"][k] or "") for k in keys]
        msg = str(state.get("last_user_message", state) if isinstance(state, dict) else state).lower()
        logits = []
        for t in texts:
            if t.startswith("not specified"):
                logits.append(self.none_logit)
            elif t.lower() in msg:
                logits.append(self.hit_logit)
            else:
                logits.append(0.0)
        order = q.get("option_order", list(range(len(keys))))
        for slot, opt in enumerate(order):     # la position `slot` montre l'option `opt`
            if slot < len(self.position_bias):
                logits[opt] += self.position_bias[slot]
        m = max(logits)
        e = [math.exp(x - m) for x in logits]
        p = [x / sum(e) for x in e]
        k = len(p)
        ent = -sum(x * math.log(max(x, 1e-12)) for x in p)
        best = max(range(k), key=lambda i: p[i])
        return {"type": "choice", "choice": keys[best],
                "probabilities": {kk: round(v, 4) for kk, v in zip(keys, p)},
                "confidence": round(1.0 - ent / math.log(k), 4) if k > 1 else 1.0,
                "answer_confidence": round(p[best], 4), "action": {"act_probability": 1.0}}

    def _noul(self, state, q):
        p = float(self.noul_fn(state, q))
        return {"type": "noul", "noul": round(p, 4), "confidence": round(max(p, 1 - p), 4),
                "answer_confidence": round(max(p, 1 - p), 4), "action": {"act_probability": 1.0}}

    def predict_batch(self, requests, batch_size=None, hooks_timeout=None, min_confidence=None,
                      sort_by_length=False):
        self.calls.append(requests)
        out = []
        for r in requests:
            assert r["state"] is not None and isinstance(r["questions"], dict)
            answers = {}
            for qid, q in r["questions"].items():
                _check(qid, q)
                answers[qid] = (self._noul if q["type"] == "noul" else self._choice)(r["state"], q)
            model = self._route(r)
            out.append({"model": "laya-rl-agent", "answers": answers,
                        "usage": {"input_tokens": 100, "output_tokens": 0, "state_tokens": 40,
                                  "state_tokens_dropped": 0, "truncated": False,
                                  "truncated_questions": []},
                        "routing": {"model": model, "reason": "fake"}})
        return out
