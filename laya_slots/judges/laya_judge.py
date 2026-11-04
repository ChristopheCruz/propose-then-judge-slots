"""Juge Laya : une question `choice` par slot, clés neutres, option « non précisé ».

API vérifiée sur laya 0.3.22 (voir tests/judges/fake_laya.py pour le format imité) :
- `Router.predict_batch([{"state", "questions", "model"|"lang", "max_len", "head_max_len"}, ...],
  batch_size=...)` renvoie, dans l'ordre, `{"answers": {qid: réponse}, "usage": {...},
  "routing": {"model": ...}}` ;
- réponse `choice` : `{"choice": clé, "probabilities": {clé: p}, "confidence": 1 - entropie
  normalisée, "answer_confidence": max(p), "action": {...}}` ; réponse `noul` : `{"noul": P(vrai)}` ;
- une option est rendue « clé: description » ; `option_order` (permutation) change la position
  sans changer les clés ; les probabilités reviennent toujours indexées par clé.

Regroupement : toutes les questions d'une même requête Laya partagent une passe avant. On regroupe
donc les JudgeRequest qui ont le même état (même exemple, même intention, mêmes slots remplis) :
un appel Laya par exemple, une ligne par question.
"""
from __future__ import annotations

import json
import math
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

from ..interfaces import Judge
from ..schema import JudgeRequest, Judgment
from ._common import (
    LANG_NAMES,
    MAX_LETTER_OPTIONS,
    candidate_text,
    clip,
    format_filled,
    format_history,
    judgment_from_probs,
    normalize,
    option_keys,
    order_candidates,
)

CHOICE_INSTRUCTIONS = ('Which value does the user\'s last message give for the slot "{slot}" '
                       "({description})?")
CHOICE_GIVEN_EXPRESSED = ('The user\'s last message gives a value for the slot "{slot}" '
                          "({description}). Which one?")
NONE_DESCRIPTION = "not specified: the last message gives no value for this slot"
EXPRESSED_INSTRUCTIONS = ('Does the user\'s last message give a value for the slot "{slot}" '
                          "({description})?")
EXPRESSED_TRUE = "yes, the last message gives a value for this slot"
EXPRESSED_FALSE = "no, the last message gives no value for this slot"

CHECKPOINTS = ("multilingual", "english", "lang", "auto")


@dataclass
class _Plan:
    """Ce qu'on demande à Laya pour une JudgeRequest."""
    state: dict
    routing: dict
    chunks: list[list[int]]                 # indices de candidats, dans l'ordre de présentation
    dropped: int = 0
    round1: dict[str, dict] = field(default_factory=dict)
    answers: dict[str, dict] = field(default_factory=dict)
    usage: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0


class LayaJudge(Judge):
    """Laya comme juge de slots.

    mode : "single" (une question, « non précisé » en dernière option) ou "two_step"
        (P(slot exprimé), puis choix parmi les candidats seuls, dans la même passe).
    checkpoint : "multilingual" (forcé, défaut), "english" (forcé), "lang" (routage par
        example.lang : en -> english, sinon multilingual) ou "auto" (détection de Laya).
    expressed_qtype : "noul" ou "choice" (repli oui/non à clés neutres A/B, si noul est biaisé).
    noul_labels : étiquettes montrées au modèle pour noul, ex. {"true": "A", "false": "B"}.
    expressed_threshold : en two_step, en dessous on répond « non précisé ».
    max_options : nombre maximal de candidats par question (hors « non précisé »), ≤ 25.
    overflow : "truncate" (on garde max_options candidats, les meilleurs scores générateur s'il y
        en a, sinon les premiers) ou "tournament" (paquets ≤ max_options sans « non précisé »,
        puis finale entre les vainqueurs ; les éliminés ont une probabilité 0).
    candidate_order : "shuffle" (défaut, graine), "given" ou "score".
    rotations : nombre d'ordres circulaires moyennés via `option_order` (1 = ordre unique).
        Laya a un fort biais de position ; les rotations passent dans la même passe avant.
    history_turns, history_max_chars, utterance_max_chars, option_max_chars : troncatures
        côté texte, avant la tokenisation (Laya coupe ensuite l'état à droite et chaque option à
        48 jetons).
    max_len, head_max_len : budget de jetons transmis à Laya (None = valeur du checkpoint :
        1024/256 multilingue, 512/192 anglais).
    temperature : "shipped" (températures du checkpoint) ou "neutral" (T = 1 partout).
    model_path : dossier d'un checkpoint fine-tuné (écrit par laya_slots.train), chargé à la place
        du checkpoint `checkpoint` ("multilingual" ou "english") ; None = checkpoint publié.
    batch_size : nombre maximal d'états par passe avant (None = tout le lot).
    predictor : objet exposant `predict_batch(requests, batch_size=...)` comme `laya.Router` ;
        créé à la demande s'il est absent (pour les tests, on injecte un faux).
    """
    name = "laya"

    def __init__(self, mode: str = "single", checkpoint: str = "multilingual",
                 expressed_qtype: str = "noul", noul_labels: dict | None = None,
                 expressed_threshold: float = 0.5, max_options: int = 15,
                 overflow: str = "truncate", candidate_order: str = "shuffle", rotations: int = 1,
                 seed: int = 0, history_turns: int = 4, history_max_chars: int = 1500,
                 utterance_max_chars: int = 1000, option_max_chars: int = 200,
                 include_intent: bool = True, include_filled: bool = True,
                 max_len: int | None = None, head_max_len: int | None = None,
                 temperature: str = "shipped", batch_size: int | None = None,
                 device: str | None = None, revision: str | None = None,
                 model_path: str | None = None,
                 choice_instructions: str = CHOICE_INSTRUCTIONS,
                 choice_given_expressed: str = CHOICE_GIVEN_EXPRESSED,
                 none_description: str = NONE_DESCRIPTION,
                 expressed_instructions: str = EXPRESSED_INSTRUCTIONS,
                 predictor: Any = None):
        if mode not in ("single", "two_step"):
            raise ValueError(f"mode inconnu : {mode!r}")
        if checkpoint not in CHECKPOINTS:
            raise ValueError(f"checkpoint inconnu : {checkpoint!r} ; choisir parmi {CHECKPOINTS}")
        if expressed_qtype not in ("noul", "choice"):
            raise ValueError(f"expressed_qtype inconnu : {expressed_qtype!r}")
        if overflow not in ("truncate", "tournament"):
            raise ValueError(f"overflow inconnu : {overflow!r}")
        if not 1 <= max_options <= MAX_LETTER_OPTIONS:
            raise ValueError(f"max_options doit être entre 1 et {MAX_LETTER_OPTIONS}")
        if temperature not in ("shipped", "neutral"):
            raise ValueError(f"temperature inconnue : {temperature!r}")
        if model_path is not None and checkpoint not in ("multilingual", "english"):
            raise ValueError("model_path demande checkpoint='multilingual' ou 'english' (un seul modèle)")
        self.mode, self.checkpoint = mode, checkpoint
        self.expressed_qtype, self.noul_labels = expressed_qtype, noul_labels
        self.expressed_threshold = expressed_threshold
        self.max_options, self.overflow = max_options, overflow
        self.candidate_order, self.rotations, self.seed = candidate_order, max(1, rotations), seed
        self.history_turns, self.history_max_chars = history_turns, history_max_chars
        self.utterance_max_chars, self.option_max_chars = utterance_max_chars, option_max_chars
        self.include_intent, self.include_filled = include_intent, include_filled
        self.max_len, self.head_max_len = max_len, head_max_len
        self.temperature, self.batch_size = temperature, batch_size
        self.device, self.revision = device, revision
        self.model_path = model_path
        self.choice_instructions = choice_instructions
        self.choice_given_expressed = choice_given_expressed
        self.none_description = none_description
        self.expressed_instructions = expressed_instructions
        self.predictor = predictor
        self.name = f"laya-{mode}-{checkpoint}" + ("-ft" if model_path else "")

    # ------------------------------------------------------------------ modèle
    def _get_predictor(self):
        if self.predictor is None:
            from laya import Router  # dépendance optionnelle : uv sync --extra laya
            agent_kwargs = {}
            if self.temperature == "neutral":
                # Le fichier remplace toutes les températures du checkpoint (laya.calibrate).
                with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                    json.dump({"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}}, f)
                agent_kwargs["calibration"] = f.name
            # Un checkpoint fine-tuné remplace, sous le même nom, le checkpoint publié qu'il prolonge.
            models = {self.checkpoint: str(self.model_path)} if self.model_path else None
            self.predictor = Router(models=models, device=self.device,
                                    revision=None if self.model_path else self.revision,
                                    agent_kwargs=agent_kwargs or None)
            if self.checkpoint in ("multilingual", "english"):
                self.predictor.preload([self.checkpoint])
        return self.predictor

    # ------------------------------------------------------------------ construction
    def build_state(self, req: JudgeRequest) -> dict:
        """État envoyé à Laya. Énoncé en tête : Laya tronque un état dict par la droite."""
        state = {"last_user_message": clip(req.example.utterance, self.utterance_max_chars)}
        if self.include_intent:
            state["intent"] = req.intent or "unknown"
        if self.include_filled and req.filled:
            state["slots_already_filled"] = format_filled(req.filled)
        hist = format_history(req, self.history_turns, self.history_max_chars)
        if hist:
            state["history"] = hist
        return state

    def _routing(self, req: JudgeRequest) -> dict:
        if self.checkpoint in ("multilingual", "english"):
            return {"model": self.checkpoint}
        if self.checkpoint == "lang":
            return {"lang": req.example.lang}
        return {}

    def _fmt(self, template: str, req: JudgeRequest) -> str:
        return template.format(slot=req.slot.name, description=req.slot.description or req.slot.name,
                               lang=LANG_NAMES.get(req.example.lang, req.example.lang))

    def _choice(self, req: JudgeRequest, idx: list[int], with_none: bool, template: str) -> dict:
        texts = [clip(candidate_text(req.candidates[i]), self.option_max_chars) for i in idx]
        if with_none:
            texts.append(self.none_description)
        crit = dict(zip(option_keys(len(texts)), texts))
        return {"type": "choice", "instructions": self._fmt(template, req), "criteria": crit}

    def _expressed(self, req: JudgeRequest) -> dict:
        ins = self._fmt(self.expressed_instructions, req)
        if self.expressed_qtype == "choice":
            return {"type": "choice", "instructions": ins,
                    "criteria": {"A": EXPRESSED_TRUE, "B": EXPRESSED_FALSE}}
        q = {"type": "noul", "instructions": ins,
             "criteria": {"true": EXPRESSED_TRUE, "false": EXPRESSED_FALSE}}
        if self.noul_labels:
            q["labels"] = dict(self.noul_labels)
        return q

    def _with_rotations(self, qid: str, q: dict) -> dict[str, dict]:
        """La question et ses rotations circulaires (même passe avant)."""
        k = len(q["criteria"])
        n = min(self.rotations, k) if q["type"] == "choice" else 1
        if n <= 1:
            return {qid: q}
        shifts = sorted({round(j * k / n) % k for j in range(n)})
        return {f"{qid}@{r}": dict(q, option_order=[(i + r) % k for i in range(k)]) for r in shifts}

    def _plan(self, req: JudgeRequest) -> _Plan:
        if self.overflow == "truncate":
            idx, dropped = order_candidates(req, self.candidate_order, self.seed, self.max_options)
            chunks = [idx]
        else:
            idx, dropped = order_candidates(req, self.candidate_order, self.seed, None)
            n_chunks = max(1, math.ceil(len(idx) / self.max_options))
            size = math.ceil(len(idx) / n_chunks) if idx else 0
            chunks = [idx[i:i + size] for i in range(0, len(idx), size)] if idx else [[]]
        plan = _Plan(state=self.build_state(req), routing=self._routing(req), chunks=chunks,
                     dropped=dropped)
        if not req.candidates:
            return plan
        single = self.mode == "single"
        if len(chunks) == 1:
            template = self.choice_instructions if single else self.choice_given_expressed
            plan.round1.update(self._with_rotations("v", self._choice(req, chunks[0], single, template)))
        else:
            for c, chunk in enumerate(chunks):
                plan.round1.update(self._with_rotations(
                    f"t{c}", self._choice(req, chunk, False, self.choice_given_expressed)))
        if not single:
            plan.round1["x"] = self._expressed(req)
        return plan

    # ------------------------------------------------------------------ exécution
    def _run(self, plans: dict[int, _Plan], questions: dict[int, dict[str, dict]]) -> None:
        """Un appel `predict_batch` ; regroupe les requêtes qui partagent un état."""
        groups: dict[str, dict] = {}
        for i, qs in questions.items():
            if not qs:
                continue
            p = plans[i]
            key = json.dumps([p.state, p.routing], sort_keys=True, ensure_ascii=False)
            g = groups.setdefault(key, {"state": p.state, "routing": p.routing, "questions": {},
                                        "members": []})
            g["members"].append(i)
            for qid, q in qs.items():
                g["questions"][f"r{i}|{qid}"] = q
        if not groups:
            return
        calls = []
        for g in groups.values():
            r = {"state": g["state"], "questions": g["questions"], **g["routing"]}
            if self.max_len is not None:
                r["max_len"] = self.max_len
            if self.head_max_len is not None:
                r["head_max_len"] = self.head_max_len
            calls.append(r)
        predictor = self._get_predictor()
        t0 = time.perf_counter()
        results = predictor.predict_batch(calls, batch_size=self.batch_size)
        per_req = (time.perf_counter() - t0) * 1000.0 / sum(len(g["members"]) for g in groups.values())
        for g, res in zip(groups.values(), results):
            usage = dict(res.get("usage") or {})
            usage["model"] = (res.get("routing") or {}).get("model")
            for i in g["members"]:
                plans[i].latency_ms += per_req
                plans[i].usage.append(usage)
            for full_qid, ans in res["answers"].items():
                i, qid = full_qid.split("|", 1)
                plans[int(i[1:])].answers[qid] = ans

    def judge_batch(self, requests: list[JudgeRequest]) -> list[Judgment]:
        plans = {i: self._plan(r) for i, r in enumerate(requests)}
        self._run(plans, {i: p.round1 for i, p in plans.items()})
        # Finale du tournoi : les vainqueurs de chaque paquet, plus « non précisé » en single.
        finals: dict[int, dict] = {}
        for i, p in plans.items():
            if len(p.chunks) > 1:
                winners = []
                for c, chunk in enumerate(p.chunks):
                    probs = _read_choice(p.answers, f"t{c}", len(chunk))
                    winners.append(chunk[max(range(len(chunk)), key=lambda j: (probs[j], -j))])
                p.chunks.append(winners)          # la dernière entrée est la finale
                single = self.mode == "single"
                template = self.choice_instructions if single else self.choice_given_expressed
                finals[i] = self._with_rotations("v", self._choice(requests[i], winners, single, template))
        self._run(plans, finals)
        return [self._judgment(r, plans[i]) for i, r in enumerate(requests)]

    # ------------------------------------------------------------------ lecture
    def _judgment(self, req: JudgeRequest, p: _Plan) -> Judgment:
        n = len(req.candidates)
        raw: dict[str, Any] = {"judge": self.name, "dropped": p.dropped, "usage": p.usage}
        if n == 0:
            raw["skipped"] = "no_candidates"
            return Judgment(choice=None, proba=1.0, probs=[1.0], latency_ms=0.0, raw=raw)
        final = p.chunks[-1]
        if len(p.chunks) > 1:
            raw["tournament"] = {f"t{c}": _read_choice(p.answers, f"t{c}", len(ch))
                                 for c, ch in enumerate(p.chunks[:-1])}
        probs = [0.0] * (n + 1)
        if self.mode == "single":
            q = _read_choice(p.answers, "v", len(final) + 1)
            for pos, ci in enumerate(final):
                probs[ci] = q[pos]
            probs[n] = q[-1]
            raw["order"] = final
            return judgment_from_probs(normalize(probs), latency_ms=p.latency_ms, raw=raw)
        pe = _read_expressed(p.answers.get("x"))
        q = _read_choice(p.answers, "v", len(final))
        for pos, ci in enumerate(final):
            probs[ci] = pe * q[pos]
        probs[n] = 1.0 - pe
        raw.update(order=final, p_choice=q)
        if pe < self.expressed_threshold:
            return Judgment(choice=None, proba=probs[n], probs=probs, p_expressed=pe,
                            latency_ms=p.latency_ms, raw=raw)
        best = final[max(range(len(final)), key=lambda j: (q[j], -j))]
        return Judgment(choice=best, proba=probs[best], probs=probs, p_expressed=pe,
                        latency_ms=p.latency_ms, raw=raw)


def _read_choice(answers: dict[str, dict], qid: str, k: int) -> list[float]:
    """Distribution sur les k clés A, B, … ; moyenne sur les rotations s'il y en a.

    Laya arrondit les probabilités à 4 décimales : on renormalise. Si `probabilities` manque
    (ancienne API), on se replie sur un one-hot au choix retourné.
    """
    keys = option_keys(k)
    found = [a for q, a in answers.items() if q == qid or q.startswith(qid + "@")]
    if not found:
        raise KeyError(f"pas de réponse Laya pour la question {qid!r}")
    acc = [0.0] * k
    for a in found:
        probs = a.get("probabilities")
        if probs:
            dist = normalize([float(probs.get(key, 0.0)) for key in keys])
        else:
            dist = [1.0 if key == a.get("choice") else 0.0 for key in keys]
        acc = [x + y / len(found) for x, y in zip(acc, dist)]
    return acc


def _read_expressed(ans: dict | None) -> float:
    if ans is None:
        raise KeyError("pas de réponse Laya pour la question « slot exprimé »")
    if ans.get("type") == "noul" or "noul" in ans:
        return float(ans["noul"])
    probs = ans.get("probabilities") or {}
    if probs:
        return normalize([float(probs.get("A", 0.0)), float(probs.get("B", 0.0))])[0]
    return 1.0 if ans.get("choice") == "A" else 0.0
