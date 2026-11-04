"""Juges de référence, sans modèle.

Toutes renvoient `probs` (len(candidates)+1, dernière case = « non précisé ») : one-hot pour les
règles déterministes, uniforme pour `random`. Ces probabilités ne sont pas informatives ; elles
servent seulement à garder un format homogène en aval.
"""
from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from ..interfaces import Judge
from ..schema import NONE_VALUE, Example, JudgeRequest, Judgment, read_examples
from ._common import fold_text, request_rng


def _one_hot(n_candidates: int, choice: int | None) -> list[float]:
    probs = [0.0] * (n_candidates + 1)
    probs[n_candidates if choice is None else choice] = 1.0
    return probs


class _RuleJudge(Judge):
    """Squelette : une règle par requête, latence mesurée par requête."""

    def judge_batch(self, requests: list[JudgeRequest]) -> list[Judgment]:
        out = []
        for req in requests:
            t0 = time.perf_counter()
            j = self._judge_one(req)
            j.latency_ms = (time.perf_counter() - t0) * 1000.0
            out.append(j)
        return out

    def _judge_one(self, req: JudgeRequest) -> Judgment:
        raise NotImplementedError


class FirstJudge(_RuleJudge):
    """Premier candidat dans l'ordre du générateur ; « non précisé » si la liste est vide.

    Mesure en creux le biais d'ordre du générateur (un oracle qui place la bonne valeur en tête
    donne 100 % à cette baseline : à vérifier côté candidats).
    """
    name = "first"

    def _judge_one(self, req):
        choice = 0 if req.candidates else None
        return Judgment(choice=choice, proba=1.0, probs=_one_hot(len(req.candidates), choice))


class RandomJudge(_RuleJudge):
    """Tirage uniforme, déterministe par (graine, exemple, slot).

    include_none : « non précisé » fait partie des options tirées.
    """
    name = "random"

    def __init__(self, seed: int = 0, include_none: bool = True):
        self.seed = seed
        self.include_none = include_none

    def _judge_one(self, req):
        n = len(req.candidates)
        k = n + 1 if self.include_none else n
        if k == 0:
            return Judgment(choice=None, proba=1.0, probs=[1.0])
        draw = request_rng(self.seed, req, "random").randrange(k)
        choice = None if draw == n else draw
        probs = [1.0 / k] * n + ([1.0 / k] if self.include_none else [0.0])
        return Judgment(choice=choice, proba=1.0 / k, probs=probs)


class GeneratorScoreJudge(_RuleJudge):
    """Candidat de meilleur score générateur (ex aequo : le premier).

    min_score : en dessous, « non précisé » (None = jamais).
    Sans aucun score : premier candidat si fallback_first, sinon « non précisé ».
    proba : le score s'il est dans [0, 1], sinon 1.0 (scores non calibrés).
    """
    name = "generator_score"

    def __init__(self, min_score: float | None = None, fallback_first: bool = True):
        self.min_score = min_score
        self.fallback_first = fallback_first

    def _judge_one(self, req):
        scored = [(c.score, i) for i, c in enumerate(req.candidates) if c.score is not None]
        if not scored:
            choice = 0 if (req.candidates and self.fallback_first) else None
            return Judgment(choice=choice, proba=1.0, probs=_one_hot(len(req.candidates), choice))
        score, best = max(scored, key=lambda t: (t[0], -t[1]))
        if self.min_score is not None and score < self.min_score:
            return Judgment(choice=None, proba=1.0, probs=_one_hot(len(req.candidates), None))
        proba = score if 0.0 <= score <= 1.0 else 1.0
        return Judgment(choice=best, proba=proba, probs=_one_hot(len(req.candidates), best))


class StringMatchJudge(_RuleJudge):
    """Candidat dont la forme apparaît dans l'énoncé courant (mots entiers, sans casse ni accents).

    fields : formes testées, parmi "surface" et "value" (valeur normalisée).
    Plusieurs candidats trouvés : le plus long gagne, puis le premier. Aucun : « non précisé ».
    Attention : si seuls certains générateurs remplissent `surface` (ex. l'oracle), cette baseline
    en profite ; c'est une propriété des candidats, pas du juge.
    """
    name = "string_match"

    def __init__(self, fields: Iterable[str] = ("surface", "value")):
        self.fields = tuple(fields)
        bad = set(self.fields) - {"surface", "value"}
        if bad:
            raise ValueError(f"champs inconnus : {sorted(bad)}")

    def _judge_one(self, req):
        text = f" {fold_text(req.example.utterance)} "
        best, best_len = None, 0
        for i, c in enumerate(req.candidates):
            for f in self.fields:
                form = getattr(c, f)
                if not form or form == NONE_VALUE:
                    continue
                folded = fold_text(form)
                if folded and len(folded) > best_len and re.search(
                        rf"(?<!\w){re.escape(folded)}(?!\w)", text):
                    best, best_len = i, len(folded)
        return Judgment(choice=best, proba=1.0, probs=_one_hot(len(req.candidates), best),
                        raw={"matched": best is not None})


class MajorityJudge(_RuleJudge):
    """Valeur la plus fréquente au train pour ce slot.

    `fit(train)` compte les valeurs gold (normalisées) par slot sur le TRAIN : c'est la seule
    lecture de `slots` autorisée, jamais sur l'exemple jugé.
    Décision : le candidat dont la valeur a la plus grande fréquence au train (ex aequo : le
    premier). Si aucun candidat n'a été vu au train, « non précisé » (abstain_if_unseen) ou le
    premier candidat. proba = fréquence relative de la valeur retenue parmi les valeurs du slot.
    """
    name = "majority"

    def __init__(self, train_path: str | Path | None = None, abstain_if_unseen: bool = True):
        self.abstain_if_unseen = abstain_if_unseen
        self.counts: dict[str, Counter] = defaultdict(Counter)
        if train_path is not None:
            self.fit(read_examples(train_path))

    def fit(self, train: Iterable[Example]) -> MajorityJudge:
        self.counts = defaultdict(Counter)
        for ex in train:
            for sv in ex.slots:
                self.counts[sv.name][sv.value] += 1
        return self

    def _judge_one(self, req):
        counts = self.counts.get(req.slot.name, Counter())
        total = sum(counts.values())
        freqs = [counts.get(c.value, 0) for c in req.candidates]
        if not freqs or max(freqs) == 0:
            choice = 0 if (req.candidates and not self.abstain_if_unseen) else None
            return Judgment(choice=choice, proba=1.0, probs=_one_hot(len(req.candidates), choice))
        best = max(range(len(freqs)), key=lambda i: (freqs[i], -i))
        proba = freqs[best] / total if total else 1.0
        return Judgment(choice=best, proba=proba, probs=_one_hot(len(req.candidates), best),
                        raw={"train_count": freqs[best], "train_total": total})
