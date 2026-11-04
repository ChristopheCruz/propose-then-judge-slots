"""Calibration des probabilités d'un juge, réglée sur le dev (QR4).

Unité : une requête (exemple, slot) avec la distribution du juge sur ses options, candidats puis
« non précisé » en dernier (`probs` de predictions.jsonl, `output.save_probs: true`). La cible est
l'option correcte : la position de la valeur gold parmi les candidats, sinon « non précisé ».
Les requêtes sans candidat (liste vide : « non précisé » sans interroger le juge) sont exclues :
leur décision est certaine et ne dépend pas du juge.

Deux calibrations, ajustées sur le dev puis appliquées telles quelles au test :
- `TemperatureScaling` : p_T ∝ p^(1/T), T minimisant la log-vraisemblance négative du dev. Elle
  garde l'ordre des options (la décision ne change pas), seule la confiance change ;
- `IsotonicConfidence` : régression isotone de « la réponse retenue est juste » sur la confiance
  de la réponse retenue (calibration de la confiance, pas de la distribution complète).

Mesures : NLL, score de Brier multiclasse, ECE (15 intervalles de même largeur sur la confiance).
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ..evaluation.metrics import candidate_values_of, gold_values_of

EPS = 1e-6


@dataclass
class Requests:
    """Requêtes jugées d'un run : distributions (liste de vecteurs), cible, identifiants."""
    probs: list[np.ndarray]
    target: np.ndarray            # indice de l'option correcte
    example_ids: list[str]
    expressed: np.ndarray         # le slot est exprimé (gold non vide)

    def __len__(self) -> int:
        return len(self.probs)

    def subset(self, mask: np.ndarray) -> Requests:
        idx = np.flatnonzero(mask)
        return Requests([self.probs[i] for i in idx], self.target[idx],
                        [self.example_ids[i] for i in idx], self.expressed[idx])


def requests_from_records(records: Iterable[dict]) -> Requests:
    """Lignes de predictions.jsonl -> requêtes jugées (liste non vide, `probs` présent)."""
    probs, target, ids, expressed = [], [], [], []
    for r in records:
        cands = candidate_values_of(r)
        p = r.get("probs")
        if not cands or not p or len(p) != len(cands) + 1:
            continue
        gold = gold_values_of(r)
        t = next((i for i, v in enumerate(cands) if v in gold), len(cands))
        q = np.clip(np.asarray(p, dtype=float), 0.0, None)
        q = q / q.sum() if q.sum() > 0 else np.full(len(q), 1.0 / len(q))
        probs.append(q)
        target.append(t)
        ids.append(r["example_id"])
        expressed.append(bool(gold))
    return Requests(probs, np.asarray(target, dtype=int), ids, np.asarray(expressed, dtype=bool))


def _tempered(p: np.ndarray, temperature: float) -> np.ndarray:
    lp = np.log(np.clip(p, EPS, 1.0)) / temperature
    lp -= lp.max()
    e = np.exp(lp)
    return e / e.sum()


def nll(reqs: Requests, temperature: float = 1.0) -> float:
    if not len(reqs):
        return float("nan")
    return float(-np.mean([np.log(max(_tempered(p, temperature)[t], EPS))
                           for p, t in zip(reqs.probs, reqs.target)]))


def brier(reqs: Requests, temperature: float = 1.0) -> float:
    vals = []
    for p, t in zip(reqs.probs, reqs.target):
        q = _tempered(p, temperature)
        y = np.zeros_like(q)
        y[t] = 1.0
        vals.append(float(((q - y) ** 2).sum()))
    return float(np.mean(vals)) if vals else float("nan")


def confidence_and_correct(reqs: Requests, temperature: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Confiance de l'option retenue (argmax) et justesse de cette option."""
    conf, ok = [], []
    for p, t in zip(reqs.probs, reqs.target):
        q = _tempered(p, temperature)
        k = int(np.argmax(q))
        conf.append(float(q[k]))
        ok.append(k == t)
    return np.asarray(conf), np.asarray(ok, dtype=float)


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Erreur de calibration attendue : écart |exactitude − confiance| pondéré par intervalle."""
    if not len(conf):
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


class TemperatureScaling:
    """Une température T > 0 pour tout le juge, réglée par NLL sur le dev."""

    def __init__(self, bounds: tuple[float, float] = (0.05, 20.0)):
        self.bounds = bounds
        self.temperature = 1.0

    def fit(self, reqs: Requests) -> TemperatureScaling:
        from scipy.optimize import minimize_scalar
        lo, hi = np.log(self.bounds[0]), np.log(self.bounds[1])
        res = minimize_scalar(lambda lt: nll(reqs, float(np.exp(lt))), bounds=(lo, hi), method="bounded")
        self.temperature = float(np.exp(res.x))
        return self

    def confidence(self, reqs: Requests) -> tuple[np.ndarray, np.ndarray]:
        return confidence_and_correct(reqs, self.temperature)


class IsotonicConfidence:
    """Confiance calibrée = P(juste | confiance brute), régression isotone ajustée sur le dev."""

    def __init__(self):
        from sklearn.isotonic import IsotonicRegression
        self.model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, reqs: Requests) -> IsotonicConfidence:
        conf, ok = confidence_and_correct(reqs)
        self.model.fit(conf, ok)
        return self

    def confidence(self, reqs: Requests) -> tuple[np.ndarray, np.ndarray]:
        conf, ok = confidence_and_correct(reqs)
        return self.model.predict(conf), ok


def calibration_report(reqs: Requests, temperature: float = 1.0) -> dict:
    conf, ok = confidence_and_correct(reqs, temperature)
    return {"n": len(reqs), "nll": nll(reqs, temperature), "brier": brier(reqs, temperature),
            "ece": ece(conf, ok), "accuracy": float(ok.mean()) if len(ok) else None,
            "mean_confidence": float(conf.mean()) if len(conf) else None}
