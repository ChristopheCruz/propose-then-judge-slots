"""Décision à trois zones et courbes risque-couverture (QR4, H4).

Pour chaque requête jugée, le juge retient l'option la plus probable (une valeur, ou « non
précisé ») avec une confiance c. Une politique à seuil τ :
- c ≥ τ et option = valeur        -> **agir** : la valeur est remplie sans rien demander ;
- c ≥ τ et option = « non précisé » -> **ignorer** : le slot reste vide ;
- c < τ                           -> **redemander** : la décision est renvoyée à l'utilisateur
  (ou à un juge plus coûteux) ; on la compte comme non couverte.
Couverture = part des requêtes décidées sans redemander ; risque = part d'erreurs parmi elles.
La courbe risque-couverture parcourt tous les seuils ; l'AURC est son aire (plus bas = mieux).
Le seuil opérationnel est choisi **sur le dev** pour un risque cible, puis appliqué au test.

Les requêtes à liste vide (« non précisé » sans interroger le juge) ne sont pas dans ces calculs
(voir calibration.requests_from_records) : elles sont sûres et identiques pour tous les juges.
"""
from __future__ import annotations

import numpy as np


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> dict:
    """Courbe risque-couverture (seuils = confiances observées, ex aequo traités ensemble) et AURC."""
    n = len(conf)
    if n == 0:
        return {"coverage": [], "risk": [], "aurc": None, "n": 0}
    order = np.argsort(-conf, kind="stable")
    c, ok = conf[order], correct[order]
    err = np.cumsum(1.0 - ok)
    # dernier indice de chaque groupe d'ex aequo : on couvre tout le groupe ou rien
    last = np.r_[np.flatnonzero(np.diff(c) != 0), n - 1]
    cov = (last + 1) / n
    risk = err[last] / (last + 1)
    # aire sous la courbe en escalier, couverture de 0 à 1 (risque du premier point sur [0, cov0])
    widths = np.diff(np.r_[0.0, cov])
    aurc = float((widths * risk).sum())
    return {"coverage": cov.tolist(), "risk": risk.tolist(), "aurc": aurc, "n": n,
            "thresholds": c[last].tolist(), "full_coverage_risk": float(risk[-1])}


def threshold_for_risk(conf: np.ndarray, correct: np.ndarray, target_risk: float) -> float | None:
    """Plus petit seuil (couverture maximale) dont le risque sur ces données est ≤ target_risk.

    None si aucun seuil n'y arrive (même la requête la plus sûre se trompe)."""
    rc = risk_coverage(conf, correct)
    best = None
    for t, r in zip(rc.get("thresholds", []), rc["risk"]):
        if r <= target_risk:
            best = t                  # seuils décroissants : le dernier admissible couvre le plus
    return best


def apply_threshold(conf: np.ndarray, correct: np.ndarray, is_value: np.ndarray, threshold: float | None) -> dict:
    """Bilan d'une politique à seuil : parts agir / ignorer / redemander, risque et couverture."""
    n = len(conf)
    if threshold is None or n == 0:
        return {"threshold": threshold, "coverage": 0.0, "risk": None, "act": 0.0, "ignore": 0.0, "ask": 1.0, "n": n}
    covered = conf >= threshold
    k = int(covered.sum())
    return {"threshold": float(threshold), "coverage": k / n,
            "risk": float(1.0 - correct[covered].mean()) if k else None,
            "act": float((covered & is_value).mean()), "ignore": float((covered & ~is_value).mean()),
            "ask": float((~covered).mean()), "n": n}
