"""Métriques minimales (première version, infra).

Fonctions pures sur des structures simples, sans dépendance au reste du pipeline.
Une version plus complète (calibration, AURC, bootstrap) viendra s'ajouter ici.

Conventions (CONTRAT.md) :
- métrique principale : F1 micro sur les paires (slot, valeur) par exemple ; une prédiction
  valant NONE_VALUE n'est pas une prédiction ;
- les valeurs sont supposées déjà normalisées : la comparaison est une égalité de chaînes ;
- une « ligne de prédiction » (record) est un dict au format de predictions.jsonl :
  example_id, slot, candidates (liste de dicts avec au moins "value"), choice_value, proba,
  gold_values (liste, vide si le slot n'est pas exprimé ; à défaut gold_value).
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

import numpy as np

from laya_slots.schema import NONE_VALUE

Pair = tuple[str, str]


def _pairs(pairs: Iterable[Pair]) -> Counter:
    """Multiensemble des paires (slot, valeur), NONE_VALUE exclu."""
    return Counter((s, v) for s, v in pairs if v != NONE_VALUE)


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


# ---------------------------------------------------------------------------
# Slots
# ---------------------------------------------------------------------------
def slot_prf(gold: Mapping[str, Iterable[Pair]], pred: Mapping[str, Iterable[Pair]]) -> dict:
    """Précision, rappel et F1 micro des slots.

    gold, pred : example_id -> paires (slot, valeur). Les exemples absents d'un côté comptent
    comme vides. Un slot à plusieurs valeurs gold est un multiensemble : chaque valeur compte.
    """
    tp = fp = fn = 0
    for ex_id in set(gold) | set(pred):
        g, p = _pairs(gold.get(ex_id, ())), _pairs(pred.get(ex_id, ()))
        inter = sum((g & p).values())
        tp += inter
        fp += sum(p.values()) - inter
        fn += sum(g.values()) - inter
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    if precision is None or recall is None:
        # convention : pas de prédiction ET pas de gold -> parfait ; sinon l'absent vaut 0
        precision = 1.0 if precision is None and fn == 0 else (precision or 0.0)
        recall = 1.0 if recall is None and fp == 0 else (recall or 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"slot_precision": precision, "slot_recall": recall, "slot_f1": f1,
            "slot_tp": tp, "slot_fp": fp, "slot_fn": fn}


def frame_accuracy(gold: Mapping[str, Iterable[Pair]], pred: Mapping[str, Iterable[Pair]],
                   gold_intents: Mapping[str, str | None] | None = None,
                   pred_intents: Mapping[str, str | None] | None = None) -> float | None:
    """Part des exemples dont l'ensemble des slots (et l'intention, si fournie) est exact."""
    ids = list(gold)
    if not ids:
        return None
    ok = 0
    for ex_id in ids:
        good = _pairs(gold[ex_id]) == _pairs(pred.get(ex_id, ()))
        if good and gold_intents is not None and pred_intents is not None:
            good = gold_intents.get(ex_id) == pred_intents.get(ex_id)
        ok += good
    return ok / len(ids)


# ---------------------------------------------------------------------------
# Intention
# ---------------------------------------------------------------------------
def intent_accuracy(gold: Mapping[str, str | None], pred: Mapping[str, str | None]) -> float | None:
    """Exactitude de l'intention sur les exemples dont l'intention gold est connue."""
    ids = [i for i, g in gold.items() if g is not None]
    if not ids:
        return None
    return sum(gold[i] == pred.get(i) for i in ids) / len(ids)


# ---------------------------------------------------------------------------
# Candidats et décomposition des erreurs (QR2)
# ---------------------------------------------------------------------------
def gold_values_of(record: dict) -> list[str]:
    """Valeurs gold d'une ligne de prédiction (liste vide = slot non exprimé)."""
    if "gold_values" in record:
        return [v for v in record["gold_values"] if v != NONE_VALUE]
    g = record.get("gold_value", NONE_VALUE)
    return [] if g in (None, NONE_VALUE) else [g]


def candidate_values_of(record: dict) -> list[str]:
    return [c["value"] if isinstance(c, dict) else c for c in record.get("candidates", [])]


def candidate_recall(records: Iterable[dict]) -> dict:
    """Rappel des candidats : part des requêtes à slot exprimé où une valeur gold figure parmi
    les candidats. Ne voit que les slots interrogés (voir error_decomposition pour le reste).
    Donne aussi le nombre moyen de candidats par requête."""
    n_expressed = hit = n = n_cands = 0
    for r in records:
        cands = candidate_values_of(r)
        n += 1
        n_cands += len(cands)
        gv = gold_values_of(r)
        if gv:
            n_expressed += 1
            hit += any(v in cands for v in gv)
    return {"candidate_recall": _ratio(hit, n_expressed), "n_expressed_requests": n_expressed,
            "mean_candidates": _ratio(n_cands, n)}


def error_decomposition(records: Iterable[dict], gold: Mapping[str, Iterable[Pair]]) -> dict:
    """Décompose les erreurs du point de vue des paires gold et des slots non exprimés.

    Pour chaque paire gold (exemple, slot, valeur) :
      correct        le juge a choisi cette valeur ;
      choice_error   la valeur était parmi les candidats, le juge a choisi autre chose
                     (dont choice_abstain : il a répondu « non précisé ») ;
      recall_error   la valeur n'était pas parmi les candidats (dont not_queried : le slot
                     n'a même pas été interrogé ; dont recall_error_abstain : le juge a répondu
                     « non précisé », la bonne réaction quand la gold manque à la liste).
    Pour chaque requête dont le slot n'est pas exprimé :
      correct_abstention / false_positive.
    Un slot à plusieurs valeurs gold ne peut en récupérer qu'une (un choix par requête) :
    les autres comptent en choice_error ou recall_error.
    """
    by_key = {}
    for r in records:
        by_key[(r["example_id"], r["slot"])] = r
    c = Counter()
    for ex_id, pairs in gold.items():
        for slot, value in pairs:
            if value == NONE_VALUE:
                continue
            c["gold_pairs"] += 1
            r = by_key.get((ex_id, slot))
            if r is None:
                c["recall_error"] += 1
                c["not_queried"] += 1
            elif r.get("choice_value") == value:
                c["correct"] += 1
            elif value in candidate_values_of(r):
                c["choice_error"] += 1
                c["choice_abstain"] += r.get("choice_value") == NONE_VALUE
            else:
                c["recall_error"] += 1
                c["recall_error_abstain"] += r.get("choice_value") in (None, NONE_VALUE)
    for r in by_key.values():
        if not gold_values_of(r):
            c["unexpressed_requests"] += 1
            if r.get("choice_value") in (None, NONE_VALUE):
                c["correct_abstention"] += 1
            else:
                c["false_positive"] += 1
    n = c["gold_pairs"]
    present = c["correct"] + c["choice_error"]
    return {
        **{k: c[k] for k in ("gold_pairs", "correct", "choice_error", "choice_abstain", "recall_error",
                             "not_queried", "recall_error_abstain", "unexpressed_requests",
                             "correct_abstention", "false_positive")},
        "recall_error_rate": _ratio(c["recall_error"], n),
        "choice_error_rate": _ratio(c["choice_error"], n),
        # exactitude du choix quand la bonne valeur est présente (« le juge seul »)
        "choice_accuracy_given_present": _ratio(c["correct"], present),
        # rappel des candidats sur toutes les paires gold, slots non interrogés compris
        "candidate_recall_all": _ratio(present, n),
        "false_positive_rate": _ratio(c["false_positive"], c["unexpressed_requests"]),
        # slot exprimé mais gold absente de la liste (p_absent, candidats réels) : part des
        # « non précisé », la seule réponse qui ne crée pas de faux positif
        "absent_abstention_rate": _ratio(c["recall_error_abstain"], c["recall_error"] - c["not_queried"]),
    }


def abstention_stats(records: Iterable[dict]) -> dict:
    """Précision et rappel de l'abstention (« non précisé ») sur les requêtes."""
    tp = fp = fn = 0
    for r in records:
        abstain = r.get("choice_value") in (None, NONE_VALUE)
        unexpressed = not gold_values_of(r)
        tp += abstain and unexpressed
        fp += abstain and not unexpressed
        fn += (not abstain) and unexpressed
    return {"abstention_precision": _ratio(tp, tp + fp), "abstention_recall": _ratio(tp, tp + fn)}


# ---------------------------------------------------------------------------
# Coût
# ---------------------------------------------------------------------------
def latency_stats(latencies_ms: Iterable[float]) -> dict:
    """Moyenne, médiane et p95 des latences (ms)."""
    a = np.asarray([x for x in latencies_ms if x is not None], dtype=float)
    if a.size == 0:
        return {"latency_ms_mean": None, "latency_ms_p50": None, "latency_ms_p95": None}
    return {"latency_ms_mean": float(a.mean()), "latency_ms_p50": float(np.percentile(a, 50)),
            "latency_ms_p95": float(np.percentile(a, 95))}
