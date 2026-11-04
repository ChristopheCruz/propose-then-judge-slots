"""Exemples d'entraînement de Laya, construits exactement comme les questions vues à l'inférence.

Chaque `JudgeRequest` passe par `LayaJudge._plan`, la fonction qui prépare les questions envoyées à
Laya pendant l'évaluation : état, gabarit d'instructions, ordre et troncature des candidats, option
« non précisé ». Seule la cible s'ajoute. Un écart de format entre entraînement et évaluation est
ainsi impossible par construction.

Cibles (indice d'option, dans l'ordre présenté au modèle) :
- mode single, question "v" : la position de la valeur gold parmi les candidats, sinon la dernière
  option (« non précisé ») quand le slot n'est pas exprimé ou que la gold manque à la liste ;
- mode two_step, question "x" (slot exprimé ?) : choice A/B -> 0 si exprimé, 1 sinon ; noul ->
  1 (true) si exprimé, 0 (false) sinon (ordre des options noul de Laya : [false, true]) ;
- mode two_step, question "v" (choix parmi les candidats) : seulement si la gold est présentée.

Ce module ne dépend ni de torch ni de laya : il se teste sur CPU.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from ..judges.laya_judge import LayaJudge
from ..schema import JudgeRequest


@dataclass
class TrainItem:
    state: dict
    question: dict              # question au format de l'API Laya (type, instructions, criteria…)
    target: int                 # indice de l'option correcte
    kind: str                   # "choice" (avec « non précisé »), "choice_given" ou "expressed"
    expressed: bool             # le slot est exprimé dans le tour (gold non vide)
    meta: dict = field(default_factory=dict)


def items_from_request(judge: LayaJudge, request: JudgeRequest,
                       gold_values: Iterable[str]) -> list[TrainItem]:
    """Questions d'entraînement pour une requête. `gold_values` : valeurs gold du slot (vide si
    le slot n'est pas exprimé), lues par l'appelant hors de la requête, qui n'en contient pas."""
    if judge.rotations != 1 or judge.overflow != "truncate":
        # les rotations et le tournoi sont des procédés d'inférence : on entraîne sur la question simple
        judge = _plain_copy(judge)
    gold = set(gold_values)
    expressed = bool(gold)
    plan = judge._plan(request)
    meta = {"example_id": request.example.id, "slot": request.slot.name}
    out: list[TrainItem] = []
    if not request.candidates:
        return out
    order = plan.chunks[0]
    pos = next((p for p, ci in enumerate(order) if request.candidates[ci].value in gold), None)
    v = plan.round1["v"]
    if judge.mode == "single":
        target = len(order) if pos is None else pos       # dernière option = « non précisé »
        out.append(TrainItem(plan.state, v, target, "choice", expressed, meta))
        return out
    x = plan.round1["x"]
    if x["type"] == "noul":
        out.append(TrainItem(plan.state, x, 1 if expressed else 0, "expressed", expressed, meta))
    else:
        out.append(TrainItem(plan.state, x, 0 if expressed else 1, "expressed", expressed, meta))
    if pos is not None:
        out.append(TrainItem(plan.state, v, pos, "choice_given", expressed, meta))
    return out


def _plain_copy(judge: LayaJudge) -> LayaJudge:
    """Même juge, sans rotation ni tournoi (copie superficielle, le prédicteur n'est pas partagé)."""
    clone = object.__new__(type(judge))
    clone.__dict__.update(judge.__dict__)
    clone.rotations, clone.overflow, clone.predictor = 1, "truncate", None
    return clone
