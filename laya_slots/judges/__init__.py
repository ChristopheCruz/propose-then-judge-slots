"""Juges de slots : Laya, LLM-juge, baselines, et un registre construit depuis la config.

    from laya_slots.judges import build_judge
    judge = build_judge({"type": "laya", "mode": "two_step", "checkpoint": "multilingual"})
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..interfaces import Judge
from .baselines import FirstJudge, GeneratorScoreJudge, MajorityJudge, RandomJudge, StringMatchJudge
from .laya_judge import LayaJudge
from .llm_judge import LLMJudge, OpenAICompatBackend, VLLMBackend

JUDGES: dict[str, type[Judge]] = {
    "laya": LayaJudge,
    "llm": LLMJudge,
    "first": FirstJudge,
    "random": RandomJudge,
    "generator_score": GeneratorScoreJudge,
    "string_match": StringMatchJudge,
    "majority": MajorityJudge,
}


def build_judge(config: Mapping[str, Any], **objects: Any) -> Judge:
    """Construit un juge à partir de {"type": ..., <paramètres du constructeur>}.

    Les objets non sérialisables (faux prédicteur, backend déjà construit…) passent par
    `objects`. Accepte aussi une DictConfig OmegaConf.
    """
    if hasattr(config, "_metadata"):            # DictConfig -> dict
        from omegaconf import OmegaConf
        config = OmegaConf.to_container(config, resolve=True)
    params = dict(config)
    kind = params.pop("type", None)
    if kind not in JUDGES:
        raise ValueError(f"type de juge inconnu : {kind!r} ; choisir parmi {sorted(JUDGES)}")
    return JUDGES[kind](**params, **objects)


__all__ = [
    "JUDGES",
    "FirstJudge",
    "GeneratorScoreJudge",
    "LLMJudge",
    "LayaJudge",
    "MajorityJudge",
    "OpenAICompatBackend",
    "RandomJudge",
    "StringMatchJudge",
    "VLLMBackend",
    "build_judge",
]
