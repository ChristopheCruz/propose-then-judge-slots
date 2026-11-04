"""Générateurs de candidats, échantillonneur de distracteurs et mesure du rappel.

Construction depuis une config (dict ou OmegaConf converti en dict) :

    build_generator({"name": "oracle", "k": 5, "p_absent": 0.2, "seed": 13, "distractors": "hard"})
    build_generator({"name": "union", "k": 10, "generators": [{"name": "lexicon"}, {"name": "ontology"}]})
    build_generator({"name": "embedding_topk", "k": 10, "base": {"name": "ontology", "mode": "all"}})

`normalize` (fonction de normalisation) se passe à `build_generator` et est propagée à tous les
sous-générateurs ; par défaut, normalisation minimale locale (voir base.minimal_normalize).
"""
from __future__ import annotations

from typing import Any

from ..interfaces import CandidateGenerator
from .base import NormalizeFn, Normalizer, minimal_normalize, stable_seed
from .distractors import PRESETS, STRATEGIES, DistractorConfig, DistractorSampler
from .gliner import GlinerGenerator
from .history import HistoryGenerator
from .lexicon import LexiconGenerator, ValueIndex
from .ontology import OntologyGenerator
from .oracle import OracleGenerator
from .recall import candidate_recall, candidate_records, query_slots, report_to_frame, summarize
from .tagger import TaggerGenerator
from .union import EmbeddingTopK, UnionGenerator

GENERATORS: dict[str, type[CandidateGenerator]] = {
    "oracle": OracleGenerator,
    "ontology": OntologyGenerator,
    "history": HistoryGenerator,
    "lexicon": LexiconGenerator,
    "gliner": GlinerGenerator,
    "tagger": TaggerGenerator,
}


def build_generator(config: dict[str, Any], normalize: NormalizeFn | Normalizer | None = None
                    ) -> CandidateGenerator:
    cfg = dict(config)
    name = cfg.pop("name")
    if name == "union":
        subs = [build_generator(c, normalize) for c in cfg.pop("generators")]
        return UnionGenerator(subs, **cfg)
    if name == "embedding_topk":
        return EmbeddingTopK(build_generator(cfg.pop("base"), normalize), **cfg)
    if name not in GENERATORS:
        raise ValueError(f"générateur inconnu : {name} (connus : {sorted(GENERATORS) + ['union', 'embedding_topk']})")
    if normalize is not None:
        cfg.setdefault("normalize", normalize)
    return GENERATORS[name](**cfg)


__all__ = [
    "GENERATORS",
    "PRESETS",
    "STRATEGIES",
    "DistractorConfig",
    "DistractorSampler",
    "EmbeddingTopK",
    "GlinerGenerator",
    "HistoryGenerator",
    "LexiconGenerator",
    "Normalizer",
    "OntologyGenerator",
    "OracleGenerator",
    "TaggerGenerator",
    "UnionGenerator",
    "ValueIndex",
    "build_generator",
    "candidate_recall",
    "candidate_records",
    "minimal_normalize",
    "query_slots",
    "report_to_frame",
    "stable_seed",
    "summarize",
]
