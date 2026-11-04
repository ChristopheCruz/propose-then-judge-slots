"""Union de générateurs et top-k par embeddings."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from ..interfaces import CandidateGenerator
from ..schema import Candidate, Example, Ontology, SlotSpec
from .base import dedupe, sort_by_score

Order = Literal["round_robin", "priority", "score"]


class UnionGenerator(CandidateGenerator):
    """Fusionne plusieurs générateurs, déduplique et tronque au top-k.

    order :
    - "round_robin" (défaut) : on prend le 1er candidat de chaque générateur, puis le 2e, etc.
      Robuste quand les scores des générateurs ne sont pas comparables ;
    - "priority" : tous les candidats du 1er générateur, puis ceux du 2e, etc. ;
    - "score" : tri global par score (n'a de sens que si les scores sont comparables).
    Les doublons gardent la première occurrence ; la source devient "a+b" si merge_sources.
    """
    name = "union"

    def __init__(self, generators: Sequence[CandidateGenerator], k: int | None = None,
                 order: Order = "round_robin", merge_sources: bool = True):
        if any(getattr(g, "name", None) == "oracle" for g in generators):
            raise ValueError("l'oracle ne se combine pas avec des générateurs réels")
        self.generators = list(generators)
        self.k = k
        self.order = order
        self.merge_sources = merge_sources

    def fit(self, train: list[Example], ontology: Ontology) -> None:
        for g in self.generators:
            g.fit(train, ontology)

    def generate(self, example: Example, slot: SlotSpec, ontology: Ontology) -> list[Candidate]:
        lists = [g.generate(example, slot, ontology) for g in self.generators]
        if self.order == "priority":
            merged = [c for lst in lists for c in lst]
        elif self.order == "score":
            merged = sort_by_score([c for lst in lists for c in lst])
        else:
            merged = [lst[i] for i in range(max(map(len, lists), default=0)) for lst in lists if i < len(lst)]
        out = dedupe(merged, merge_sources=self.merge_sources)
        return out if self.k is None else out[: self.k]


Encoder = Callable[[list[str]], "object"]   # list[str] -> array (n, d)


class EmbeddingTopK(CandidateGenerator):
    """Reclasse les candidats d'un générateur de base par similarité cosinus et garde les k premiers.

    Requête : "<description du slot> : <énoncé>" (query="slot_utterance") ou l'énoncé seul
    (query="utterance"). Le score du candidat devient la similarité cosinus.
    Le modèle sentence-transformers est chargé à la première utilisation (extra `candidates`) ;
    `encoder` permet d'injecter une fonction d'encodage (tests, cache externe).
    """
    name = "embedding_topk"

    def __init__(self, base: CandidateGenerator, k: int = 10,
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 query: Literal["slot_utterance", "utterance"] = "slot_utterance",
                 encoder: Encoder | None = None, device: str | None = None, keep_source: bool = True):
        self.base = base
        self.k = k
        self.model_name = model_name
        self.query = query
        self.device = device
        self.keep_source = keep_source
        self._encoder = encoder
        self._cache: dict[str, object] = {}

    def fit(self, train: list[Example], ontology: Ontology) -> None:
        self.base.fit(train, ontology)

    def _encode(self, texts: list[str]):
        import numpy as np
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer  # import paresseux
            model = SentenceTransformer(self.model_name, device=self.device)
            self._encoder = lambda xs: model.encode(xs, normalize_embeddings=True, convert_to_numpy=True)
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            for t, e in zip(missing, np.asarray(self._encoder(missing), dtype=float)):
                n = np.linalg.norm(e)
                self._cache[t] = e / n if n else e
        return np.stack([self._cache[t] for t in texts])

    def generate(self, example: Example, slot: SlotSpec, ontology: Ontology) -> list[Candidate]:
        cands = self.base.generate(example, slot, ontology)
        if len(cands) <= self.k:
            return cands
        q = example.utterance if self.query == "utterance" else f"{slot.description or slot.name} : {example.utterance}"
        emb = self._encode([q] + [c.value for c in cands])
        sims = emb[1:] @ emb[0]
        order = sorted(range(len(cands)), key=lambda i: -sims[i])[: self.k]
        return [Candidate(cands[i].value, cands[i].source if self.keep_source else self.name, float(sims[i]),
                          cands[i].surface, cands[i].span) for i in order]
