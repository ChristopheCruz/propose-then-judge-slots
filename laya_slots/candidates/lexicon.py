"""Générateur « lexicon » : index des valeurs vues au train, appariées dans l'énoncé par n-grammes.

`fit` indexe, pour chaque slot, les valeurs gold normalisées du TRAIN (et, en option, les valeurs
fermées de l'ontologie). `generate` normalise chaque n-gramme de l'énoncé et garde ceux qui sont
dans l'index du slot.

Score : affinité P(slot | valeur) = n(slot, valeur) / n(valeur, tous slots), estimée au train
(les valeurs ajoutées depuis l'ontologie comptent pour 1). Ex. « boston » est à la fois un
départ et une arrivée : les deux slots le reçoivent avec un score d'environ 0,5. À score égal,
le span le plus long passe devant.
"""
from __future__ import annotations

from collections import Counter

from ..interfaces import CandidateGenerator
from ..schema import NONE_VALUE, Candidate, Example, Ontology, SlotSpec
from .base import NormalizeFn, Normalizer, as_normalizer, dedupe, ngrams, tokens


class ValueIndex:
    """slot -> Counter(valeur normalisée), construit sur le train et l'ontologie."""

    def __init__(self, normalize: NormalizeFn | Normalizer | None = None, min_count: int = 1,
                 use_ontology: bool = True):
        self.normalize = as_normalizer(normalize)
        self.min_count = min_count
        self.use_ontology = use_ontology
        self.by_slot: dict[str, Counter] = {}
        self.by_value: Counter = Counter()
        self.max_tokens = 1

    def fit(self, train: list[Example], ontology: Ontology | None) -> ValueIndex:
        by_slot: dict[str, Counter] = {}
        for ex in train:
            for sv in ex.slots:
                v = self.normalize(sv.value, slot=sv.name, lang=ex.lang)
                if v and v != NONE_VALUE:
                    by_slot.setdefault(sv.name, Counter())[v] += 1
        by_slot = {s: Counter({v: n for v, n in c.items() if n >= self.min_count}) for s, c in by_slot.items()}
        if self.use_ontology and ontology is not None:
            for spec in ontology.slots.values():
                c = by_slot.setdefault(spec.name, Counter())
                for v in spec.values or []:
                    nv = self.normalize(v, slot=spec.name, lang=ontology.lang)
                    if nv and nv != NONE_VALUE and nv not in c:
                        c[nv] = 1
        self.by_slot = by_slot
        self.by_value = Counter()
        for c in by_slot.values():
            self.by_value.update(c)
        self.max_tokens = max((len(tokens(v)) for v in self.by_value), default=1)
        return self

    def affinity(self, slot: str, value: str) -> float:
        n = self.by_slot.get(slot, {}).get(value, 0)
        return n / self.by_value[value] if n else 0.0


class LexiconGenerator(CandidateGenerator):
    name = "lexicon"

    def __init__(self, min_count: int = 1, max_ngram: int = 6, use_ontology: bool = True,
                 min_affinity: float = 0.0, normalize: NormalizeFn | Normalizer | None = None):
        self.normalize = as_normalizer(normalize)
        self.index = ValueIndex(self.normalize, min_count, use_ontology)
        self.max_ngram = max_ngram
        self.min_affinity = min_affinity

    def fit(self, train: list[Example], ontology: Ontology) -> None:
        self.index.fit(train, ontology)

    def generate(self, example: Example, slot: SlotSpec, ontology: Ontology) -> list[Candidate]:
        known = self.index.by_slot.get(slot.name)
        if not known:
            return []
        max_n = min(self.max_ngram, self.index.max_tokens)
        found = []
        for sp in ngrams(example.utterance, max_n):
            v = self.normalize(sp.text, slot=slot.name, lang=example.lang)
            if v in known:
                a = self.index.affinity(slot.name, v)
                if a >= self.min_affinity:
                    found.append((a, sp.n_tokens, Candidate(v, self.name, a, sp.text, (sp.start, sp.end))))
        found.sort(key=lambda t: (-t[0], -t[1], t[2].span))
        return dedupe(c for _, _, c in found)
