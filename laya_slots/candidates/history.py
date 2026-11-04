"""Générateur « history » : valeurs issues de l'état du dialogue et des tours précédents.

Sources, par score décroissant :
- `state_before[slot]` : score 1.0 ;
- même attribut dans un autre domaine (MultiWOZ « hotel-area » -> « restaurant-area », suffixe
  après `domain_sep`) : score `cross_domain_score` ;
- toutes les autres valeurs de l'état (option `all_state`) : score `other_state_score` ;
- valeurs connues du slot (index du train et de l'ontologie, voir `fit`) citées dans les
  `max_turns` derniers tours : score `turn_score * decay ** distance` (distance 0 = dernier tour).
"""
from __future__ import annotations

from ..interfaces import CandidateGenerator
from ..schema import NONE_VALUE, Candidate, Example, Ontology, SlotSpec
from .base import NormalizeFn, Normalizer, as_normalizer, dedupe, ngrams, sort_by_score
from .lexicon import ValueIndex


class HistoryGenerator(CandidateGenerator):
    name = "history"

    def __init__(self, cross_domain: bool = True, all_state: bool = False, scan_turns: bool = True,
                 max_turns: int | None = None, max_ngram: int = 6, domain_sep: str = "-",
                 cross_domain_score: float = 0.8, other_state_score: float = 0.3,
                 turn_score: float = 0.6, decay: float = 0.8,
                 normalize: NormalizeFn | Normalizer | None = None):
        self.cross_domain = cross_domain
        self.all_state = all_state
        self.scan_turns = scan_turns
        self.max_turns = max_turns
        self.max_ngram = max_ngram
        self.domain_sep = domain_sep
        self.cross_domain_score = cross_domain_score
        self.other_state_score = other_state_score
        self.turn_score = turn_score
        self.decay = decay
        self.normalize = as_normalizer(normalize)
        self.index = ValueIndex(self.normalize)

    def fit(self, train: list[Example], ontology: Ontology) -> None:
        self.index.fit(train, ontology)

    def _attr(self, slot: str) -> str:
        return slot.split(self.domain_sep, 1)[-1]

    def generate(self, example: Example, slot: SlotSpec, ontology: Ontology) -> list[Candidate]:
        norm = lambda v: self.normalize(v, slot=slot.name, lang=example.lang)
        out: list[Candidate] = []
        attr = self._attr(slot.name)
        for name, value in sorted(example.state_before.items()):
            v = norm(value)
            if not v or v == NONE_VALUE:
                continue
            if name == slot.name:
                out.append(Candidate(v, self.name, 1.0))
            elif self.cross_domain and self._attr(name) == attr and self.domain_sep in slot.name:
                out.append(Candidate(v, self.name, self.cross_domain_score))
            elif self.all_state:
                out.append(Candidate(v, self.name, self.other_state_score))
        known = self.index.by_slot.get(slot.name)
        if self.scan_turns and known:
            turns = example.history if self.max_turns is None else example.history[-self.max_turns:]
            max_n = min(self.max_ngram, self.index.max_tokens)
            for dist, turn in enumerate(reversed(turns)):
                for sp in ngrams(turn.text, max_n):
                    v = norm(sp.text)
                    if v in known:
                        out.append(Candidate(v, self.name, self.turn_score * self.decay ** dist, sp.text))
        return dedupe(sort_by_score(out))
