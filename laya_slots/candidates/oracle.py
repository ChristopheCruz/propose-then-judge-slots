"""Générateur oracle contrôlé : les gold + des distracteurs.

Seul générateur autorisé à lire `example.slots` (voir CONTRAT.md). Il sert à mesurer le juge
seul (jalon 1, H1) et à balayer k et le taux de valeur absente (QR2, H2).

Paramètres :
- k : taille totale de la liste (gold comprise) ; elle peut être plus courte si les
  réservoirs de distracteurs sont épuisés ;
- p_absent : probabilité de retirer la valeur correcte quand le slot est exprimé ;
- seed : graine ; le résultat est déterministe pour (seed, example.id, slot) ;
- distractors : préréglage ("easy", "same_slot", "hard"), dict ou DistractorConfig ;
- blind : rend les candidats indiscernables hors valeur (source "oracle", ni surface, ni span,
  ni score), pour qu'un juge ne puisse pas reconnaître la gold par ses métadonnées.

Trois flux aléatoires indépendants, tous dérivés de (seed, example.id, slot) :
- « absent » : un seul tirage u ~ U(0, 1), gold retirée si u < p_absent. Les exemples sans gold
  sont donc emboîtés quand p_absent augmente ;
- « distractors » : tirage séquentiel, la liste pour k est un préfixe de celle pour k+1 ;
- « order » : mélange final.
Les balayages k × p_absent comparent donc des conditions appariées.
"""
from __future__ import annotations

import random

from ..interfaces import CandidateGenerator
from ..schema import NONE_VALUE, Candidate, Example, Ontology, SlotSpec
from .base import NormalizeFn, Normalizer, as_normalizer, stable_seed
from .distractors import DistractorConfig, DistractorSampler


class OracleGenerator(CandidateGenerator):
    name = "oracle"

    def __init__(self, k: int = 5, p_absent: float = 0.0, seed: int = 0,
                 distractors: DistractorConfig | dict | str | None = "same_slot",
                 blind: bool = True, use_other_slots: bool = True,
                 normalize: NormalizeFn | Normalizer | None = None,
                 sampler: DistractorSampler | None = None):
        if k < 1:
            raise ValueError("k doit être ≥ 1")
        if not 0.0 <= p_absent <= 1.0:
            raise ValueError("p_absent doit être dans [0, 1]")
        self.k = k
        self.p_absent = p_absent
        self.seed = seed
        self.blind = blind
        self.use_other_slots = use_other_slots
        self.normalize = as_normalizer(normalize)
        self.sampler = sampler or DistractorSampler(distractors, self.normalize)

    def fit(self, train: list[Example], ontology: Ontology) -> None:
        self.sampler.fit(train, ontology)

    def _rng(self, stream: str, example: Example, slot: SlotSpec) -> random.Random:
        return random.Random(stable_seed(self.seed, stream, example.id, slot.name))

    def is_absent(self, example: Example, slot: SlotSpec) -> bool:
        """Vrai si la gold est retirée pour ce couple (indépendant de k)."""
        return self._rng("absent", example, slot).random() < self.p_absent

    def generate(self, example: Example, slot: SlotSpec, ontology: Ontology) -> list[Candidate]:
        norm = lambda v, s=slot.name: self.normalize(v, slot=s, lang=example.lang)
        gold: dict[str, Candidate] = {}
        others: list[str] = []
        for sv in example.slots:
            v = norm(sv.value, sv.name)
            if not v or v == NONE_VALUE:
                continue
            if sv.name == slot.name:
                gold.setdefault(v, Candidate(v, "oracle", None, sv.surface, sv.span))
            else:
                others.append(v)

        keep = [] if (gold and self.is_absent(example, slot)) else list(gold.values())[: self.k]
        cands = keep + self.sampler.sample(
            self.k - len(keep), example, slot, self._rng("distractors", example, slot),
            exclude=gold, reference=list(gold), other_values=others if self.use_other_slots else ())
        self._rng("order", example, slot).shuffle(cands)
        if self.blind:
            cands = [Candidate(c.value, "oracle") for c in cands]
        return cands
