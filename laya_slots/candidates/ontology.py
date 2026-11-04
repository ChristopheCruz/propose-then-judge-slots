"""Générateur « ontology » : valeurs fermées du slot dont la forme apparaît dans l'énoncé.

Appariement :
- exact (après normalisation, sur des frontières de mots) : score 1.0 ;
- approché : meilleur ratio difflib entre la valeur et un n-gramme de l'énoncé de longueur
  voisine (± 1 mot), retenu si ≥ `threshold`. Les valeurs de moins de `min_fuzzy_len` caractères
  ne sont appariées qu'exactement (sinon trop de bruit : « no », « 2 »...).

mode="all" renvoie toutes les valeurs fermées sans appariement (score None), à combiner avec
`EmbeddingTopK` pour les slots à grande ontologie.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Literal

from ..interfaces import CandidateGenerator
from ..schema import NONE_VALUE, Candidate, Example, Ontology, SlotSpec
from .base import NormalizeFn, Normalizer, as_normalizer, dedupe, ngrams, sort_by_score, tokens


class OntologyGenerator(CandidateGenerator):
    name = "ontology"

    def __init__(self, threshold: float = 0.85, mode: Literal["match", "all"] = "match",
                 min_fuzzy_len: int = 4, search_history: bool = False, history_weight: float = 0.5,
                 aliases: dict[str, dict[str, list[str]]] | None = None,
                 normalize: NormalizeFn | Normalizer | None = None):
        """aliases : slot -> valeur -> formes de surface supplémentaires (ex. « centre » -> centre-ville)."""
        self.threshold = threshold
        self.mode = mode
        self.min_fuzzy_len = min_fuzzy_len
        self.search_history = search_history
        self.history_weight = history_weight
        self.aliases = aliases or {}
        self.normalize = as_normalizer(normalize)

    def _match(self, form: str, spans: list[tuple[str, int, tuple[int, int], str]]):
        """Meilleur appariement de `form` parmi les n-grammes : (score, surface, span) ou None."""
        n = len(tokens(form))
        best = None
        sm = SequenceMatcher(autojunk=False)
        sm.set_seq2(form)
        for norm_text, n_tok, span, surface in spans:
            if norm_text == form:
                return 1.0, surface, span
            if len(form) < self.min_fuzzy_len or abs(n_tok - n) > 1:
                continue
            sm.set_seq1(norm_text)
            if sm.real_quick_ratio() < self.threshold or sm.quick_ratio() < self.threshold:
                continue
            r = sm.ratio()
            if r >= self.threshold and (best is None or r > best[0]):
                best = (r, surface, span)
        return best

    def _spans(self, text: str, slot: SlotSpec, lang: str, max_n: int):
        return [(self.normalize(sp.text, slot=slot.name, lang=lang), sp.n_tokens, (sp.start, sp.end), sp.text)
                for sp in ngrams(text, max_n)]

    def generate(self, example: Example, slot: SlotSpec, ontology: Ontology) -> list[Candidate]:
        values = slot.values or []
        norm = lambda v: self.normalize(v, slot=slot.name, lang=example.lang)
        if self.mode == "all":
            return dedupe(Candidate(norm(v), self.name) for v in values)
        forms: list[tuple[str, str]] = []           # (valeur normalisée, forme cherchée)
        for v in values:
            nv = norm(v)
            if not nv or nv == NONE_VALUE:
                continue
            forms.append((nv, nv))
            forms.extend((nv, norm(a)) for a in self.aliases.get(slot.name, {}).get(v, []))
        if not forms:
            return []
        max_n = max(len(tokens(f)) for _, f in forms) + 1
        utt = self._spans(example.utterance, slot, example.lang, max_n)
        hist = []
        if self.search_history:
            hist = [self._spans(t.text, slot, example.lang, max_n) for t in example.history]
        out = []
        for nv, form in forms:
            m = self._match(form, utt)
            if m:
                out.append(Candidate(nv, self.name, m[0], m[1], m[2]))
                continue
            for h in reversed(hist):
                m = self._match(form, h)
                if m:
                    out.append(Candidate(nv, self.name, m[0] * self.history_weight, m[1], None))
                    break
        return dedupe(sort_by_score(out))
