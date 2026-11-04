"""Mesure du rappel des candidats (base de QR2 / H2).

Deux étapes :
1. `candidate_records` interroge un générateur et produit un enregistrement par requête
   (exemple, slot) : gold, candidats, rang de chaque gold dans la liste. Ces enregistrements
   se sauvegardent en JSONL et se joignent aux prédictions du juge (clé example_id + slot) ;
2. `summarize` agrège les enregistrements, globalement, par slot et par source.

Définitions (une « requête » = un couple exemple × slot interrogé ; « exprimée » = au moins une gold) :
- recall : part des valeurs gold présentes dans la liste ;
- recall@k : part des valeurs gold présentes dans les k premiers candidats (n'a de sens que pour
  des listes ordonnées, pas pour l'oracle mélangé) ;
- gold_absent_rate : part des requêtes exprimées dont la liste ne contient aucune gold ;
- mean_size : taille moyenne des listes (aussi séparément sur requêtes exprimées / non exprimées) ;
- empty_rate : part des listes vides.
Par source (une source composée « a+b » est créditée à a et à b) :
- candidates : nombre de candidats proposés ; hits : gold trouvées par cette source ;
- recall : hits / nombre total de gold ; precision : hits / candidates.

Seule l'évaluation lit les gold ici ; le générateur reçoit l'exemple tel quel.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from ..interfaces import CandidateGenerator
from ..schema import NONE_VALUE, Example, Ontology, SlotSpec
from .base import NormalizeFn, Normalizer, as_normalizer

SlotSelector = Literal["gold", "intent", "all"] | Callable[[Example, Ontology], list[SlotSpec]]
DEFAULT_KS = (1, 2, 5, 10, 20)


def query_slots(example: Example, ontology: Ontology, mode: SlotSelector = "intent") -> list[SlotSpec]:
    """Slots interrogés pour un exemple.

    "all" : tous les slots de l'ontologie ; "intent" : ceux de l'intention gold si l'ontologie les
    liste (sinon tous) ; "gold" : seulement les slots exprimés (ne mesure que le rappel).
    Les slots gold absents de l'ontologie sont toujours ajoutés (SlotSpec minimal).
    """
    if callable(mode):
        return mode(example, ontology)
    gold_names = list(dict.fromkeys(sv.name for sv in example.slots))
    if mode == "gold":
        names = gold_names
    elif mode == "intent" and example.intent in ontology.intents and ontology.intents[example.intent].slots:
        names = list(dict.fromkeys(ontology.intents[example.intent].slots + gold_names))
    else:
        names = list(dict.fromkeys(list(ontology.slots) + gold_names))
    return [ontology.slots.get(n) or SlotSpec(n, n) for n in names]


def candidate_records(examples: Iterable[Example], generator: CandidateGenerator, ontology: Ontology,
                      slots: SlotSelector = "intent",
                      normalize: NormalizeFn | Normalizer | None = None) -> Iterable[dict]:
    """Un enregistrement par requête : example_id, slot, gold, candidates, sources, gold_ranks."""
    norm = as_normalizer(normalize)
    for ex in examples:
        gold_by_slot: dict[str, list[str]] = defaultdict(list)
        for sv in ex.slots:
            v = norm(sv.value, slot=sv.name, lang=ex.lang)
            if v and v != NONE_VALUE and v not in gold_by_slot[sv.name]:
                gold_by_slot[sv.name].append(v)
        for spec in query_slots(ex, ontology, slots):
            cands = generator.generate(ex, spec, ontology)
            values = [c.value for c in cands]
            pos = {}
            for i, v in enumerate(values):
                pos.setdefault(v, i)
            gold = gold_by_slot.get(spec.name, [])
            yield {
                "example_id": ex.id,
                "slot": spec.name,
                "gold": gold,
                "candidates": values,
                "sources": [c.source for c in cands],
                "gold_ranks": [pos.get(g) for g in gold],
            }


@dataclass
class _Acc:
    queries: int = 0
    expressed: int = 0
    gold: int = 0
    found: int = 0
    absent: int = 0
    size: int = 0
    size_expressed: int = 0
    empty: int = 0
    ranks: Counter = field(default_factory=Counter)

    def add(self, rec: dict) -> None:
        n = len(rec["candidates"])
        self.queries += 1
        self.size += n
        self.empty += n == 0
        if rec["gold"]:
            self.expressed += 1
            self.size_expressed += n
            self.gold += len(rec["gold"])
            hits = [r for r in rec["gold_ranks"] if r is not None]
            self.found += len(hits)
            self.absent += not hits
            self.ranks.update(hits)

    def to_dict(self, ks: Sequence[int]) -> dict:
        div = lambda a, b: a / b if b else None
        unexpressed = self.queries - self.expressed
        d = {
            "queries": self.queries, "expressed": self.expressed, "gold": self.gold,
            "recall": div(self.found, self.gold),
            "gold_absent_rate": div(self.absent, self.expressed),
            "mean_size": div(self.size, self.queries),
            "mean_size_expressed": div(self.size_expressed, self.expressed),
            "mean_size_unexpressed": div(self.size - self.size_expressed, unexpressed),
            "empty_rate": div(self.empty, self.queries),
        }
        for k in ks:
            d[f"recall@{k}"] = div(sum(c for r, c in self.ranks.items() if r < k), self.gold)
        return d


def summarize(records: Iterable[dict], ks: Sequence[int] = DEFAULT_KS) -> dict:
    """Agrège les enregistrements : {"overall": ..., "by_slot": {...}, "by_source": {...}}."""
    overall, by_slot = _Acc(), defaultdict(_Acc)
    src_cands: Counter = Counter()
    src_hits: Counter = Counter()
    for rec in records:
        overall.add(rec)
        by_slot[rec["slot"]].add(rec)
        for s in rec["sources"]:
            for part in s.split("+"):
                src_cands[part] += 1
        for r in rec["gold_ranks"]:
            if r is not None:
                for part in rec["sources"][r].split("+"):
                    src_hits[part] += 1
    by_source = {
        s: {"candidates": src_cands[s], "hits": src_hits[s],
            "recall": src_hits[s] / overall.gold if overall.gold else None,
            "precision": src_hits[s] / src_cands[s] if src_cands[s] else None}
        for s in sorted(src_cands)
    }
    return {"overall": overall.to_dict(ks),
            "by_slot": {s: a.to_dict(ks) for s, a in sorted(by_slot.items())},
            "by_source": by_source}


def candidate_recall(examples: Iterable[Example], generator: CandidateGenerator, ontology: Ontology,
                     slots: SlotSelector = "intent", ks: Sequence[int] = DEFAULT_KS,
                     normalize: NormalizeFn | Normalizer | None = None) -> dict:
    """Raccourci : `summarize(candidate_records(...))`."""
    return summarize(candidate_records(examples, generator, ontology, slots, normalize), ks)


def report_to_frame(report: dict):
    """Rapport -> DataFrame pandas (une ligne pour l'ensemble, puis une par slot)."""
    import pandas as pd
    rows = [{"slot": "__all__", **report["overall"]}]
    rows += [{"slot": s, **d} for s, d in report["by_slot"].items()]
    return pd.DataFrame(rows)
