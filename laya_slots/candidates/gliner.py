"""Générateur « gliner » : extraction zero-shot de spans par GLiNER (extra `candidates`).

Chaque slot devient une étiquette GLiNER (sa description, ou son nom rendu lisible). Une seule
prédiction est faite par exemple pour toutes les étiquettes (par paquets de `labels_per_call`),
puis mise en cache : les appels suivants pour d'autres slots du même exemple sont gratuits.
Un même span peut recevoir plusieurs étiquettes (multi_label), ce qui laisse au juge les
confusions départ/arrivée.

Le modèle est chargé à la première utilisation ; `model` permet d'injecter un objet qui expose
`predict_entities(text, labels, threshold=..., ...)` (tests).
"""
from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import replace
from typing import Literal

from ..interfaces import CandidateGenerator
from ..schema import NONE_VALUE, Candidate, Example, Ontology, SlotSpec
from .base import NormalizeFn, Normalizer, as_normalizer, dedupe, sort_by_score


def readable_name(slot: str) -> str:
    """« fromloc.city_name » -> « fromloc city name » ; « hotel-pricerange » -> « hotel pricerange »."""
    return re.sub(r"[._\-:/]+", " ", slot).strip().lower()


class GlinerGenerator(CandidateGenerator):
    name = "gliner"

    def __init__(self, model_name: str = "urchade/gliner_multi-v2.1", threshold: float = 0.3,
                 label: Literal["description", "name"] = "description", labels_per_call: int = 25,
                 multi_label: bool = True, device: str | None = None, cache_size: int = 256,
                 model=None, normalize: NormalizeFn | Normalizer | None = None):
        self.model_name = model_name
        self.threshold = threshold
        self.label = label
        self.labels_per_call = labels_per_call
        self.multi_label = multi_label
        self.device = device
        self.cache_size = cache_size
        self.normalize = as_normalizer(normalize)
        self._model = model
        self._cache: OrderedDict[str, dict[str, list[Candidate]]] = OrderedDict()

    def _load(self):
        if self._model is None:
            from gliner import GLiNER  # import paresseux
            self._model = GLiNER.from_pretrained(self.model_name)
            if self.device:
                self._model = self._model.to(self.device)
        return self._model

    def label_of(self, spec: SlotSpec) -> str:
        if self.label == "description" and spec.description:
            return spec.description
        return readable_name(spec.name)

    def _predict(self, text: str, labels: list[str]) -> list[dict]:
        model = self._load()
        try:
            return model.predict_entities(text, labels, threshold=self.threshold,
                                          flat_ner=not self.multi_label, multi_label=self.multi_label)
        except TypeError:   # versions de gliner sans multi_label
            return model.predict_entities(text, labels, threshold=self.threshold)

    def _all(self, example: Example, ontology: Ontology) -> dict[str, list[Candidate]]:
        key = f"{example.id}\x1f{example.utterance}\x1f{len(ontology.slots)}"
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        by_label: dict[str, list[str]] = {}
        for spec in ontology.slots.values():
            by_label.setdefault(self.label_of(spec), []).append(spec.name)
        labels = list(by_label)
        out: dict[str, list[Candidate]] = {}
        for i in range(0, len(labels), self.labels_per_call):
            for ent in self._predict(example.utterance, labels[i:i + self.labels_per_call]):
                for slot in by_label.get(ent["label"], []):
                    v = self.normalize(ent["text"], slot=slot, lang=example.lang)
                    if v and v != NONE_VALUE:
                        span = (ent["start"], ent["end"]) if "start" in ent else None
                        out.setdefault(slot, []).append(
                            Candidate(v, self.name, float(ent.get("score", 0.0)), ent["text"], span))
        out = {s: dedupe(sort_by_score(c)) for s, c in out.items()}
        self._cache[key] = out
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return out

    def generate(self, example: Example, slot: SlotSpec, ontology: Ontology) -> list[Candidate]:
        if slot.name not in ontology.slots:
            ontology = Ontology(ontology.dataset, ontology.lang, ontology.intents,
                                {**ontology.slots, slot.name: slot})
        return [replace(c) for c in self._all(example, ontology).get(slot.name, [])]
