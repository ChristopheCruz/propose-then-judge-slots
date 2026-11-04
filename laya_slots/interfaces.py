"""Interfaces communes — CONTRAT PARTAGÉ (voir CONTRAT.md).

Les implémentations vivent dans data/, candidates/, judges/ ; elles doivent respecter ces protocoles.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from .schema import Candidate, Example, Judgment, JudgeRequest, Ontology, SlotSpec, Split


class DatasetLoader(ABC):
    """Convertit un jeu brut vers le format unifié."""
    name: str

    @abstractmethod
    def languages(self) -> list[str]: ...

    @abstractmethod
    def load(self, lang: str, split: Split) -> Iterator[Example]: ...

    @abstractmethod
    def ontology(self, lang: str) -> Ontology: ...


class CandidateGenerator(ABC):
    """Propose des valeurs candidates pour un slot d'un exemple."""
    name: str

    def fit(self, train: list[Example], ontology: Ontology) -> None:
        """Optionnel : apprentissage ou indexation sur le train."""

    @abstractmethod
    def generate(self, example: Example, slot: SlotSpec, ontology: Ontology) -> list[Candidate]: ...


class Judge(ABC):
    """Choisit un candidat (ou « non précisé ») pour un slot.

    Implémenter judge_batch ; judge() est fourni. Un juge ne doit jamais lire
    request.example.slots (les gold) : c'est vérifié par les tests.
    """
    name: str

    @abstractmethod
    def judge_batch(self, requests: list[JudgeRequest]) -> list[Judgment]: ...

    def judge(self, request: JudgeRequest) -> Judgment:
        return self.judge_batch([request])[0]
