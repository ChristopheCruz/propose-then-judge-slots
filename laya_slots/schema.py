"""Format unifié du projet — CONTRAT PARTAGÉ.

Ne pas modifier sans passer par le coordinateur (voir CONTRAT.md).
Tous les modules échangent ces objets ; ils se sérialisent en JSONL (un objet par ligne).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Literal

Dataset = Literal["multiatis", "mtop", "multi3woz"]
Split = Literal["train", "dev", "test"]
Speaker = Literal["user", "system"]

# Valeur réservée : le slot n'est pas exprimé dans le tour / « non précisé ».
NONE_VALUE = "__none__"


# ---------------------------------------------------------------------------
# Exemples annotés
# ---------------------------------------------------------------------------
@dataclass
class Turn:
    speaker: Speaker
    text: str


@dataclass
class SlotValue:
    """Une valeur de slot de référence (gold).

    value : valeur NORMALISÉE (voir laya_slots.normalize) ; c'est elle qui est comparée.
    surface : texte tel qu'il apparaît dans l'énoncé, si applicable.
    span : offsets caractères [début, fin) dans `utterance`, ou None si la valeur
           ne vient pas de l'énoncé courant (historique, contexte) ou si le jeu ne fournit pas de spans.
    origin : d'où vient la valeur.
    """
    name: str
    value: str
    surface: str | None = None
    span: tuple[int, int] | None = None
    origin: Literal["utterance", "history", "context"] = "utterance"


@dataclass
class Example:
    """Un tour utilisateur à analyser.

    id : "<dataset>:<lang>:<split>:<identifiant d'origine>[:<n° de tour>]", unique et stable.
    history : tours PRÉCÉDENTS (vide pour les jeux mono-tour).
    slots : slots exprimés DANS CE TOUR (niveau tour, pas l'état cumulé du dialogue).
    state_before : état cumulé avant ce tour (slot -> valeur), pour les jeux multi-tours.
    """
    id: str
    dataset: Dataset
    lang: str
    split: Split
    utterance: str
    intent: str | None
    domain: str | None = None
    slots: list[SlotValue] = field(default_factory=list)
    history: list[Turn] = field(default_factory=list)
    state_before: dict[str, str] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict) -> "Example":
        d = dict(d)
        d["slots"] = [SlotValue(**{**s, "span": tuple(s["span"]) if s.get("span") else None})
                      for s in d.get("slots", [])]
        d["history"] = [Turn(**t) for t in d.get("history", [])]
        return Example(**d)


# ---------------------------------------------------------------------------
# Ontologie (une par jeu et par langue : ontology.json)
# ---------------------------------------------------------------------------
@dataclass
class SlotSpec:
    name: str
    description: str
    values: list[str] | None = None      # valeurs fermées connues (catégoriel), sinon None


@dataclass
class IntentSpec:
    name: str
    description: str
    domain: str | None = None
    slots: list[str] = field(default_factory=list)   # slots autorisés pour cette intention


@dataclass
class Ontology:
    dataset: Dataset
    lang: str
    intents: dict[str, IntentSpec]
    slots: dict[str, SlotSpec]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @staticmethod
    def from_dict(d: dict) -> "Ontology":
        return Ontology(
            dataset=d["dataset"], lang=d["lang"],
            intents={k: IntentSpec(**v) for k, v in d["intents"].items()},
            slots={k: SlotSpec(**v) for k, v in d["slots"].items()},
        )


# ---------------------------------------------------------------------------
# Candidats et décisions du juge
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    """Une valeur candidate pour un slot.

    value : valeur normalisée (comparable à SlotValue.value).
    source : générateur d'origine — "oracle", "tagger", "gliner", "ontology", "history", "distractor", "context"...
    score : score du générateur (non calibré), si disponible.
    """
    value: str
    source: str
    score: float | None = None
    surface: str | None = None
    span: tuple[int, int] | None = None


@dataclass
class JudgeRequest:
    """Une question posée au juge : quel candidat pour `slot` dans `example` ?

    La liste `candidates` NE contient PAS l'option « non précisé » : chaque juge l'ajoute
    lui-même. `candidates` peut être vide.
    filled : slots déjà décidés pour ce tour ou l'état du dialogue (contexte pour le juge).
    """
    example: Example
    slot: SlotSpec
    candidates: list[Candidate]
    intent: str | None = None
    filled: dict[str, str] = field(default_factory=dict)


@dataclass
class Judgment:
    """Réponse du juge.

    choice : indice dans request.candidates, ou None pour « non précisé ».
    proba : probabilité du choix retenu (non calibrée ; la calibration est faite dans decision/).
    probs : distribution complète, len(candidates) + 1, la dernière case = « non précisé ». Optionnel.
    p_expressed : P(le slot est exprimé), pour la décision en deux temps (H4). Optionnel.
    latency_ms : temps de calcul attribuable à cette requête.
    """
    choice: int | None
    proba: float
    probs: list[float] | None = None
    p_expressed: float | None = None
    latency_ms: float | None = None
    raw: dict | None = None

    def value(self, request: JudgeRequest) -> str:
        return NONE_VALUE if self.choice is None else request.candidates[self.choice].value


# ---------------------------------------------------------------------------
# E/S JSONL
# ---------------------------------------------------------------------------
def _default(o):
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)
    raise TypeError(type(o))


def write_jsonl(path: str | Path, objs: Iterable) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(asdict(o) if hasattr(o, "__dataclass_fields__") else o,
                               ensure_ascii=False, default=_default) + "\n")
            n += 1
    return n


def read_examples(path: str | Path) -> Iterator[Example]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Example.from_dict(json.loads(line))
