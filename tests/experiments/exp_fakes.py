"""Faux composants et petite fixture pour tester experiments/ sans les vrais modules.

Les configs de test les désignent par `class: exp_fakes.X` (le dossier de ce fichier est
dans sys.path pendant les tests ; ajouter PYTHONPATH pour un sous-processus).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from laya_slots.interfaces import CandidateGenerator, DatasetLoader, Judge
from laya_slots.schema import (
    Candidate,
    Example,
    IntentSpec,
    Judgment,
    Ontology,
    SlotSpec,
    SlotValue,
    write_jsonl,
)

CITIES = ["boston", "denver", "dallas", "atlanta", "seattle", "miami"]


def make_ontology() -> Ontology:
    return Ontology(
        dataset="multiatis", lang="en",
        intents={"atis_flight": IntentSpec("atis_flight", "find a flight",
                                           slots=["fromloc.city_name", "toloc.city_name"]),
                 "atis_airfare": IntentSpec("atis_airfare", "price of a flight",
                                            slots=["fromloc.city_name", "toloc.city_name", "cost_relative"])},
        slots={"fromloc.city_name": SlotSpec("fromloc.city_name", "departure city", CITIES),
               "toloc.city_name": SlotSpec("toloc.city_name", "arrival city", CITIES),
               "cost_relative": SlotSpec("cost_relative", "relative cost", ["cheapest", "expensive"])},
    )


def make_examples(n: int = 12, split: str = "test") -> list[Example]:
    rng = random.Random(0)
    out = []
    for i in range(n):
        a, b = rng.sample(CITIES, 2)
        slots = [SlotValue("fromloc.city_name", a, a), SlotValue("toloc.city_name", b, b)]
        intent, text = "atis_flight", f"flights from {a} to {b}"
        if i % 3 == 0:
            intent, text = "atis_airfare", f"cheapest fare from {a} to {b}"
            slots.append(SlotValue("cost_relative", "cheapest", "cheapest"))
        if i % 4 == 1:   # slot non exprimé
            slots = slots[:1]
            text = f"flights from {a}"
        out.append(Example(id=f"multiatis:en:{split}:{i}", dataset="multiatis", lang="en", split=split,
                           utterance=text, intent=intent, slots=slots))
    return out


def write_fixture(root: Path, n: int = 12) -> Path:
    """Écrit <root>/multiatis/en/{train,test}.jsonl + ontology.json ; rend le dossier."""
    d = root / "multiatis" / "en"
    write_jsonl(d / "test.jsonl", make_examples(n, "test"))
    write_jsonl(d / "train.jsonl", make_examples(n, "train"))
    (d / "ontology.json").write_text(make_ontology().to_json(), encoding="utf-8")
    return d


class FakeLoader(DatasetLoader):
    name = "fake"

    def __init__(self, n: int = 12):
        self.n = n

    def languages(self):
        return ["en"]

    def load(self, lang, split):
        yield from make_examples(self.n, split)

    def ontology(self, lang):
        return make_ontology()


class FakeOracle(CandidateGenerator):
    """Oracle : la valeur gold (sauf avec proba p_absent) + des distracteurs de l'ontologie."""
    name = "oracle"

    def __init__(self, k: int = 3, p_absent: float = 0.0, seed: int = 0):
        self.k, self.p_absent, self.seed = k, p_absent, seed
        self.fitted = False

    def fit(self, train, ontology):
        self.fitted = len(train) > 0

    def generate(self, example, slot, ontology):
        rng = random.Random(f"{self.seed}:{example.id}:{slot.name}")
        true = [s.value for s in example.slots if s.name == slot.name]
        gold = [] if true and rng.random() < self.p_absent else true
        pool = [v for v in (slot.values or []) if v not in true]
        vals = gold + rng.sample(pool, max(0, min(len(pool), self.k - len(gold))))
        rng.shuffle(vals)
        return [Candidate(v, "oracle" if v in gold else "distractor") for v in vals]


class EmptyGenerator(CandidateGenerator):
    name = "empty"

    def generate(self, example, slot, ontology):
        return []


class LexicalJudge(Judge):
    """Choisit le candidat présent dans l'énoncé à la bonne place (après from/to), sinon abstient.
    Vérifie au passage qu'il ne voit jamais les gold."""
    name = "lexical"

    def __init__(self, sleep_s: float = 0.0):
        self.sleep_s = sleep_s
        self.calls = 0

    def judge_batch(self, requests):
        self.calls += 1
        if self.sleep_s:
            time.sleep(self.sleep_s)
        out = []
        for r in requests:
            assert r.example.slots == [], "fuite : le juge voit les slots gold"
            words = r.example.utterance.split()
            prev = {"fromloc.city_name": "from", "toloc.city_name": "to"}.get(r.slot.name)
            choice = None
            for i, c in enumerate(r.candidates):
                if prev is None and c.value in words or prev and any(words[k] == prev and words[k + 1] == c.value for k in range(len(words) - 1)):
                    choice = i
            n = len(r.candidates) + 1
            probs = [0.1 / max(n - 1, 1)] * n
            probs[n - 1 if choice is None else choice] = 0.9
            out.append(Judgment(choice=choice, proba=0.9, probs=probs))
        return out


class FirstJudge(Judge):
    """Choisit toujours le premier candidat."""
    name = "first"

    def judge_batch(self, requests):
        return [Judgment(choice=0 if r.candidates else None, proba=0.5, latency_ms=1.0) for r in requests]


class FakeIntentPredictor:
    def __init__(self, answer: str = "atis_flight"):
        self.answer = answer

    def predict_batch(self, examples):
        assert all(ex.intent is None and ex.slots == [] for ex in examples)
        return [self.answer] * len(examples)


def write_config(path: Path, data_root: Path, **extra) -> Path:
    cfg = {
        "name": "fake", "condition": "test", "seed": 7,
        "data": {"dataset": "multiatis", "lang": "en", "split": "test", "root": str(data_root)},
        "candidates": {"generators": [{"class": "exp_fakes.FakeOracle", "params": {"k": 3}}]},
        "judge": {"class": "exp_fakes.LexicalJudge", "params": {}, "batch_size": 4},
    }
    for k, v in extra.items():
        cfg[k] = v
    path.write_text(json.dumps(cfg), encoding="utf-8")   # le JSON est du YAML valide
    return path
