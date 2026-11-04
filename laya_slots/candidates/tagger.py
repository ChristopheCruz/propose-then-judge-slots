"""Générateur « tagger » : spans d'un tagger BIO fine-tuné (C0, style JointBERT sans tête d'intention).

L'énoncé est découpé en mots par `base.tokens` (le même découpage que les n-grammes) ; le tagger
prédit une étiquette par mot (O, B-<slot>, I-<slot>), lue sur le premier sous-mot. Les spans sont
décodés de façon tolérante : un I-<slot> qui ne suit pas le même slot ouvre un span. Score d'un
span : moyenne des probabilités de l'étiquette prédite sur ses mots.

Une seule passe avant par exemple (mise en cache) : les appels suivants pour d'autres slots du
même exemple sont gratuits. `model` permet d'injecter un objet qui expose
`predict(words) -> [(étiquette, probabilité), ...]` (tests).

L'entraînement est dans experiments.train_tagger (make entrainer-c0) ; `word_tags` y sert à
construire les étiquettes gold, ce qui garantit le même découpage à l'entraînement et à l'inférence.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import replace

from ..interfaces import CandidateGenerator
from ..schema import NONE_VALUE, Candidate, Example, Ontology, SlotSpec, SlotValue
from .base import NormalizeFn, Normalizer, as_normalizer, dedupe, sort_by_score, tokens

OUTSIDE = "O"


def word_tags(text: str, slots: Iterable[SlotValue]) -> tuple[list[tuple[str, int, int]], list[str], dict]:
    """Mots de `text` (texte, début, fin) et étiquettes BIO gold, avec quelques effectifs.

    Seuls les slots qui ont un span dans l'énoncé comptent. Un mot qui chevauche le span reçoit
    l'étiquette ; un mot déjà étiqueté par un autre slot garde la première (conflit compté).
    """
    words = tokens(text)
    tags = [OUTSIDE] * len(words)
    stats = {"spans": 0, "conflicts": 0, "misaligned": 0, "empty": 0}
    for sv in slots:
        if sv.span is None or sv.origin != "utterance":
            continue
        s, e = sv.span
        idx = [i for i, (_, ws, we) in enumerate(words) if ws < e and we > s]
        if not idx:
            stats["empty"] += 1
            continue
        stats["spans"] += 1
        if words[idx[0]][1] != s or words[idx[-1]][2] != e:
            stats["misaligned"] += 1
        if any(tags[i] != OUTSIDE for i in idx):
            stats["conflicts"] += 1
            continue
        for n, i in enumerate(idx):
            tags[i] = f"{'B' if n == 0 else 'I'}-{sv.name}"
    return words, tags, stats


def decode(tags: list[str], probs: list[float] | None = None) -> list[tuple[str, int, int, float]]:
    """Spans (slot, premier mot, dernier mot inclus, score) d'une séquence BIO."""
    probs = probs if probs is not None else [1.0] * len(tags)
    spans, cur = [], None          # cur = [slot, début, fin, probabilités]
    for i, tag in enumerate(tags):
        prefix, _, slot = tag.partition("-")
        if tag == OUTSIDE or not slot:
            if cur:
                spans.append(cur)
            cur = None
        elif prefix == "I" and cur and cur[0] == slot:
            cur[2], cur[3] = i, cur[3] + [probs[i]]
        else:
            if cur:
                spans.append(cur)
            cur = [slot, i, i, [probs[i]]]
    if cur:
        spans.append(cur)
    return [(s, a, b, sum(p) / len(p)) for s, a, b, p in spans]


class TaggerModel:
    """Tagger HF (AutoModelForTokenClassification) : étiquette et probabilité par mot."""

    def __init__(self, checkpoint: str, device: str | None = None, max_len: int = 256):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForTokenClassification.from_pretrained(checkpoint, dtype=torch.float32).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.id2label = {int(i): lab for i, lab in self.model.config.id2label.items()}
        self.max_len = max_len

    def predict(self, words: list[str]) -> list[tuple[str, float]]:
        if not words:
            return []
        enc = self.tokenizer(words, is_split_into_words=True, truncation=True, max_length=self.max_len,
                             return_tensors="pt")
        word_ids = enc.word_ids(0)
        with self.torch.inference_mode():
            logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits[0]
        p, lab = logits.float().softmax(-1).max(-1)
        out: list[tuple[str, float] | None] = [None] * len(words)
        for t, w in enumerate(word_ids):
            if w is not None and out[w] is None:
                out[w] = (self.id2label[int(lab[t])], float(p[t]))
        return [o or (OUTSIDE, 1.0) for o in out]     # mots tronqués : hors slot


class TaggerGenerator(CandidateGenerator):
    name = "tagger"

    def __init__(self, checkpoint: str | None = None, top_k: int | None = None, min_score: float = 0.0,
                 device: str | None = None, max_len: int = 256, cache_size: int = 256, model=None,
                 normalize: NormalizeFn | Normalizer | None = None):
        if checkpoint is None and model is None:
            raise ValueError("TaggerGenerator : checkpoint (make entrainer-c0) ou model requis")
        self.checkpoint = checkpoint
        self.top_k = top_k
        self.min_score = min_score
        self.device = device
        self.max_len = max_len
        self.cache_size = cache_size
        self.normalize = as_normalizer(normalize)
        self._model = model
        self._cache: OrderedDict[str, dict[str, list[Candidate]]] = OrderedDict()

    def _load(self):
        if self._model is None:
            self._model = TaggerModel(self.checkpoint, self.device, self.max_len)
        return self._model

    def _all(self, example: Example) -> dict[str, list[Candidate]]:
        key = f"{example.id}\x1f{example.utterance}"
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        words = tokens(example.utterance)
        pred = self._load().predict([w for w, _, _ in words])
        out: dict[str, list[Candidate]] = {}
        for slot, a, b, score in decode([t for t, _ in pred], [p for _, p in pred]):
            if score < self.min_score:
                continue
            start, end = words[a][1], words[b][2]
            surface = example.utterance[start:end]
            v = self.normalize(surface, slot=slot, lang=example.lang)
            if v and v != NONE_VALUE:
                out.setdefault(slot, []).append(Candidate(v, self.name, score, surface, (start, end)))
        out = {s: dedupe(sort_by_score(c)) for s, c in out.items()}
        self._cache[key] = out
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return out

    def generate(self, example: Example, slot: SlotSpec, ontology: Ontology) -> list[Candidate]:
        cands = self._all(example).get(slot.name, [])
        return [replace(c) for c in cands[: self.top_k]]
