"""Outils partagés par les juges : clés d'options, ordre des candidats, contexte, probabilités.

Règle de non-fuite : on ne lit de `request.example` que `id`, `lang`, `utterance` et `history`.
`example.slots`, `example.intent`, `example.state_before` et `example.meta` peuvent contenir
la vérité terrain ; l'intention et les slots déjà remplis viennent de `request.intent` et
`request.filled`.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections.abc import Sequence

from ..schema import Candidate, JudgeRequest, Judgment

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# Nombre maximal de candidats par question (hors « non précisé ») : une lettre par option.
MAX_LETTER_OPTIONS = len(LETTERS) - 1

LANG_NAMES = {"en": "English", "fr": "French", "de": "German", "es": "Spanish", "pt": "Portuguese",
              "it": "Italian", "tr": "Turkish", "ar": "Arabic", "zh": "Chinese", "ja": "Japanese",
              "hi": "Hindi"}


def option_keys(n: int) -> list[str]:
    """Clés neutres A, B, C… pour n options (n ≤ 26)."""
    if n > len(LETTERS):
        raise ValueError(f"{n} options : au plus {len(LETTERS)} clés-lettres disponibles")
    return list(LETTERS[:n])


def request_rng(seed: int, request: JudgeRequest, salt: str = "") -> random.Random:
    """Générateur déterministe par requête : même résultat quel que soit l'ordre du lot."""
    key = f"{seed}|{salt}|{request.example.id}|{request.slot.name}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


def order_candidates(request: JudgeRequest, policy: str = "shuffle", seed: int = 0,
                     max_options: int | None = None) -> tuple[list[int], int]:
    """Indices (dans request.candidates) des candidats présentés, dans l'ordre de présentation.

    policy : "given" (ordre du générateur), "score" (score décroissant, None en dernier, ordre
    stable), "shuffle" (permutation aléatoire déterministe par requête).
    Troncature : au-delà de max_options, on garde les premiers de l'ordre "score" si des scores
    existent, sinon ceux de l'ordre donné ; la troncature ne dépend donc pas du tirage.
    Renvoie (indices, nombre de candidats écartés).
    """
    n = len(request.candidates)
    idx = list(range(n))
    dropped = 0
    if max_options is not None and n > max_options:
        if any(c.score is not None for c in request.candidates):
            idx = _by_score(request.candidates)
        idx = idx[:max_options]
        dropped = n - max_options
        idx.sort()        # on repart de l'ordre du générateur avant d'appliquer la politique
    if policy == "given":
        pass
    elif policy == "score":
        rank = {i: r for r, i in enumerate(_by_score(request.candidates))}
        idx.sort(key=lambda i: rank[i])
    elif policy == "shuffle":
        request_rng(seed, request, "order").shuffle(idx)
    else:
        raise ValueError(f"politique d'ordre inconnue : {policy!r}")
    return idx, dropped


def _by_score(cands: Sequence[Candidate]) -> list[int]:
    return sorted(range(len(cands)),
                  key=lambda i: (cands[i].score is None, -(cands[i].score or 0.0), i))


def format_history(request: JudgeRequest, n_turns: int, max_chars: int) -> str:
    """Les n derniers tours, tronqués par la gauche (on garde les plus récents)."""
    turns = request.example.history[-n_turns:] if n_turns > 0 else []
    text = "\n".join(f"{t.speaker}: {t.text}" for t in turns)
    if max_chars and len(text) > max_chars:
        text = "…" + text[-max_chars:]
    return text


def format_filled(filled: dict[str, str]) -> str:
    return json.dumps(filled, ensure_ascii=False) if filled else ""


def clip(text: str, max_chars: int) -> str:
    return text if not max_chars or len(text) <= max_chars else text[: max_chars - 1] + "…"


def candidate_text(c: Candidate) -> str:
    """Texte montré au juge : la forme de surface si elle existe, sinon la valeur normalisée."""
    return c.surface or c.value


def normalize(values: Sequence[float]) -> list[float]:
    s = float(sum(values))
    if s <= 0:
        return [1.0 / len(values)] * len(values) if values else []
    return [float(v) / s for v in values]


def judgment_from_probs(probs: list[float], latency_ms: float | None = None,
                        p_expressed: float | None = None, raw: dict | None = None) -> Judgment:
    """Judgment à partir d'une distribution len(candidates)+1 (dernière case = « non précisé »)."""
    best = max(range(len(probs)), key=lambda i: (probs[i], -i))
    choice = None if best == len(probs) - 1 else best
    return Judgment(choice=choice, proba=probs[best], probs=probs, p_expressed=p_expressed,
                    latency_ms=latency_ms, raw=raw)


def fold_text(text: str) -> str:
    """Minuscules, sans accents, espaces normalisés (comparaison de surface)."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip()
