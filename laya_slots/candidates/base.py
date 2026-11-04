"""Outils communs aux générateurs de candidats.

- normalisation injectable : par défaut `project_normalize`, qui délègue à
  `laya_slots.normalize.normalize_value` (la même que pour les gold) ;
  `minimal_normalize` reste disponible pour les tests ;
- graines dérivées stables (indépendantes de PYTHONHASHSEED) ;
- découpage de l'énoncé en n-grammes avec offsets caractères ;
- déduplication et troncature des listes de candidats.
"""
from __future__ import annotations

import hashlib
import inspect
import re
import unicodedata
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import lru_cache

from ..normalize import normalize_value
from ..schema import NONE_VALUE, Candidate

NormalizeFn = Callable[..., str]

_SPACES = re.compile(r"\s+")
# Un « mot » : élision française (l', d', qu'...) détachée du mot suivant, sinon suite
# alphanumérique avec apostrophes et tirets internes (don't, saint-denis), ou un signe isolé.
_TOKEN = re.compile(r"(?i:\b(?:jusqu|lorsqu|puisqu|qu|[cdjlmnst])['’](?=\w))|\w+(?:[-'’]\w+)*|[^\w\s]")


def minimal_normalize(value: str) -> str:
    """Normalisation minimale locale : NFC, minuscules, espaces compactés."""
    return _SPACES.sub(" ", unicodedata.normalize("NFC", value).lower()).strip()


def project_normalize(value: str, slot: str | None = None, lang: str | None = None) -> str:
    """Normalisation du projet, identique à celle appliquée aux gold par les chargeurs."""
    return normalize_value(slot or "", value, lang or "en")


class Normalizer:
    """Enveloppe une fonction de normalisation.

    La fonction reçoit la valeur, et en plus `slot` et `lang` si sa signature les accepte
    (une normalisation de dates ou de nombres peut dépendre du slot et de la langue).
    """

    def __init__(self, fn: NormalizeFn | None = None):
        self.fn = fn or project_normalize
        try:
            params = inspect.signature(self.fn).parameters
        except (TypeError, ValueError):
            params = {}
        var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        self._slot = var_kw or "slot" in params
        self._lang = var_kw or "lang" in params

    def __call__(self, value: str, slot: str | None = None, lang: str | None = None) -> str:
        kw = {}
        if self._slot:
            kw["slot"] = slot
        if self._lang:
            kw["lang"] = lang
        return self.fn(value, **kw)


def as_normalizer(fn: NormalizeFn | Normalizer | None) -> Normalizer:
    return fn if isinstance(fn, Normalizer) else Normalizer(fn)


def stable_seed(*parts: object) -> int:
    """Graine entière stable dérivée de `parts` (sha256, indépendante du processus)."""
    h = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


# ---------------------------------------------------------------------------
# N-grammes de l'énoncé
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    text: str                 # texte de surface
    start: int                # offset caractère [start, end)
    end: int
    n_tokens: int


def tokens(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(text)]


def ngrams(text: str, max_n: int = 4, min_n: int = 1) -> Iterator[Span]:
    """Tous les n-grammes de mots (min_n ≤ n ≤ max_n), avec leurs offsets dans `text`."""
    toks = tokens(text)
    for i in range(len(toks)):
        for n in range(min_n, max_n + 1):
            j = i + n
            if j > len(toks):
                break
            s, e = toks[i][1], toks[j - 1][2]
            yield Span(text[s:e], s, e, n)


# Mots vides EN/FR : un n-gramme formé uniquement de ces mots (ou de ponctuation)
# n'est pas un candidat plausible.
_STOPWORDS_TEXT = """
a an the of to from in on at for by with and or but is are was were be been am i me my we us our you your
he she it they them this that these those what which who whom where when how please can could would will
do does did have has had not no yes any some all just want like need show list give get find tell there
here about into as if so than then also okay ok hi hello thanks thank
le la les l' un une des du de d' au aux à a et ou mais est sont était être je j' me m' moi mon ma mes nous
notre nos vous votre vos il elle ils elles on ce cet cette ces c' ça qui que qu' quoi où quand comment
pour par avec sans sur dans en y ne n' pas plus oui non s'il plait plaît veux voudrais voulons aimerais
peux pouvez pourriez faut merci bonjour salut alors donc aussi très tout tous toute toutes se s' sa son ses
"""
STOPWORDS = frozenset(_STOPWORDS_TEXT.split())


def is_content_span(text: str, normalize: Callable[[str], str] = minimal_normalize) -> bool:
    """Vrai si le n-gramme contient au moins un mot plein (ni mot vide, ni ponctuation)."""
    for tok, _, _ in tokens(normalize(text)):
        if tok.replace("’", "'") not in STOPWORDS and any(c.isalnum() for c in tok):
            return True
    return False


# ---------------------------------------------------------------------------
# Listes de candidats
# ---------------------------------------------------------------------------
def dedupe(candidates: Iterable[Candidate], merge_sources: bool = False) -> list[Candidate]:
    """Supprime les doublons de valeur (les valeurs sont déjà normalisées) et les NONE_VALUE.

    Garde la première occurrence ; son score devient le maximum des scores des doublons.
    merge_sources : la source devient "a+b" quand plusieurs générateurs proposent la même valeur.
    """
    out: dict[str, Candidate] = {}
    for c in candidates:
        if not c.value or c.value == NONE_VALUE:
            continue
        prev = out.get(c.value)
        if prev is None:
            out[c.value] = Candidate(c.value, c.source, c.score, c.surface, c.span)
            continue
        if c.score is not None and (prev.score is None or c.score > prev.score):
            prev.score = c.score
        if prev.span is None and c.span is not None:
            prev.surface, prev.span = c.surface, c.span
        if merge_sources and c.source not in prev.source.split("+"):
            prev.source = f"{prev.source}+{c.source}"
    return list(out.values())


def char_similarity(a: str, b: str) -> float:
    """Similarité lexicale : Jaccard des trigrammes de caractères (chaînes bordées d'espaces)."""
    if a == b:
        return 1.0
    ta, tb = _trigrams(a), _trigrams(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


@lru_cache(maxsize=200_000)
def _trigrams(s: str) -> frozenset[str]:
    s = f"  {s} "
    return frozenset(s[i:i + 3] for i in range(len(s) - 2))


def sort_by_score(candidates: list[Candidate]) -> list[Candidate]:
    """Tri stable par score décroissant (les scores absents passent en dernier)."""
    return sorted(candidates, key=lambda c: -(c.score if c.score is not None else float("-inf")))
