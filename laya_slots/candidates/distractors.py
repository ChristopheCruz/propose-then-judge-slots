"""Échantillonneur de distracteurs, partagé par tous les générateurs.

Un distracteur est une valeur plausible mais fausse. Chaque *stratégie* fournit un réservoir
(pool) de valeurs normalisées :

- "train_values" : valeurs du même slot vues au train (poids = fréquence) ;
- "ontology"     : valeurs fermées du slot dans l'ontologie ;
- "utterance"    : n-grammes pleins de l'énoncé courant (autres spans) ;
- "other_slots"  : valeurs d'autres slots du même exemple, fournies par l'appelant
                   (confusions de type départ/arrivée) ;
- "history"      : valeurs de `state_before` et valeurs connues citées dans l'historique ;
- "random"       : n'importe quelle valeur connue, tous slots confondus (distracteurs « faciles »).

Tirage : à chaque pas, une stratégie non épuisée est tirée selon son poids, puis une valeur
dans son réservoir selon `selection` :
- "uniform"   : uniforme ;
- "frequency" : proportionnelle à la fréquence au train (uniforme quand elle est inconnue) ;
- "similar"   : proportionnelle à exp(sim / temperature), où sim est la similarité lexicale
                (trigrammes de caractères) avec les valeurs de référence (en général les gold) ;
                c'est le réglage des distracteurs « difficiles ». Sans référence, retombe sur "frequency".

Les valeurs sont dédupliquées après normalisation et les valeurs exclues (gold, NONE_VALUE)
ne sont jamais tirées. Avec `exclude_overlapping`, les variantes de frontière des valeurs de
référence (« on april » ou « apr » pour « april » : l'une contient l'autre, en mots entiers) ne
sont pas tirées non plus ; préréglage "hard_no_overlap". Le tirage est séquentiel et ne dépend pas du nombre demandé : avec le même
générateur aléatoire, les n premiers distracteurs sont un préfixe des n+1 premiers (listes emboîtées
quand on fait varier k).

L'échantillonneur ne lit jamais `example.slots` : les valeurs des autres slots lui sont passées
explicitement (`other_values`).
"""
from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from ..schema import NONE_VALUE, Candidate, Example, Ontology, SlotSpec
from .base import NormalizeFn, Normalizer, as_normalizer, char_similarity, is_content_span, ngrams

STRATEGIES = ("train_values", "ontology", "utterance", "other_slots", "history", "random")
Selection = Literal["uniform", "frequency", "similar"]
_STATIC = frozenset({"train_values", "ontology", "random"})   # réservoirs indépendants de l'exemple


@dataclass
class DistractorConfig:
    """strategies : stratégie -> poids relatif (poids nul ou absent = stratégie désactivée)."""
    strategies: dict[str, float] = field(default_factory=lambda: {"train_values": 1.0, "ontology": 1.0})
    selection: Selection = "frequency"
    temperature: float = 0.1          # pour selection="similar"
    max_ngram: int = 3                # longueur max des n-grammes ("utterance", "history")
    history_turns: int | None = None  # nombre de tours d'historique lus (None = tous)
    exclude_overlapping: bool = False  # écarte les valeurs qui contiennent une référence ou y sont contenues

    def __post_init__(self):
        unknown = set(self.strategies) - set(STRATEGIES)
        if unknown:
            raise ValueError(f"stratégies inconnues : {sorted(unknown)} (connues : {STRATEGIES})")
        if self.selection not in ("uniform", "frequency", "similar"):
            raise ValueError(f"selection inconnue : {self.selection}")


# Préréglages correspondant aux axes du plan (distracteurs faciles / difficiles).
PRESETS: dict[str, DistractorConfig] = {
    "easy": DistractorConfig(strategies={"random": 1.0}, selection="uniform"),
    "same_slot": DistractorConfig(strategies={"train_values": 1.0, "ontology": 1.0}, selection="frequency"),
    "hard": DistractorConfig(
        strategies={"train_values": 1.0, "ontology": 1.0, "other_slots": 1.0, "utterance": 1.0,
                    "history": 1.0},
        selection="similar"),
    "hard_no_overlap": DistractorConfig(
        strategies={"train_values": 1.0, "ontology": 1.0, "other_slots": 1.0, "utterance": 1.0,
                    "history": 1.0},
        selection="similar", exclude_overlapping=True),
    # Confusion entre slots : surtout des valeurs d'autres slots du même énoncé (le piège principal des
    # candidats réels : « boston » proposé pour l'escale quand c'est le départ), sans variantes de frontière.
    "confusion": DistractorConfig(
        strategies={"other_slots": 3.0, "train_values": 1.0, "ontology": 1.0, "utterance": 0.5,
                    "history": 1.0},
        selection="similar", exclude_overlapping=True),
}


def overlaps(a: str, b: str) -> bool:
    """Vrai si `a` et `b` diffèrent et que l'une est une suite de mots entiers de l'autre."""
    ta, tb = a.split(), b.split()
    if not ta or not tb or ta == tb:
        return False
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    return any(long_[i:i + len(short)] == short for i in range(len(long_) - len(short) + 1))


def make_config(config: DistractorConfig | dict | str | None) -> DistractorConfig:
    if config is None:
        return DistractorConfig()
    if isinstance(config, DistractorConfig):
        return config
    if isinstance(config, str):
        return PRESETS[config]
    config = dict(config)
    base = PRESETS[config.pop("preset")] if "preset" in config else DistractorConfig()
    merged = {**base.__dict__, **config}
    return DistractorConfig(**merged)


@dataclass
class _Item:
    value: str
    surface: str | None = None
    span: tuple[int, int] | None = None


class _Pool:
    """Réservoir pondéré avec tirage sans remise par rejet (les valeurs prises sont bannies)."""
    MAX_TRIES = 32

    def __init__(self, items: list[_Item], weights: list[float] | None):
        self.items = items
        self.weights = weights

    def draw(self, rng: random.Random, banned: set[str]) -> _Item | None:
        for _ in range(self.MAX_TRIES):
            if not self.items:
                return None
            if self.weights is None:
                it = self.items[rng.randrange(len(self.items))]
            else:
                it = rng.choices(self.items, weights=self.weights)[0]
            if it.value not in banned:
                return it
        # beaucoup de rejets : on retire les valeurs bannies et on recommence
        keep = [i for i, it in enumerate(self.items) if it.value not in banned]
        self.items = [self.items[i] for i in keep]
        if self.weights is not None:
            self.weights = [self.weights[i] for i in keep]
            if not any(self.weights):
                self.weights = None
        return self.draw(rng, banned) if self.items else None


class DistractorSampler:
    def __init__(self, config: DistractorConfig | dict | str | None = None,
                 normalize: NormalizeFn | Normalizer | None = None):
        self.config = make_config(config)
        self.normalize = as_normalizer(normalize)
        self.train_counts: dict[str, Counter] = {}
        self.global_counts: Counter = Counter()
        self._sorted: dict[str, list[str]] = {}
        self._global_sorted: list[str] = []
        self._cache: dict[tuple, tuple[list[_Item], list[float] | None]] = {}
        self.fitted = False

    # ------------------------------------------------------------------
    def fit(self, train: Iterable[Example], ontology: Ontology | None = None) -> DistractorSampler:
        """Indexe les valeurs gold du TRAIN (pas de l'exemple courant) et de l'ontologie."""
        counts: dict[str, Counter] = {}
        for ex in train:
            for sv in ex.slots:
                v = self.normalize(sv.value, slot=sv.name, lang=ex.lang)
                if v and v != NONE_VALUE:
                    counts.setdefault(sv.name, Counter())[v] += 1
        self.train_counts = counts
        g: Counter = Counter()
        for c in counts.values():
            g.update(c)
        if ontology is not None:
            for spec in ontology.slots.values():
                for v in spec.values or []:
                    nv = self.normalize(v, slot=spec.name, lang=ontology.lang)
                    if nv and nv != NONE_VALUE and nv not in g:
                        g[nv] = 1
        self.global_counts = g
        self._sorted = {s: sorted(c) for s, c in counts.items()}
        self._global_sorted = sorted(g)
        self._cache = {}
        self.fitted = True
        return self

    # ------------------------------------------------------------------
    def _norm(self, v: str, slot: SlotSpec, example: Example) -> str:
        return self.normalize(v, slot=slot.name, lang=example.lang)

    def _items(self, strategy: str, example: Example, slot: SlotSpec,
               other_values: Iterable[str]) -> list[_Item]:
        cfg = self.config
        if strategy == "train_values":
            return [_Item(v) for v in self._sorted.get(slot.name, [])]
        if strategy == "ontology":
            return [_Item(self._norm(v, slot, example)) for v in sorted(slot.values or [])]
        if strategy == "utterance":
            return [_Item(self._norm(sp.text, slot, example), sp.text, (sp.start, sp.end))
                    for sp in ngrams(example.utterance, cfg.max_ngram)
                    if is_content_span(sp.text, self.normalize)]
        if strategy == "other_slots":
            return [_Item(self._norm(v, slot, example)) for v in other_values]
        if strategy == "history":
            items = [_Item(self._norm(v, slot, example)) for _, v in sorted(example.state_before.items())]
            turns = example.history if cfg.history_turns is None else example.history[-cfg.history_turns:]
            known = self.global_counts
            for t in turns:
                for sp in ngrams(t.text, cfg.max_ngram):
                    v = self._norm(sp.text, slot, example)
                    if v in known and is_content_span(sp.text, self.normalize):
                        items.append(_Item(v, sp.text))
            return items
        if strategy == "random":
            return [_Item(v) for v in self._global_sorted]
        raise ValueError(strategy)

    def _weights(self, strategy: str, items: list[_Item], slot: SlotSpec,
                 reference: list[str]) -> list[float] | None:
        sel = self.config.selection
        if sel == "similar" and reference:
            t = self.config.temperature
            return [math.exp(max(char_similarity(it.value, r) for r in reference) / t) for it in items]
        if sel in ("frequency", "similar"):
            counts = self.global_counts if strategy == "random" else self.train_counts.get(slot.name)
            if counts:
                w = [float(counts.get(it.value, 0)) for it in items]
                # les valeurs jamais vues gardent un poids minimal
                floor = min((x for x in w if x > 0), default=1.0)
                return [x if x > 0 else floor for x in w]
        return None

    def pools(self, example: Example, slot: SlotSpec, other_values: Iterable[str] = (),
              reference: Iterable[str] = ()) -> dict[str, _Pool]:
        reference = list(reference)
        other_values = list(other_values)
        similar = self.config.selection == "similar" and bool(reference)
        out = {}
        for strategy, w in self.config.strategies.items():
            if w <= 0:
                continue
            # les réservoirs qui ne dépendent pas de l'exemple sont mis en cache (le tirage ne
            # modifie jamais les listes partagées : _Pool les remplace au lieu de les muter)
            static = strategy in _STATIC
            key = (strategy, "" if strategy == "random" else slot.name, example.lang)
            if static and key in self._cache:
                items, weights = self._cache[key]
            else:
                items, seen = [], set()
                for it in self._items(strategy, example, slot, other_values):
                    if it.value and it.value != NONE_VALUE and it.value not in seen:
                        seen.add(it.value)
                        items.append(it)
                weights = None if similar else self._weights(strategy, items, slot, reference)
                if static:
                    self._cache[key] = (items, weights)
            if similar:
                weights = self._weights(strategy, items, slot, reference)
            if items:
                out[strategy] = _Pool(items, weights)
        return out

    def sample(self, n: int, example: Example, slot: SlotSpec, rng: random.Random,
               exclude: Iterable[str] = (), reference: Iterable[str] = (),
               other_values: Iterable[str] = ()) -> list[Candidate]:
        """Tire jusqu'à `n` distracteurs distincts (moins si les réservoirs sont épuisés).

        exclude : valeurs normalisées interdites (les gold, les candidats déjà proposés).
        reference : valeurs pour la similarité (selection="similar"), en général les gold.
        other_values : valeurs d'autres slots du même exemple (stratégie "other_slots").
        Source des candidats : "distractor:<stratégie>".
        """
        reference = list(reference)
        needs_fit = {s for s, w in self.config.strategies.items() if w > 0} & {"train_values", "random"}
        if needs_fit and not self.fitted:
            # sans fit, ces réservoirs sont vides et les listes silencieusement trop courtes
            raise RuntimeError(f"distracteurs {sorted(needs_fit)} : appeler fit(train, ontology) d'abord "
                               "(candidates.fit: true dans la config)")
        pools = self.pools(example, slot, other_values, reference)
        banned = {NONE_VALUE, *exclude}
        strict = self.config.exclude_overlapping
        out: list[Candidate] = []
        names = list(pools)
        while len(out) < n and names:
            weights = [self.config.strategies[s] for s in names]
            s = rng.choices(names, weights=weights)[0]
            it = pools[s].draw(rng, banned)
            if it is None:
                names.remove(s)
                continue
            banned.add(it.value)
            if strict and any(overlaps(it.value, r) for r in reference):
                continue
            out.append(Candidate(it.value, f"distractor:{s}", None, it.surface, it.span))
        return out
