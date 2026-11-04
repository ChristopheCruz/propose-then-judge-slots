"""Intervalles de confiance par bootstrap et comparaisons appariées, sur les prédictions d'un run.

Protocole (plan de recherche, phase 3) :
- intervalles à 95 % par bootstrap sur les **dialogues**, pas sur les tours : on rééchantillonne
  des grappes (un dialogue Multi3WOZ, ou un énoncé pour les jeux à un tour) ;
- comparaisons appariées par **test de permutation** (échange des deux systèmes grappe par
  grappe), avec correction de **Holm** sur l'ensemble des comparaisons.

Tout se calcule depuis `predictions.jsonl`. Chaque grappe est résumée par des compteurs
additifs (vrais positifs, faux positifs, choix corrects…) ; une métrique est un rapport de
sommes de compteurs, donc un tirage bootstrap n'est qu'une somme pondérée : 1 000 tirages
prennent une fraction de seconde.

Les paires gold sont reconstruites depuis les lignes de prédiction (`gold_values`). Une paire
gold dont le slot n'a pas été interrogé n'y figure pas : l'estimation ponctuelle peut différer
de `metrics.json` de ce fait (en pratique : une paire sur 2 837 sur MultiATIS++ EN).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from ..schema import NONE_VALUE
from .metrics import candidate_values_of, gold_values_of

# Compteurs par grappe (colonnes de la matrice des statistiques).
FIELDS = ("tp", "fp", "fn", "frames", "frames_ok", "present", "correct",
          "unexpressed", "false_pos", "absent", "absent_abstain")
_I = {f: i for i, f in enumerate(FIELDS)}


def _ratio(num, den):
    return np.divide(num, den, out=np.full(np.shape(num), np.nan, dtype=float), where=np.asarray(den) > 0)


def _f1(s):
    p, r = _ratio(s[..., _I["tp"]], s[..., _I["tp"]] + s[..., _I["fp"]]), \
        _ratio(s[..., _I["tp"]], s[..., _I["tp"]] + s[..., _I["fn"]])
    return _ratio(2 * p * r, p + r)


# Métriques : fonction des compteurs sommés (dernier axe = FIELDS). Mêmes définitions que metrics.py.
METRICS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "slot_f1": _f1,
    "slot_precision": lambda s: _ratio(s[..., _I["tp"]], s[..., _I["tp"]] + s[..., _I["fp"]]),
    "slot_recall": lambda s: _ratio(s[..., _I["tp"]], s[..., _I["tp"]] + s[..., _I["fn"]]),
    "frame_accuracy": lambda s: _ratio(s[..., _I["frames_ok"]], s[..., _I["frames"]]),
    "choice_accuracy_given_present": lambda s: _ratio(s[..., _I["correct"]], s[..., _I["present"]]),
    "false_positive_rate": lambda s: _ratio(s[..., _I["false_pos"]], s[..., _I["unexpressed"]]),
    "absent_abstention_rate": lambda s: _ratio(s[..., _I["absent_abstain"]], s[..., _I["absent"]]),
}
DEFAULT_METRICS = ("slot_f1", "slot_precision", "slot_recall", "frame_accuracy",
                   "choice_accuracy_given_present", "false_positive_rate", "absent_abstention_rate")


def cluster_of(example_id: str) -> str:
    """Grappe de rééchantillonnage : le dialogue pour Multi3WOZ (`multi3woz:<lang>:<split>:<dial>:<tour>`),
    l'énoncé lui-même pour les jeux à un tour."""
    if example_id.startswith("multi3woz:"):
        return example_id.rsplit(":", 1)[0]
    return example_id


def cluster_stats(records: Iterable[dict], cluster: Callable[[str], str] = cluster_of
                  ) -> tuple[list[str], np.ndarray]:
    """(identifiants des grappes, matrice grappes × FIELDS) depuis les lignes de prédiction."""
    by_ex: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_ex[r["example_id"]].append(r)
    rows: dict[str, np.ndarray] = {}
    for ex_id, recs in by_ex.items():
        v = rows.setdefault(cluster(ex_id), np.zeros(len(FIELDS)))
        gold, pred = Counter(), Counter()
        for r in recs:
            gv = gold_values_of(r)
            choice = r.get("choice_value")
            gold.update((r["slot"], g) for g in gv)
            if choice not in (None, NONE_VALUE):
                pred[(r["slot"], choice)] += 1
            if gv:
                cands = candidate_values_of(r)
                for g in gv:                      # par paire gold, comme error_decomposition
                    if g in cands:
                        v[_I["present"]] += 1
                        v[_I["correct"]] += choice == g
                    else:
                        v[_I["absent"]] += 1
                        v[_I["absent_abstain"]] += choice in (None, NONE_VALUE)
            else:
                v[_I["unexpressed"]] += 1
                v[_I["false_pos"]] += choice not in (None, NONE_VALUE)
        inter = sum((gold & pred).values())
        v[_I["tp"]] += inter
        v[_I["fp"]] += sum(pred.values()) - inter
        v[_I["fn"]] += sum(gold.values()) - inter
        v[_I["frames"]] += 1
        v[_I["frames_ok"]] += gold == pred
    ids = sorted(rows)
    return ids, np.stack([rows[i] for i in ids]) if ids else np.zeros((0, len(FIELDS)))


def point_estimates(stats: np.ndarray, metrics: Sequence[str] = DEFAULT_METRICS) -> dict[str, float | None]:
    tot = stats.sum(0)
    return {m: _none(METRICS[m](tot)) for m in metrics}


def _none(x) -> float | None:
    x = float(x)
    return None if np.isnan(x) else x


def _weights(n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Poids bootstrap : nombre de tirages de chaque grappe, pour n_boot rééchantillonnages."""
    return rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(float)


def bootstrap_ci(stats: np.ndarray, metrics: Sequence[str] = DEFAULT_METRICS, n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> dict[str, dict]:
    """Intervalle percentile à 1 - alpha pour chaque métrique : {m: {value, lo, hi, n_boot, n_clusters}}."""
    n = len(stats)
    point = point_estimates(stats, metrics)
    if n == 0:
        return {m: {"value": None, "lo": None, "hi": None, "n_boot": 0, "n_clusters": 0} for m in metrics}
    sums = _weights(n, n_boot, np.random.default_rng(seed)) @ stats          # n_boot × FIELDS
    out = {}
    for m in metrics:
        vals = METRICS[m](sums)
        vals = vals[~np.isnan(vals)]
        lo, hi = (np.quantile(vals, [alpha / 2, 1 - alpha / 2]) if len(vals) else (np.nan, np.nan))
        out[m] = {"value": point[m], "lo": _none(lo), "hi": _none(hi), "n_boot": int(n_boot), "n_clusters": n}
    return out


def align(ids_a: list[str], stats_a: np.ndarray, ids_b: list[str], stats_b: np.ndarray
          ) -> tuple[np.ndarray, np.ndarray]:
    """Grappes communes aux deux runs, dans le même ordre (comparaison appariée)."""
    common = sorted(set(ids_a) & set(ids_b))
    if not common:
        raise ValueError("aucune grappe commune : les deux runs ne portent pas sur les mêmes exemples")
    ia, ib = {c: i for i, c in enumerate(ids_a)}, {c: i for i, c in enumerate(ids_b)}
    return stats_a[[ia[c] for c in common]], stats_b[[ib[c] for c in common]]


def paired_test(stats_a: np.ndarray, stats_b: np.ndarray, metrics: Sequence[str] = DEFAULT_METRICS,
                n_perm: int = 10000, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict[str, dict]:
    """Différence A − B, son intervalle bootstrap apparié et la p-valeur bilatérale d'un test de
    permutation (échange de A et B grappe par grappe). Les grappes doivent être alignées (`align`)."""
    if stats_a.shape != stats_b.shape:
        raise ValueError("statistiques non alignées : utiliser align()")
    rng = np.random.default_rng(seed)
    n = len(stats_a)
    swap = rng.integers(0, 2, size=(n_perm, n)).astype(float)                 # 1 = A et B échangés
    perm_a = (1 - swap) @ stats_a + swap @ stats_b
    perm_b = swap @ stats_a + (1 - swap) @ stats_b
    w = _weights(n, n_boot, rng)
    boot_a, boot_b = w @ stats_a, w @ stats_b
    tot_a, tot_b = stats_a.sum(0), stats_b.sum(0)
    out = {}
    for m in metrics:
        f = METRICS[m]
        obs = float(f(tot_a) - f(tot_b))
        if np.isnan(obs):
            out[m] = {"a": None, "b": None, "diff": None, "lo": None, "hi": None, "p": None, "n_clusters": n}
            continue
        null = f(perm_a) - f(perm_b)
        null = null[~np.isnan(null)]
        p = (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (len(null) + 1)
        d = f(boot_a) - f(boot_b)
        d = d[~np.isnan(d)]
        lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
        out[m] = {"a": _none(f(tot_a)), "b": _none(f(tot_b)), "diff": obs, "lo": float(lo), "hi": float(hi),
                  "p": float(p), "n_clusters": n}
    return out


def holm(pvalues: Sequence[float | None]) -> list[float | None]:
    """p-valeurs ajustées par la méthode de Holm (les None restent None)."""
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    order = sorted(idx, key=lambda i: pvalues[i])
    m = len(order)
    adj: list[float | None] = [None] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj
