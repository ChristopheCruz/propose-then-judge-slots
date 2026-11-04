"""Chargement des configurations d'expérience (OmegaConf).

Une config est un YAML, complété par les valeurs par défaut ci-dessous puis par les
surcharges de la ligne de commande (`cle.sous_cle=valeur`). La clé `inherit` (chemin ou liste
de chemins, relatifs au fichier) fusionne d'abord les configs parentes : les dicts sont fusionnés
récursivement, les listes remplacées.

Les composants sont désignés par chemin d'import :
    judge: {class: laya_slots.judges.laya.LayaJudge, params: {...}}
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import re
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

# Valeurs par défaut ; "???" = obligatoire.
DEFAULTS: dict[str, Any] = {
    "name": None,                 # nom lisible du run (tableaux)
    "condition": None,            # C0..C5, pilote...
    "seed": 13,
    "data": {
        "dataset": None,          # multiatis | mtop | multi3woz
        "lang": "en",
        "split": "test",
        "root": "data/processed",  # <root>/<dataset>/<lang>/<split>.jsonl + ontology.json
        "path": None,             # chemin explicite du JSONL (prioritaire sur root)
        "ontology": None,         # chemin explicite de ontology.json
        "train_split": "train",   # lu seulement si candidates.fit
        "train_path": None,
        "loader": None,           # {class, params} : DatasetLoader à la place des fichiers convertis
        "limit": None,            # n premiers exemples (essais rapides)
    },
    "requests": {
        "slots": "intent",        # intent : slots de l'intention (gold ou prédite) ; ontology : tous
        "intent": "gold",         # intention passée au juge : gold | predicted | none | domain (domaine gold du tour)
        "filled": "none",         # contexte « déjà rempli » : none | state_before
        "strip_gold": True,       # le juge reçoit l'exemple sans ses slots gold
    },
    "candidates": {
        "generators": [],         # liste de {class, params} ; union dédoublonnée par valeur
        "fit": False,             # appeler generator.fit(train, ontology)
        "shuffle": False,         # mélanger l'ordre des candidats (graine par requête)
        "max_candidates": None,
    },
    "judge": {
        "class": "???",
        "params": {},
        "batch_size": 32,         # requêtes par appel à judge_batch
    },
    "intent_predictor": None,     # {class, params} : objet avec predict_batch(examples) -> intents
    "output": {
        "save_probs": True,       # écrire probs et raw dans predictions.jsonl
        "log_every": 500,         # requêtes entre deux lignes de progression
    },
}


def _load_with_inherit(path: Path, stack: tuple[Path, ...] = ()) -> DictConfig:
    path = path.resolve()
    if path in stack:
        raise ValueError(f"héritage circulaire : {' -> '.join(map(str, stack + (path,)))}")
    cfg = OmegaConf.load(path)
    parents = cfg.pop("inherit", None)
    if parents is None:
        return cfg
    if isinstance(parents, str):
        parents = [parents]
    bases = [_load_with_inherit(path.parent / p, stack + (path,)) for p in parents]
    return OmegaConf.merge(*bases, cfg)


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Config résolue (interpolations calculées), valeurs obligatoires vérifiées."""
    cfg = OmegaConf.merge(OmegaConf.create(DEFAULTS), _load_with_inherit(Path(path)))
    for o in overrides or []:
        apply_override(cfg, o)
    container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    cfg = OmegaConf.create(container)
    d = cfg.data
    if not (d.path or d.loader or d.dataset):
        raise ValueError("data : préciser dataset (+ lang, split), path ou loader")
    if cfg.requests.slots not in ("intent", "ontology"):
        raise ValueError(f"requests.slots inconnu : {cfg.requests.slots}")
    if cfg.requests.intent not in ("gold", "predicted", "none", "domain"):
        raise ValueError(f"requests.intent inconnu : {cfg.requests.intent}")
    if cfg.requests.intent == "predicted" and not cfg.intent_predictor:
        raise ValueError("requests.intent=predicted demande un intent_predictor")
    if cfg.requests.filled not in ("none", "state_before"):
        raise ValueError(f"requests.filled inconnu : {cfg.requests.filled}")
    return cfg


def apply_override(cfg: DictConfig, override: str) -> None:
    """Applique `a.b.0.c=valeur` (valeur lue en YAML ; les indices de liste sont permis)."""
    key, sep, raw = override.partition("=")
    if not sep or not key:
        raise ValueError(f"surcharge invalide : {override!r} (attendu cle.sous_cle=valeur)")
    OmegaConf.update(cfg, key.strip(), yaml.safe_load(raw) if raw.strip() else None, merge=True,
                     force_add=True)


# Paramètres qui distinguent deux runs d'une même condition (clés de regroupement des tableaux).
CANDIDATE_SETTINGS = ("k", "p_absent", "distractors")
JUDGE_SETTINGS = ("mode", "checkpoint", "model", "rotations", "expressed_qtype")


def run_settings(cfg: DictConfig | dict) -> dict:
    """Réglages lisibles d'un run : limite d'exemples, générateurs, k / p_absent / distracteurs
    (premier générateur qui les fixe), réglages du juge. Écrit sous `settings` dans metrics.json."""
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    gens = [{"class": g} if isinstance(g, str) else g for g in cfg.get("candidates", {}).get("generators") or []]
    out: dict[str, Any] = {
        "limit": (cfg.get("data") or {}).get("limit"),
        "generators": "+".join(g["class"].rpartition(".")[2] for g in gens),
        "fit": bool((cfg.get("candidates") or {}).get("fit")),
    }
    for g in gens:
        for k in CANDIDATE_SETTINGS:
            if k in (g.get("params") or {}):
                out.setdefault(k, g["params"][k])
    judge = cfg.get("judge") or {}
    out["judge"] = str(judge.get("class", "")).rpartition(".")[2]
    if judge.get("batch_size") == 1:
        # une requête à la fois : mesure de latence (H1), à ne pas mélanger aux runs par lots
        out["mesure"] = "latence"
    if judge.get("finetuned"):
        # données d'entraînement du checkpoint, sans la graine : « mtop-en », « mtop-en+fr », « multiatis-en-hard »
        out["train_data"] = re.sub(r"-s\d+(?=-|$)", "", Path(str(judge["finetuned"])).name)
    for k in JUDGE_SETTINGS:
        if k in (judge.get("params") or {}):
            out[f"judge_{k}"] = judge["params"][k]
    return out


def config_sha256(cfg: DictConfig) -> str:
    return hashlib.sha256(OmegaConf.to_yaml(cfg).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Instanciation par chemin d'import
# ---------------------------------------------------------------------------
def import_class(path: str):
    module, _, name = path.rpartition(".")
    if not module:
        raise ValueError(f"chemin de classe invalide : {path!r} (attendu module.Classe)")
    return getattr(importlib.import_module(module), name)


def instantiate(spec: Any, seed: int | None = None):
    """Instancie {class: a.b.C, params: {...}} (ou la chaîne "a.b.C").

    Si le constructeur accepte un argument `seed` et que params ne le fixe pas, la graine
    de la config est injectée : toute source d'aléa reçoit une graine explicite.
    """
    if isinstance(spec, DictConfig):
        spec = OmegaConf.to_container(spec, resolve=True)
    if isinstance(spec, str):
        spec = {"class": spec}
    cls = import_class(spec["class"])
    params = dict(spec.get("params") or {})
    if seed is not None and "seed" not in params:
        try:
            if "seed" in inspect.signature(cls).parameters:
                params["seed"] = seed
        except (TypeError, ValueError):
            pass
    return cls(**params)
