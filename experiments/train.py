"""Fine-tuning de Laya (C2, C3) à partir d'une config d'entraînement.

    uv run python -m experiments.train --config configs/train/c2.yaml --output checkpoints/c2/multiatis-en-s13
    ... data.dataset=mtop data.lang=fr seed=14 train.epochs=2      # surcharges cle.sous_cle=valeur

Les requêtes sont construites par le même code que l'évaluation (`experiments.run.RequestBuilder`,
`LayaJudge._plan`) sur le split d'entraînement, avec les candidats de la config (oracle).
C3 ajoute `train.augment` : pour chaque exemple, k, taux de valeur absente et type de distracteurs
sont tirés au hasard parmi les listes données (augmentation par options aléatoires, QR5).

Sorties dans --output : le checkpoint (format Laya, meilleure perte dev), config.yaml résolu,
train_metrics.json (effectifs, historique par époque), train.log.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import random
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from experiments.config import config_sha256, instantiate, load_config, run_settings
from experiments.run import RequestBuilder, git_info, load_data, load_train, set_seeds
from laya_slots.train.items import TrainItem, items_from_request

log = logging.getLogger("experiments.train")

TRAIN_DEFAULTS = {
    "unexpressed_ratio": 2.0,   # requêtes à slot non exprimé gardées par requête exprimée (null = toutes)
    "limit_train": None,        # n premiers exemples d'entraînement (essais rapides)
    "dev_split": "dev",
    "dev_limit": 300,           # exemples dev pour suivre la perte (null = tout le split)
    "augment": None,            # C3 : {k: [...], p_absent: [...], distractors: [...]}
    "langs": None,              # langues d'entraînement, ex. "en+fr" (null = data.lang) ; le dev reste data.lang
    "mix": None,                # mélange multi-jeux : liste de {dataset, lang ("*" = toutes), exclude, max_train,
                                # dev_limit, requests} ; remplace data.dataset / train.langs (modèle publié)
    "dev_candidates": "oracle", # candidats du dev de sélection en mode mélange : oracle | reels | both
    "reels_config": "configs/candidats/reels.yaml",
}


def setup_logging(out: Path) -> None:
    """Journal dans la console et dans <out>/train.log (même format que les runs)."""
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in (logging.StreamHandler(sys.stderr), logging.FileHandler(out / "train.log", encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)


def _train_section(cfg: DictConfig) -> dict:
    return {**TRAIN_DEFAULTS, **OmegaConf.to_container(cfg.get("train") or {}, resolve=True)}


def _augmented_builders(cfg: DictConfig, ontology, train_examples, aug: dict) -> list[RequestBuilder]:
    """Un RequestBuilder par combinaison (k, p_absent, distracteurs), oracles ajustés sur le train."""
    from laya_slots.candidates.oracle import OracleGenerator
    builders = []
    for k, p_absent, distractors in itertools.product(aug["k"], aug["p_absent"], aug["distractors"]):
        gen = OracleGenerator(k=int(k), p_absent=float(p_absent), distractors=distractors, seed=cfg.seed)
        gen.fit(train_examples, ontology)
        builders.append(RequestBuilder(cfg, ontology, [gen]))
    return builders


def build_items(cfg: DictConfig, examples, ontology, train_examples, judge, augment: dict | None,
                unexpressed_ratio: float | None, salt: str) -> tuple[list[TrainItem], dict]:
    """Items d'entraînement d'un ensemble d'exemples, et leurs effectifs."""
    if augment:
        builders = _augmented_builders(cfg, ontology, train_examples, augment)
    else:
        generators = [instantiate(g, cfg.seed) for g in cfg.candidates.generators]
        if cfg.candidates.fit:
            for g in generators:
                g.fit(train_examples, ontology)
        builders = [RequestBuilder(cfg, ontology, generators)]
    intents = builders[0].predict_intents(examples)
    items: list[TrainItem] = []
    stats = {"examples": len(examples), "requests_expressed": 0, "requests_unexpressed": 0,
             "requests_unexpressed_kept": 0}
    for ex, intent in zip(examples, intents):
        rng = random.Random(f"{cfg.seed}:{salt}:{ex.id}")
        builder = builders[rng.randrange(len(builders))]
        pending = builder.build(ex, intent)
        pos = [p for p in pending if p.gold_values]
        neg = [p for p in pending if not p.gold_values]
        stats["requests_expressed"] += len(pos)
        stats["requests_unexpressed"] += len(neg)
        if unexpressed_ratio is not None and len(neg) > (n := math.ceil(unexpressed_ratio * max(1, len(pos)))):
            neg = rng.sample(neg, n)
        stats["requests_unexpressed_kept"] += len(neg)
        for p in pos + neg:
            items.extend(items_from_request(judge, p.request, p.gold_values))
    stats["items"] = len(items)
    for kind in sorted({it.kind for it in items}):
        stats[f"items_{kind}"] = sum(it.kind == kind for it in items)
    return items, stats


def train_langs(cfg: DictConfig) -> list[str]:
    """Langues d'entraînement : `train.langs` (« en+fr » ou liste), sinon data.lang."""
    langs = _train_section(cfg)["langs"] or cfg.data.lang
    return [x for x in (langs.split("+") if isinstance(langs, str) else langs) if x]


def _examples_and_ontology(cfg: DictConfig, lang: str, limit: int | None):
    c = OmegaConf.merge(cfg, {"data": {"lang": lang}})
    examples = load_train(c)
    if limit is not None:
        examples = examples[: int(limit)]
    _, ontology, _ = load_data(OmegaConf.merge(c, {"data": {"split": "dev", "limit": 1}}))
    return c, examples, ontology


def mix_entries(cfg: DictConfig, root: str | Path | None = None) -> list[dict]:
    """Entrées de `train.mix` développées : `lang: "*"` = toutes les langues préparées du jeu, moins `exclude`."""
    root = Path(root or cfg.data.root)
    out = []
    for e in _train_section(cfg)["mix"]:
        e = dict(e)
        if e.get("lang") == "*":
            langs = sorted(p.name for p in (root / e["dataset"]).iterdir() if (p / "train.jsonl").exists())
            out += [{**e, "lang": lang} for lang in langs if lang not in set(e.get("exclude") or [])]
        else:
            out.append(e)
    return out


def prepare_mix(cfg: DictConfig) -> tuple[list[TrainItem], list[TrainItem], dict]:
    """Mélange multi-jeux et multilingue (modèle publié).

    Pour chaque (jeu, langue) : au plus `max_train` exemples d'entraînement tirés avec la graine (les
    générateurs restent ajustés sur tout le train), items augmentés comme en C3 ; dev de sélection
    sur `dev_limit` exemples, avec les listes oracle, réelles ou les deux (`train.dev_candidates`)."""
    t = _train_section(cfg)
    judge = instantiate(OmegaConf.to_container(cfg.judge, resolve=True), cfg.seed)
    reels = OmegaConf.load(t["reels_config"]).candidates
    kinds = {"oracle": ["oracle"], "reels": ["reels"], "both": ["oracle", "reels"]}[t["dev_candidates"]]
    train_items, dev_items, by_entry = [], [], {}
    for e in mix_entries(cfg):
        key = f"{e['dataset']}-{e['lang']}"
        c = OmegaConf.merge(cfg, {"data": {"dataset": e["dataset"], "lang": e["lang"], "limit": None}},
                            {"requests": e.get("requests") or {}})
        full = load_train(c)
        examples = full
        if e.get("max_train") and len(full) > int(e["max_train"]):
            examples = random.Random(f"{cfg.seed}:{key}").sample(full, int(e["max_train"]))
        _, ontology, _ = load_data(OmegaConf.merge(c, {"data": {"split": t["dev_split"], "limit": 1}}))
        items, st = build_items(c, examples, ontology, full, judge, t["augment"], t["unexpressed_ratio"], f"train:{key}")
        train_items += items
        by_entry[key] = {"train": st}
        if e.get("dev", True):
            dc = OmegaConf.merge(c, {"data": {"split": t["dev_split"], "limit": e.get("dev_limit", t["dev_limit"])}})
            dev_examples, onto_dev, _ = load_data(dc)
            for kind in kinds:
                cc = c if kind == "oracle" else OmegaConf.merge(c, {"candidates": reels})
                items, dst = build_items(cc, dev_examples, onto_dev, full, judge, None, None, f"dev:{key}:{kind}")
                dev_items += items
                by_entry[key][f"dev_{kind}"] = dst
        log.info("mélange %s : %d items d'entraînement, %d items de dev au total", key, len(train_items), len(dev_items))
    return train_items, dev_items, {"train": {"items": len(train_items), "by_entry": by_entry},
                                    "dev": {"items": len(dev_items)}, "mix": [f"{e['dataset']}-{e['lang']}"
                                                                           for e in mix_entries(cfg)]}


def prepare(cfg: DictConfig) -> tuple[list[TrainItem], list[TrainItem], dict]:
    """Items d'entraînement (une ou plusieurs langues) et de dev (data.lang) ; ni torch ni Laya.

    Plusieurs langues (`train.langs: en+fr`) : chaque langue garde son ontologie (descriptions des
    slots) et ses propres candidats (oracle ajusté sur son train) ; les items sont concaténés."""
    t = _train_section(cfg)
    if t["mix"]:
        return prepare_mix(cfg)
    judge = instantiate(OmegaConf.to_container(cfg.judge, resolve=True), cfg.seed)
    train_items, by_lang = [], {}
    for lang in train_langs(cfg):
        c, examples, ontology = _examples_and_ontology(cfg, lang, t["limit_train"])
        items, by_lang[lang] = build_items(c, examples, ontology, examples, judge, t["augment"],
                                           t["unexpressed_ratio"], f"train:{lang}")
        train_items += items
    train_stats = {k: sum(st[k] for st in by_lang.values()) for k in next(iter(by_lang.values()))
                   if all(k in st for st in by_lang.values())}
    if len(by_lang) > 1:
        train_stats["by_lang"] = by_lang
    # dev : langue de data.lang, candidats de la config sans augmentation, toutes les requêtes
    dev_cfg = OmegaConf.merge(cfg, {"data": {"split": t["dev_split"], "limit": t["dev_limit"]}})
    dev_examples, ontology, data_info = load_data(dev_cfg)
    dev_items, dev_stats = build_items(cfg, dev_examples, ontology, load_train(cfg), judge, None, None, "dev")
    return train_items, dev_items, {"train": train_stats, "dev": dev_stats, "train_langs": list(by_lang),
                                    **data_info}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True, help="dossier du checkpoint")
    ap.add_argument("overrides", nargs="*", help="surcharges OmegaConf cle.sous_cle=valeur")
    args = ap.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out)
    cfg = load_config(args.config, args.overrides)
    (out / "config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    set_seeds(cfg.seed)

    train_items, dev_items, stats = prepare(cfg)
    log.info("effectifs : %s", json.dumps({k: stats[k] for k in ("train", "dev")}, ensure_ascii=False))

    from laya_slots.train.trainer import TrainConfig, load_base_agent, train
    t = _train_section(cfg)
    known = set(TrainConfig.__dataclass_fields__)
    tcfg = TrainConfig(**{"seed": cfg.seed, **{k: v for k, v in t.items() if k in known}})
    params = OmegaConf.to_container(cfg.judge.params, resolve=True)
    tcfg.max_len = tcfg.max_len or params.get("max_len")
    tcfg.head_max_len = tcfg.head_max_len or params.get("head_max_len")
    agent = load_base_agent(params.get("checkpoint", "multilingual"), device=params.get("device"),
                            revision=params.get("revision"))
    meta = {"name": cfg.name, "condition": cfg.condition, "settings": run_settings(cfg),
            "config_sha256": config_sha256(cfg), **git_info()}
    result = train(agent, train_items, dev_items, tcfg, out, meta)
    report = {"name": cfg.name, "condition": cfg.condition, "seed": cfg.seed, "settings": run_settings(cfg),
              "counts": stats, **result, **meta}
    (out / "train_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("terminé : meilleure valeur de sélection dev (%s, à minimiser) %.4f -> %s", tcfg.select_on,
             result["best_dev_loss"], out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
