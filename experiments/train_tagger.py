"""Fine-tuning du tagger BIO de C0 (encodeur multilingue, une étiquette par mot).

    uv run python -m experiments.train_tagger --config configs/train/c0.yaml --output checkpoints/tagger/multiatis-en-s13
    ... data.dataset=mtop data.lang=fr seed=14 train.epochs=3      # surcharges cle.sous_cle=valeur

Étiquettes gold : `laya_slots.candidates.tagger.word_tags` (même découpage en mots qu'à
l'inférence), sur les slots qui ont un span dans l'énoncé ; la perte porte sur le premier sous-mot
de chaque mot. Checkpoint gardé : meilleur F1 des spans (slot, mots exacts) sur le dev.

Sorties dans --output : le modèle et le tokenizer (format Hugging Face, lus par TaggerGenerator),
config.yaml résolu, train_metrics.json (effectifs, historique par époque), train.log.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from experiments.config import config_sha256, load_config, run_settings
from experiments.run import git_info, load_data, set_seeds
from experiments.train import _examples_and_ontology, setup_logging, train_langs
from laya_slots.candidates.tagger import OUTSIDE, decode, word_tags

log = logging.getLogger("experiments.train_tagger")

TAGGER_DEFAULTS = {
    "model": "microsoft/mdeberta-v3-base",
    "max_len": 256,
    "epochs": 10,
    "batch_size": 32,
    "lr": 3.0e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "bf16": True,
    "limit_train": None,
    "dev_split": "dev",
    "dev_limit": None,
    "langs": None,
    "log_every": 100,
}


def tagger_section(cfg: DictConfig) -> dict:
    return {**TAGGER_DEFAULTS, **OmegaConf.to_container(cfg.get("train") or {}, resolve=True)}


def tag_examples(examples) -> tuple[list[tuple[list[str], list[str]]], dict]:
    """(mots, étiquettes) par exemple, et effectifs agrégés de word_tags."""
    seqs, stats = [], {"examples": len(examples), "spans": 0, "conflicts": 0, "misaligned": 0, "empty": 0}
    for ex in examples:
        words, tags, st = word_tags(ex.utterance, ex.slots)
        for k, v in st.items():
            stats[k] += v
        seqs.append(([w for w, _, _ in words], tags))
    return seqs, stats


def label_list(seqs) -> list[str]:
    slots = sorted({t[2:] for _, tags in seqs for t in tags if t != OUTSIDE})
    return [OUTSIDE] + [f"{p}-{s}" for s in slots for p in ("B", "I")]


def align_labels(word_ids: list[int | None], word_labels: list[int], ignore: int = -100) -> list[int]:
    """Étiquette du mot sur son premier sous-mot, `ignore` ailleurs (sous-mots suivants, spéciaux)."""
    out, prev = [], None
    for w in word_ids:
        out.append(word_labels[w] if w is not None and w != prev else ignore)
        prev = w
    return out


def span_f1(gold: list[list[str]], pred: list[list[str]]) -> dict:
    """F1 micro des spans (slot, premier mot, dernier mot) : exacts seulement."""
    tp = n_gold = n_pred = 0
    for g, p in zip(gold, pred):
        gs = {s[:3] for s in decode(g)}
        ps = {s[:3] for s in decode(p)}
        tp, n_gold, n_pred = tp + len(gs & ps), n_gold + len(gs), n_pred + len(ps)
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_gold if n_gold else 0.0
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "n_gold": n_gold, "n_pred": n_pred}


def encode(tokenizer, seqs, label2id: dict[str, int], max_len: int) -> list[dict]:
    feats = []
    for words, tags in seqs:
        enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=max_len)
        ids = [label2id.get(t, label2id[OUTSIDE]) for t in tags]
        feats.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                      "labels": align_labels(enc.word_ids(), ids), "n_words": len(words)})
    return feats


def collate(batch: list[dict], pad_id: int):
    import torch
    n = max(len(f["input_ids"]) for f in batch)

    def pad(xs, v):
        return [x + [v] * (n - len(x)) for x in xs]
    return {"input_ids": torch.tensor(pad([f["input_ids"] for f in batch], pad_id)),
            "attention_mask": torch.tensor(pad([f["attention_mask"] for f in batch], 0)),
            "labels": torch.tensor(pad([f["labels"] for f in batch], -100))}


def predict_tags(model, feats, id2label, batch_size: int, device, pad_id: int, bf16: bool) -> list[list[str]]:
    """Étiquette prédite par mot (lue sur le premier sous-mot, comme TaggerModel)."""
    import torch
    out = []
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(feats), batch_size):
            chunk = feats[i:i + batch_size]
            b = collate(chunk, pad_id)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
                logits = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device)).logits
            lab = logits.argmax(-1).cpu()
            for f, row, lr in zip(chunk, lab, b["labels"]):
                tags = [id2label[int(t)] for t, g in zip(row, lr) if g != -100]
                out.append(tags + [OUTSIDE] * (f["n_words"] - len(tags)))   # mots tronqués
    model.train()
    return out


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
    t = tagger_section(cfg)

    train_seqs, train_stats = [], {}
    for lang in train_langs(cfg):
        _, examples, _ = _examples_and_ontology(cfg, lang, t["limit_train"])
        seqs, train_stats[lang] = tag_examples(examples)
        train_seqs += seqs
    dev_cfg = OmegaConf.merge(cfg, {"data": {"split": t["dev_split"], "limit": t["dev_limit"]}})
    dev_examples, _, data_info = load_data(dev_cfg)
    dev_seqs, dev_stats = tag_examples(dev_examples)
    labels = label_list(train_seqs)
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = dict(enumerate(labels))
    log.info("effectifs : %s ; %d étiquettes", json.dumps({"train": train_stats, "dev": dev_stats},
                                                         ensure_ascii=False), len(labels))

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(t["model"])
    # dtype explicite : transformers 5 garde sinon le type du fichier (fp16 pour mdeberta-v3-base),
    # et AdamW sur des poids fp16 diverge (nan) ; le bf16 passe par autocast.
    model = AutoModelForTokenClassification.from_pretrained(
        t["model"], num_labels=len(labels), id2label=id2label, label2id=label2id, dtype=torch.float32).to(device)
    pad_id = tokenizer.pad_token_id or 0
    train_feats = encode(tokenizer, train_seqs, label2id, t["max_len"])
    dev_feats = encode(tokenizer, dev_seqs, label2id, t["max_len"])
    dev_gold = [tags for _, tags in dev_seqs]

    decay = [p for n, p in model.named_parameters() if not any(k in n for k in ("bias", "LayerNorm", "layernorm"))]
    no_decay = [p for n, p in model.named_parameters() if any(k in n for k in ("bias", "LayerNorm", "layernorm"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": t["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=t["lr"])
    steps_per_epoch = math.ceil(len(train_feats) / t["batch_size"])
    total = steps_per_epoch * t["epochs"]
    warmup = int(t["warmup_ratio"] * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warmup) if s < warmup else max(0.0, (total - s) / max(1, total - warmup)))
    bf16 = bool(t["bf16"]) and device.type == "cuda"
    rng = random.Random(cfg.seed)

    history, best, step, t0 = [], -1.0, 0, time.perf_counter()
    model.train()
    for epoch in range(1, t["epochs"] + 1):
        order = list(range(len(train_feats)))
        rng.shuffle(order)
        for i in range(0, len(order), t["batch_size"]):
            b = {k: v.to(device) for k, v in collate([train_feats[j] for j in order[i:i + t["batch_size"]]], pad_id).items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["max_grad_norm"])
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % t["log_every"] == 0:
                log.info("époque %d, pas %d/%d : perte %.4f, lr %.2e", epoch, step, total, loss.item(),
                         sched.get_last_lr()[0])
        dev = span_f1(dev_gold, predict_tags(model, dev_feats, id2label, 2 * t["batch_size"], device, pad_id, bf16))
        history.append({"epoch": epoch, **dev})
        log.info("époque %d : dev F1 des spans %.4f (P %.4f, R %.4f)", epoch, dev["f1"], dev["precision"], dev["recall"])
        if dev["f1"] > best:
            best = dev["f1"]
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)
            log.info("checkpoint enregistré (meilleur F1 dev %.4f) : %s", best, out)

    meta = {"name": cfg.name, "condition": cfg.condition, "settings": run_settings(cfg),
            "config_sha256": config_sha256(cfg), **git_info()}
    report = {**meta, "seed": cfg.seed, "train": t, "labels": labels, "data": data_info,
              "counts": {"train": train_stats, "dev": dev_stats}, "history": history,
              "best_dev_f1": best, "train_time_s": time.perf_counter() - t0}
    (out / "train_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("terminé : meilleur F1 dev des spans %.4f -> %s", best, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
