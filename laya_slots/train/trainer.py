"""Fine-tuning supervisé de Laya sur des `TrainItem`.

Objectif : log score (entropie croisée) sur les options de chaque question, une règle de score
strictement propre, comme celles que Laya optimise à l'origine (RLCD). Le paquet laya 0.3.22 ne
publie pas son module d'entraînement : on réutilise ses primitives d'inférence (`Agent`), ce qui
garantit que l'encodage des questions est celui de la prédiction.

- encodage : `Agent._to_internal` + `Agent._encode_state` (mêmes séquences, mêmes marqueurs) ;
- modèle : `agent.model` (`laya.common.DecisionModel`), entraîné en fp32 avec autocast bf16 ;
- sauvegarde : dossier au format Laya (rl_agent_config.json, model.safetensors, tokenizer/,
  encoder/), lisible par `laya.Agent` et par `LayaJudge(model_path=...)`.

Les logits sont entraînés sans température : le checkpoint écrit des températures neutres
(T = 1) ; la calibration se fait ensuite sur le dev (laya_slots.decision).

Dépendances : extra `laya` (torch, transformers, safetensors).
"""
from __future__ import annotations

import json
import logging
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .items import TrainItem

log = logging.getLogger("laya_slots.train")

# Checkpoints publiés : nom LayaJudge -> (dépôt, sous-dossier), comme laya.router.DEFAULT_MODELS.
BASE_CHECKPOINTS = {"english": ("convaiinnovations/laya", None),
                    "multilingual": ("convaiinnovations/laya", "multilingual")}


@dataclass
class TrainConfig:
    epochs: int = 3
    micro_batch: int = 32
    grad_accum: int = 1
    lr_encoder: float = 2e-5
    lr_head: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    grad_checkpointing: bool = True
    bf16: bool = True
    seed: int = 13
    log_every: int = 100
    # Critère de choix du checkpoint sur le dev (on minimise selection_loss) :
    # "f1" = 1 − F1 des slots sur les questions du dev (mode single ; métrique principale, insensible à
    # la surconfiance qui fausse les pertes) ; "balanced" = moyenne des pertes des questions à slot
    # exprimé et non exprimé ; "all" = perte moyenne de toutes les questions.
    select_on: str = "f1"
    max_len: int | None = None          # None = valeur du checkpoint
    head_max_len: int | None = None


def load_base_agent(checkpoint: str = "multilingual", device: str | None = None, revision: str | None = None):
    from laya import Agent
    if checkpoint in BASE_CHECKPOINTS:
        repo, sub = BASE_CHECKPOINTS[checkpoint]
        return Agent(repo, device=device, subfolder=sub, revision=revision)
    return Agent(checkpoint, device=device)          # dossier local (reprise d'un fine-tuning)


def encode_item(agent, item: TrainItem, max_len: int | None = None, head_max_len: int | None = None) -> dict:
    """Séquence, marqueurs et cible one-hot d'un item, encodés comme à la prédiction."""
    qid = "q"
    agent._check_question(qid, item.question)
    internal = {qid: agent._to_internal(item.question)}
    enc = agent._encode_state(item.state, [qid], internal, max_len=max_len, head_max_len=head_max_len)[0]
    k = len(enc["markers"])
    if not 0 <= item.target < k:
        raise ValueError(f"cible {item.target} hors des {k} options ({item.meta})")
    target = [0.0] * k
    target[item.target] = 1.0
    return {"ids": enc["ids"], "markers": enc["markers"], "qtype": enc["qtype"], "target": target,
            "label": item.target, "kind": item.kind, "expressed": item.expressed}


def _batches(encoded: list[dict], size: int, rng: random.Random | None) -> list[list[dict]]:
    """Lots de taille `size`, regroupés par longueur pour limiter le bourrage (ordre des lots mélangé)."""
    idx = list(range(len(encoded)))
    if rng is not None:
        rng.shuffle(idx)
    # tri par longueur dans des fenêtres de 50 lots : peu de bourrage, mélange conservé
    window = size * 50
    out = []
    for w in range(0, len(idx), window):
        chunk = sorted(idx[w:w + window], key=lambda i: len(encoded[i]["ids"]))
        out += [[encoded[i] for i in chunk[b:b + size]] for b in range(0, len(chunk), size)]
    if rng is not None:
        rng.shuffle(out)
    return out


def _forward_loss(model, batch: list[dict], pad_id: int, device, bf16: bool):
    import torch
    from laya.common import collate_items
    b = collate_items([batch], pad_id)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
        logits, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                          b["marker_pos"].to(device), b["marker_mask"].to(device), b["qtype"].to(device))
    logp = torch.log_softmax(logits.float(), -1)
    target = b["target"].to(device)
    loss = -(target * logp).sum(-1)                  # log score par question
    pred = logits.argmax(-1).cpu()
    return loss, pred, b["label"]


def evaluate(agent, encoded: list[dict], batch_size: int, bf16: bool) -> dict:
    """Perte moyenne et exactitude : globales, par type de question, et séparément pour les slots
    exprimés et non exprimés (ces derniers dominent : « non précisé » partout ferait déjà ~0,9)."""
    import torch
    model, device = agent.model, agent.device
    model.eval()
    tot = defaultdict(lambda: [0.0, 0, 0])            # kind -> [somme pertes, bonnes réponses, n]
    tp = fp = fn = 0                                  # F1 des slots, questions "choice" (dernière option = « non précisé »)
    with torch.no_grad():
        for batch in _batches(encoded, batch_size, None):
            loss, pred, label = _forward_loss(model, batch, agent.tok.pad_token_id, device, bf16)
            for it, lo, p, y in zip(batch, loss.tolist(), pred.tolist(), label.tolist()):
                for key in ("all", it["kind"], "expressed" if it["expressed"] else "unexpressed"):
                    t = tot[key]
                    t[0] += lo
                    t[1] += int(p == y)
                    t[2] += 1
                if it["kind"] == "choice":
                    none = len(it["markers"]) - 1
                    tp += int(p == y != none)
                    fp += int(p != none and p != y)
                    fn += int(y != none and p != y)
    out = {k: {"loss": s / n, "accuracy": c / n, "n": n} for k, (s, c, n) in tot.items() if n}
    if tp + fp + fn:
        out["slot_f1"] = {"f1": 2 * tp / (2 * tp + fp + fn), "tp": tp, "fp": fp, "fn": fn}
    return out


def save_checkpoint(agent, out_dir: str | Path, meta: dict) -> Path:
    from safetensors.torch import save_file
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    agent.tok.save_pretrained(str(out / "tokenizer"))
    agent.model.encoder.config.save_pretrained(str(out / "encoder"))
    # copies indépendantes : safetensors refuse les tenseurs qui partagent leur stockage
    state = {k: v.detach().to("cpu").clone().contiguous() for k, v in agent.model.state_dict().items()}
    save_file(state, str(out / "model.safetensors"))
    cfg = dict(agent.cfg)
    cfg.pop("temperature_by_options", None)
    cfg.update(temperature=[1.0, 1.0, 1.0], fine_tuned=True, laya_slots=meta)
    (out / "rl_agent_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def train(agent, train_items: list[TrainItem], dev_items: list[TrainItem], cfg: TrainConfig,
          out_dir: str | Path, meta: dict | None = None) -> dict:
    """Entraîne `agent.model` en place ; garde dans `out_dir` le checkpoint de meilleure perte dev.

    Rend l'historique : une entrée par époque (perte d'entraînement, métriques dev, durée)."""
    import torch

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    device = agent.device
    model = agent.model.float()
    t0 = time.perf_counter()
    enc_train = [encode_item(agent, it, cfg.max_len, cfg.head_max_len) for it in train_items]
    enc_dev = [encode_item(agent, it, cfg.max_len, cfg.head_max_len) for it in dev_items]
    log.info("encodage : %d items d'entraînement, %d de dev (%.0f s)",
             len(enc_train), len(enc_dev), time.perf_counter() - t0)

    if cfg.grad_checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": cfg.lr_encoder},
                             {"params": head_params, "lr": cfg.lr_head}], weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(math.ceil(len(enc_train) / cfg.micro_batch) / cfg.grad_accum)
    total = max(1, steps_per_epoch * cfg.epochs)
    warmup = int(cfg.warmup_ratio * total)

    def lr_lambda(step):            # montée linéaire puis décroissance cosinus
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    history = []
    base = evaluate(agent, enc_dev, cfg.micro_batch * 2, cfg.bf16) if enc_dev else {}
    log.info("dev avant entraînement : %s", _fmt_metrics(base))
    history.append({"epoch": 0, "dev": base})
    best = math.inf
    step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t_ep = time.perf_counter()
        run_loss, n_seen = 0.0, 0
        opt.zero_grad(set_to_none=True)
        batches = _batches(enc_train, cfg.micro_batch, rng)
        for i, batch in enumerate(batches, 1):
            loss, _, _ = _forward_loss(model, batch, agent.tok.pad_token_id, device, cfg.bf16)
            (loss.mean() / cfg.grad_accum).backward()
            run_loss += float(loss.detach().sum())
            n_seen += len(batch)
            if i % cfg.grad_accum == 0 or i == len(batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % cfg.log_every == 0:
                    log.info("époque %d, pas %d/%d : perte %.4f, lr %.2e", epoch, step, total,
                             run_loss / n_seen, sched.get_last_lr()[0])
        dev = evaluate(agent, enc_dev, cfg.micro_batch * 2, cfg.bf16) if enc_dev else {}
        entry = {"epoch": epoch, "train_loss": run_loss / max(1, n_seen), "dev": dev,
                 "seconds": time.perf_counter() - t_ep}
        history.append(entry)
        log.info("époque %d : perte train %.4f, dev %s (%.0f s)", epoch, entry["train_loss"],
                 _fmt_metrics(dev), entry["seconds"])
        dev_loss = selection_loss(dev, cfg.select_on)
        entry["selection_loss"] = dev_loss
        if dev_loss is None:
            dev_loss = -epoch                               # sans dev : on garde la dernière époque
        if dev_loss < best:
            best = dev_loss
            save_checkpoint(agent, out_dir, {**(meta or {}), "epoch": epoch, "train_config": asdict(cfg),
                                             "dev": dev})
            log.info("checkpoint enregistré (meilleure perte dev %s = %.4f) : %s", cfg.select_on, dev_loss, out_dir)
    model.eval()
    return {"history": history, "best_dev_loss": best, "steps": step}


def selection_loss(dev: dict, select_on: str = "balanced") -> float | None:
    """Valeur dev à minimiser pour choisir le checkpoint (None si le dev est vide).

    "f1" retombe sur "balanced" quand le dev n'a pas de question "choice" (mode two_step)."""
    if select_on == "f1":
        if "slot_f1" in dev:
            return 1.0 - dev["slot_f1"]["f1"]
        select_on = "balanced"
    if select_on == "all":
        return dev.get("all", {}).get("loss")
    if select_on != "balanced":
        raise ValueError(f"select_on inconnu : {select_on!r} (f1 | balanced | all)")
    parts = [dev[k]["loss"] for k in ("expressed", "unexpressed") if k in dev]
    return sum(parts) / len(parts) if parts else None


def _fmt_metrics(m: dict) -> str:
    parts = [f"{k} perte {v['loss']:.3f} exact. {v['accuracy']:.3f} (n={v['n']})"
             for k, v in sorted(m.items()) if k != "slot_f1"]
    if "slot_f1" in m:
        parts.append(f"F1 slots {m['slot_f1']['f1']:.4f}")
    return ", ".join(parts) or "-"
