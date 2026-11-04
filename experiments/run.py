"""Lance une expérience : exemples -> candidats -> juge -> prédictions et métriques.

    uv run python -m experiments.run --config configs/pilote.yaml --output results/x [cle=valeur ...]

Écrit dans --output :
    config.yaml        config résolue (valeurs par défaut et surcharges comprises)
    predictions.jsonl  une ligne par JudgeRequest, écrite au fil de l'eau
    metrics.json       métriques, coût, provenance (commit git, hash de config et de données)
    run.log            journal lisible

SIGTERM / SIGINT : le lot en cours se termine, puis ce qui est calculé est écrit
(metrics.json avec "status": "interrupted", calculé sur les exemples entièrement jugés).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import itertools
import json
import logging
import os
import platform
import random
import signal
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from experiments.config import config_sha256, instantiate, load_config, run_settings
from laya_slots.evaluation import metrics as M
from laya_slots.schema import (
    NONE_VALUE,
    Candidate,
    Example,
    JudgeRequest,
    Judgment,
    Ontology,
    SlotSpec,
    read_examples,
)

log = logging.getLogger("experiments.run")


# ---------------------------------------------------------------------------
# Arrêt propre
# ---------------------------------------------------------------------------
class StopFlag:
    """Positionné par SIGTERM/SIGINT ; la boucle le consulte entre deux lots.

    Le gestionnaire ne lève pas d'exception : un lot GPU en cours n'est pas interrompu
    au milieu. Plusieurs signaux (uv relaie aussi SIGTERM au processus Python) sont sans effet
    supplémentaire ; labo.sh envoie SIGKILL 60 s après.
    """

    def __init__(self):
        self.signum: int | None = None

    def __bool__(self):
        return self.signum is not None

    def install(self):
        for s in (signal.SIGTERM, signal.SIGINT):
            signal.signal(s, self._handler)

    def _handler(self, signum, frame):
        first = self.signum is None
        self.signum = self.signum or signum   # avant le log : un 2e signal peut arriver pendant l'écriture
        if first:
            log.warning("signal %s reçu : fin du lot en cours puis écriture des résultats",
                        signal.Signals(signum).name)


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if importlib.util.find_spec("torch") is not None:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


CODE_DIRS = ("laya_slots", "experiments", "configs")
REPO_ROOT = Path(__file__).resolve().parents[1]


def code_sha256(root: Path = REPO_ROOT) -> str:
    """Empreinte du code exécuté (.py et .yaml de laya_slots/, experiments/, configs/).

    Ne dépend pas de .git : identifie le code même quand le dépôt est copié sans son historique
    (rsync). Comparer avec `python -c "from experiments.run import code_sha256; print(code_sha256())"`
    sur le poste."""
    h = hashlib.sha256()
    for d in CODE_DIRS:
        for f in sorted((root / d).rglob("*")):
            if f.suffix in (".py", ".yaml") and "__pycache__" not in f.parts:
                h.update(str(f.relative_to(root)).encode())
                h.update(f.read_bytes())
    return h.hexdigest()


def git_info() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10,
                              check=True).stdout.strip()
    try:
        info = {"git_commit": git("rev-parse", "HEAD"),
                "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.SubprocessError):
        info = {"git_commit": None, "git_dirty": None}
    info["code_sha256"] = code_sha256()
    return info


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def setup_logging(out: Path) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in (logging.StreamHandler(sys.stderr), logging.FileHandler(out / "run.log", encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------
def _split_dir(d: DictConfig) -> Path:
    return Path(d.root) / d.dataset / d.lang


def load_data(cfg: DictConfig) -> tuple[list[Example], Ontology, dict]:
    """Exemples (limités), ontologie et provenance des données."""
    d = cfg.data
    info: dict = {}
    if d.loader:
        loader = instantiate(d.loader, cfg.seed)
        examples: Iterable[Example] = loader.load(d.lang, d.split)
        ontology = loader.ontology(d.lang)
        info["data_source"] = f"loader:{d.loader['class']}"
    else:
        path = Path(d.path) if d.path else _split_dir(d) / f"{d.split}.jsonl"
        onto_path = Path(d.ontology) if d.ontology else path.parent / "ontology.json"
        if not path.exists():
            raise FileNotFoundError(f"données introuvables : {path} (voir CONTRAT.md, data/processed)")
        examples = read_examples(path)
        ontology = Ontology.from_dict(json.loads(onto_path.read_text(encoding="utf-8")))
        info.update(data_source=str(path), data_sha256=sha256_file(path),
                    ontology_sha256=sha256_file(onto_path))
    if d.limit is not None:
        examples = itertools.islice(examples, int(d.limit))
    return list(examples), ontology, info


def load_train(cfg: DictConfig) -> list[Example]:
    d = cfg.data
    if d.loader:
        return list(instantiate(d.loader, cfg.seed).load(d.lang, d.train_split))
    path = Path(d.train_path) if d.train_path else (
        (Path(d.path).parent if d.path else _split_dir(d)) / f"{d.train_split}.jsonl")
    return list(read_examples(path))


# ---------------------------------------------------------------------------
# Construction des requêtes
# ---------------------------------------------------------------------------
def slots_for(intent: str | None, ontology: Ontology, mode: str) -> tuple[list[SlotSpec], bool]:
    """Slots à interroger, et un drapeau « repli sur toute l'ontologie ».

    mode=intent : slots autorisés par l'intention (les intentions multiples « a#b » de
    MultiATIS++ prennent l'union) ; intention absente ou inconnue -> toute l'ontologie."""
    if mode == "intent":
        parts = [] if intent is None else ([intent] if intent in ontology.intents else intent.split("#"))
        if parts and all(p in ontology.intents for p in parts):
            names: list[str] = []
            for p in parts:
                names += [s for s in ontology.intents[p].slots if s not in names]
            return [ontology.slots[n] for n in names if n in ontology.slots], False
        return list(ontology.slots.values()), True
    return list(ontology.slots.values()), False


def merge_candidates(lists: Iterable[list[Candidate]], max_candidates: int | None) -> list[Candidate]:
    """Union des générateurs, dédoublonnée par valeur (le premier générateur l'emporte)."""
    seen, out = set(), []
    for c in itertools.chain.from_iterable(lists):
        if c.value != NONE_VALUE and c.value not in seen:
            seen.add(c.value)
            out.append(c)
    return out[:max_candidates] if max_candidates else out


@dataclasses.dataclass
class Pending:
    request: JudgeRequest
    candidates: list[Candidate]
    gold_values: list[str]
    gen_ms: float


class RequestBuilder:
    def __init__(self, cfg: DictConfig, ontology: Ontology, generators: list, intent_predictor=None):
        self.cfg, self.ontology, self.generators = cfg, ontology, generators
        self.intent_predictor = intent_predictor
        self.unknown_intents = 0

    def judge_view(self, ex: Example, intent: str | None) -> Example:
        """Copie de l'exemple vue par le juge : sans les gold du tour, et sans l'intention gold
        si la config ne la donne pas."""
        r = self.cfg.requests
        changes = {}
        if r.strip_gold:
            changes["slots"] = []
        if r.intent != "gold":
            changes["intent"] = intent
        return dataclasses.replace(ex, **changes) if changes else ex

    def predict_intents(self, examples: list[Example]) -> list[str | None]:
        mode = self.cfg.requests.intent
        if mode == "gold":
            return [ex.intent for ex in examples]
        if mode == "none":
            return [None] * len(examples)
        if mode == "domain":
            # Multi3WOZ : pas d'intention par tour, mais une « intention » par domaine dans l'ontologie.
            # Le domaine gold du tour joue le rôle de l'intention gold (mêmes conséquences : slots
            # interrogés = ceux du domaine ; domaine inconnu -> toute l'ontologie).
            return [ex.domain for ex in examples]
        hidden = [dataclasses.replace(ex, slots=[], intent=None) for ex in examples]
        return list(self.intent_predictor.predict_batch(hidden))

    def build(self, ex: Example, intent: str | None) -> list[Pending]:
        r, c = self.cfg.requests, self.cfg.candidates
        specs, fallback = slots_for(intent, self.ontology, r.slots)
        self.unknown_intents += fallback
        view = self.judge_view(ex, intent)
        filled = dict(ex.state_before) if r.filled == "state_before" else {}
        gold: dict[str, list[str]] = {}
        for sv in ex.slots:
            gold.setdefault(sv.name, []).append(sv.value)
        out = []
        for spec in specs:
            t0 = time.perf_counter()
            # les générateurs reçoivent l'exemple complet (seul l'oracle a le droit de lire les gold)
            cands = merge_candidates((g.generate(ex, spec, self.ontology) for g in self.generators),
                                     c.max_candidates)
            if c.shuffle:
                random.Random(f"{self.cfg.seed}:{ex.id}:{spec.name}").shuffle(cands)
            gen_ms = (time.perf_counter() - t0) * 1000
            req = JudgeRequest(example=view, slot=spec, candidates=cands, intent=intent, filled=filled)
            out.append(Pending(req, cands, gold.get(spec.name, []), gen_ms))
        return out


# ---------------------------------------------------------------------------
# Sorties
# ---------------------------------------------------------------------------
def prediction_record(p: Pending, j: Judgment, latency_ms: float, save_probs: bool) -> dict:
    choice_value = NONE_VALUE if j.choice is None else p.candidates[j.choice].value
    gold_value = (choice_value if choice_value in p.gold_values
                  else (p.gold_values[0] if p.gold_values else NONE_VALUE))
    rec = {
        "example_id": p.request.example.id,
        "slot": p.request.slot.name,
        "intent": p.request.intent,
        "candidates": [{"value": c.value, "source": c.source, "score": c.score} for c in p.candidates],
        "choice": j.choice,
        "choice_value": choice_value,
        "proba": j.proba,
        "p_expressed": j.p_expressed,
        "gold_value": gold_value,
        "gold_values": p.gold_values,
        "latency_ms": latency_ms,
        "gen_ms": p.gen_ms,
    }
    if save_probs:
        rec["probs"] = j.probs
        rec["raw"] = j.raw
    return rec


def compute_metrics(records: list[dict], examples: list[Example],
                    pred_intents: dict[str, str | None] | None) -> dict:
    gold = {ex.id: [(s.name, s.value) for s in ex.slots] for ex in examples}
    pred: dict[str, list[tuple[str, str]]] = {ex.id: [] for ex in examples}
    for r in records:
        if r["example_id"] in pred and r["choice_value"] != NONE_VALUE:
            pred[r["example_id"]].append((r["slot"], r["choice_value"]))
    kept = [r for r in records if r["example_id"] in gold]
    gold_int = {ex.id: ex.intent for ex in examples}
    out = {**M.slot_prf(gold, pred)}
    out["intent_accuracy"] = M.intent_accuracy(gold_int, pred_intents) if pred_intents else None
    out["frame_accuracy"] = M.frame_accuracy(gold, pred, gold_int if pred_intents else None, pred_intents)
    out.update(M.candidate_recall(kept))
    out["decomposition"] = M.error_decomposition(kept, gold)
    out.update(M.abstention_stats(kept))
    return out


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------
def chunks(xs: list, n: int) -> Iterator[list]:
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def run(cfg: DictConfig, out: Path, stop: StopFlag | None = None) -> dict:
    stop = StopFlag() if stop is None else stop
    started = datetime.now(UTC)
    t_start = time.perf_counter()
    (out / "config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    set_seeds(cfg.seed)

    examples, ontology, data_info = load_data(cfg)
    log.info("%d exemples, %d intentions, %d slots dans l'ontologie",
             len(examples), len(ontology.intents), len(ontology.slots))
    generators = [instantiate(g, cfg.seed) for g in cfg.candidates.generators]
    if cfg.candidates.fit and generators:
        train = load_train(cfg)
        for g in generators:
            g.fit(train, ontology)
        log.info("générateurs ajustés sur %d exemples d'entraînement", len(train))
    judge_spec = OmegaConf.to_container(cfg.judge, resolve=True)
    if judge_spec.get("finetuned"):
        # Garde-fou : un checkpoint absent ne doit pas faire tourner le Laya de base sous l'étiquette C2/C3.
        ckpt = Path(judge_spec["finetuned"])
        if not (ckpt / "model.safetensors").exists():
            raise FileNotFoundError(f"judge.finetuned={ckpt} : checkpoint introuvable "
                                    "(le produire avec : make entrainer-c2 / entrainer-c3)")
        judge_spec.setdefault("params", {})["model_path"] = str(ckpt.resolve())
    judge = instantiate(judge_spec, cfg.seed)
    intent_predictor = instantiate(cfg.intent_predictor, cfg.seed) if cfg.intent_predictor else None
    builder = RequestBuilder(cfg, ontology, generators, intent_predictor)
    log.info("juge %s, générateurs %s", type(judge).__name__, [type(g).__name__ for g in generators])

    bs = int(cfg.judge.batch_size)
    save_probs = bool(cfg.output.save_probs)
    records: list[dict] = []
    latencies: list[float] = []
    remaining: dict[str, int] = {}
    done_ids: set[str] = set()
    pred_intents: dict[str, str | None] = {}
    pending: list[Pending] = []
    judge_s = gen_s = 0.0
    status = "complete"
    t_loop = time.perf_counter()

    preds_f = (out / "predictions.jsonl").open("w", encoding="utf-8")

    def flush(batch: list[Pending]) -> None:
        nonlocal judge_s
        t0 = time.perf_counter()
        judgments = judge.judge_batch([p.request for p in batch])
        dt = time.perf_counter() - t0
        judge_s += dt
        if len(judgments) != len(batch):
            raise RuntimeError(f"le juge a rendu {len(judgments)} décisions pour {len(batch)} requêtes")
        for p, j in zip(batch, judgments):
            lat = j.latency_ms if j.latency_ms is not None else dt * 1000 / len(batch)
            rec = prediction_record(p, j, lat, save_probs)
            preds_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records.append(rec)
            latencies.append(lat)
            ex_id = rec["example_id"]
            remaining[ex_id] -= 1
            if remaining[ex_id] == 0:
                done_ids.add(ex_id)
        preds_f.flush()
        if len(records) // cfg.output.log_every != (len(records) - len(batch)) // cfg.output.log_every:
            log.info("%d requêtes jugées, %d/%d exemples terminés", len(records), len(done_ids),
                     len(examples))

    try:
        for chunk in chunks(examples, max(bs, 1)):
            if stop:
                break
            intents = builder.predict_intents(chunk)
            for ex, intent in zip(chunk, intents):
                pred_intents[ex.id] = intent
                t0 = time.perf_counter()
                reqs = builder.build(ex, intent)
                gen_s += time.perf_counter() - t0
                remaining[ex.id] = len(reqs)
                if not reqs:
                    done_ids.add(ex.id)
                pending.extend(reqs)
                while len(pending) >= bs and not stop:
                    flush(pending[:bs])
                    pending = pending[bs:]
        if pending and not stop:
            flush(pending)
            pending = []
        if stop:
            status = "interrupted"
    except Exception:
        status = "error"
        log.exception("échec du run ; écriture des résultats partiels")
        raise
    finally:
        preds_f.close()
        wall = time.perf_counter() - t_start
        loop_s = time.perf_counter() - t_loop
        done = [ex for ex in examples if ex.id in done_ids]
        recs_done = [r for r in records if r["example_id"] in done_ids]
        intents_done = ({i: pred_intents[i] for i in done_ids}
                        if cfg.requests.intent == "predicted" else None)
        m = {
            "status": status,
            "name": cfg.name, "condition": cfg.condition,
            "dataset": cfg.data.dataset, "lang": cfg.data.lang, "split": cfg.data.split,
            "judge": cfg.judge["class"], "seed": cfg.seed,
            "settings": run_settings(cfg),
            "n_examples": len(done), "n_examples_planned": len(examples),
            "n_requests": len(recs_done), "n_requests_judged": len(records),
            "n_unknown_intents": builder.unknown_intents,
            **compute_metrics(recs_done, done, intents_done),
            **M.latency_stats(latencies),
            "judge_time_s": judge_s, "generation_time_s": gen_s, "wall_time_s": wall,
            "throughput_req_per_s": len(records) / judge_s if judge_s else None,
            "throughput_ex_per_s": len(done) / loop_s if loop_s else None,
            **git_info(),
            "config_sha256": config_sha256(cfg), **data_info,
            "run_id": os.environ.get("RUN_ID"),
            "host": platform.node(), "python": platform.python_version(),
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        if "torch" in sys.modules:
            torch = sys.modules["torch"]
            if torch.cuda.is_available():
                m["gpu_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
        (out / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("%s : %d exemples, F1 slots = %s, rappel candidats = %s, p95 = %s ms",
                 status, m["n_examples"], _fmt(m["slot_f1"]), _fmt(m["candidate_recall"]),
                 _fmt(m["latency_ms_p95"]))
    return m


def _fmt(x) -> str:
    return "-" if x is None else f"{x:.4f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("overrides", nargs="*", help="surcharges OmegaConf cle.sous_cle=valeur")
    args = ap.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out)
    stop = StopFlag()
    stop.install()
    cfg = load_config(args.config, args.overrides)
    log.info("config %s -> %s %s", args.config, out, " ".join(args.overrides))
    m = run(cfg, out, stop)
    if m["status"] == "interrupted":
        return 128 + int(stop.signum or signal.SIGTERM)
    return 0


if __name__ == "__main__":
    sys.exit(main())
