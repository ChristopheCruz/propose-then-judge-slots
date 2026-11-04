"""Télécharge les modèles des expériences dans le cache Hugging Face ($HF_HOME).

    uv run --no-sync python -m experiments.telecharger_modeles              # tout, LLM de .env (*)
    uv run --no-sync python -m experiments.telecharger_modeles --sans-llm
    uv run --no-sync python -m experiments.telecharger_modeles --llm Qwen/Qwen3-32B --llm Qwen/...

Laya, GLiNER et sentence-transformers sont chargés par les classes du dépôt, sur CPU : le cache
contient exactement ce que les runs liront ensuite. Le LLM est seulement téléchargé
(`snapshot_download`), sans vLLM ni GPU.

(*) LLM par défaut : $LLM_MODEL, $LLM_MODEL_DEV, $LLM_MODEL_ALT, $LLM_MODEL_ALT2 (ceux qui sont
définis et non vides, sans doublon). `--llm` remplace cette liste.

Dépôt Mistral à double format (poids `consolidated*.safetensors` au format Mistral et
`model-*.safetensors` au format HF, ex. Mistral-Small-4 : 2 x 121 Go) : seul le format HF est
téléchargé. Hors ligne, vLLM ne peut pas lister le dépôt et charge donc le format HF
(config.json, model.safetensors.index.json) ; les fichiers consolidated ne serviraient pas.

Écrit MODELES.lock.json : dépôt, révision (sha du commit Hugging Face) et taille de chaque modèle
présent dans le cache. Ce fichier est versionné : il fixe les versions utilisées dans le papier.
Les runs se font ensuite hors ligne (HF_HUB_OFFLINE=1, posé par le Makefile).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path

log = logging.getLogger("telecharger_modeles")

GLINER_MODEL = "urchade/gliner_multi-v2.1"                                    # configs/candidats/reels.yaml
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # candidates/union.py
LAYA_CHECKPOINTS = ("multilingual", "english")
TAGGER_MODEL = "microsoft/mdeberta-v3-base"                                   # configs/train/c0.yaml


def laya() -> None:
    from laya_slots.judges.laya_judge import LayaJudge
    for ckpt in LAYA_CHECKPOINTS:
        log.info("Laya %s", ckpt)
        LayaJudge(checkpoint=ckpt, device="cpu")._get_predictor()


def gliner() -> None:
    from laya_slots.candidates.gliner import GlinerGenerator
    log.info("GLiNER %s", GLINER_MODEL)
    GlinerGenerator(model_name=GLINER_MODEL, device="cpu")._load()


def tagger() -> None:
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    log.info("tagger %s", TAGGER_MODEL)
    AutoTokenizer.from_pretrained(TAGGER_MODEL)
    AutoModelForTokenClassification.from_pretrained(TAGGER_MODEL, num_labels=3)   # poids fp16 (refs/pr/5)


def embeddings() -> None:
    from sentence_transformers import SentenceTransformer
    log.info("sentence-transformers %s", EMBEDDING_MODEL)
    SentenceTransformer(EMBEDDING_MODEL, device="cpu")


LLM_ENV_VARS = ("LLM_MODEL", "LLM_MODEL_DEV", "LLM_MODEL_ALT", "LLM_MODEL_ALT2")
LLM_ALLOW = ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.jinja", "*.py"]
MISTRAL_FORMAT = ["consolidated*.safetensors", "consolidated.safetensors.index.json"]


def default_llms(env: dict | None = None) -> list[str]:
    """LLM à télécharger par défaut : variables LLM_ENV_VARS définies, dans l'ordre, sans doublon."""
    env = os.environ if env is None else env
    return list(dict.fromkeys(m for m in (env.get(v, "").strip() for v in LLM_ENV_VARS) if m))


def ignore_patterns(files: list[str]) -> list[str] | None:
    """Fichiers à ne pas télécharger : le format Mistral quand le format HF est aussi présent."""
    names = [f.rsplit("/", 1)[-1] for f in files]
    mistral = any(fnmatch(n, MISTRAL_FORMAT[0]) for n in names)
    hf = any(n.endswith(".safetensors") and not fnmatch(n, MISTRAL_FORMAT[0]) for n in names)
    return list(MISTRAL_FORMAT) if mistral and hf else None


def llm(model: str) -> None:
    from huggingface_hub import list_repo_files, snapshot_download
    ignore = ignore_patterns(list_repo_files(model))
    log.info("LLM %s (téléchargement seul, peut prendre longtemps)%s", model,
             f" ; ignorés : {', '.join(ignore)}" if ignore else "")
    # Poids safetensors seulement : certains dépôts publient aussi des copies .bin ou .pth.
    snapshot_download(model, allow_patterns=LLM_ALLOW, ignore_patterns=ignore)


def write_lock(path: Path) -> list[dict]:
    """Inventaire du cache : une entrée par (dépôt, révision)."""
    from huggingface_hub import scan_cache_dir
    entries = []
    for repo in sorted(scan_cache_dir().repos, key=lambda r: r.repo_id):
        if repo.repo_type != "model":
            continue
        for rev in sorted(repo.revisions, key=lambda r: r.commit_hash):
            entries.append({"repo_id": repo.repo_id, "revision": rev.commit_hash,
                            "refs": sorted(rev.refs), "size_gb": round(rev.size_on_disk / 1e9, 2)})
    lock = {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "hf_home": os.environ.get("HF_HOME"), "models": entries}
    path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return entries


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", action="append", default=None,
                    help="LLM à télécharger (répétable) ; défaut : $LLM_MODEL, $LLM_MODEL_DEV, "
                         "$LLM_MODEL_ALT et $LLM_MODEL_ALT2 (ceux qui sont définis)")
    ap.add_argument("--sans-llm", action="store_true", help="ne pas télécharger de LLM")
    ap.add_argument("--lock", type=Path, default=Path("MODELES.lock.json"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        log.error("HF_HUB_OFFLINE=1 : téléchargement impossible, retirer la variable")
        return 2

    llms = [] if args.sans_llm else (args.llm or default_llms())
    steps = [("laya", laya), ("gliner", gliner), ("tagger", tagger), ("embeddings", embeddings)]
    steps += [(f"llm {m}", lambda m=m: llm(m)) for m in llms]
    failed = []
    for name, step in steps:
        try:
            step()
        except Exception as e:  # noqa: BLE001 — un échec ne doit pas bloquer les autres modèles
            log.error("échec %s : %s: %s", name, type(e).__name__, e)
            failed.append(name)

    entries = write_lock(args.lock)
    log.info("%s : %d modèles, %.1f Go", args.lock, len(entries), sum(e["size_gb"] for e in entries))
    if failed:
        log.error("étapes en échec : %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
