"""Préparation des jeux convertis.

    python -m laya_slots.data.prepare --dataset multiatis --lang en [--raw-dir ...] [--out-dir data/processed]
    python -m laya_slots.data.prepare --dataset mtop --lang fr --download

Écrit dans <out-dir>/<dataset>/<lang>/ :
- train.jsonl, dev.jsonl, test.jsonl (un `Example` par ligne) ;
- ontology.json ;
- MANIFEST.json : version source, sha256 des fichiers bruts et des fichiers produits, nombre
  d'exemples, statistiques ; il n'a pas d'horodatage pour que deux préparations identiques donnent
  des fichiers identiques.
Le résumé statistique est aussi affiché.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from ..schema import Example, write_jsonl
from . import get_loader
from .common import SPLITS, sha256_file


def split_stats(examples: list[Example]) -> dict:
    """Statistiques d'un split : exemples, intentions, slots, slots par énoncé, spans, imbrication."""
    n = len(examples)
    n_slots = [len(e.slots) for e in examples]
    intents = Counter(e.intent for e in examples if e.intent is not None)
    slot_names = Counter(s.name for e in examples for s in e.slots)
    origins = Counter(s.origin for e in examples for s in e.slots)
    with_span = sum(1 for e in examples for s in e.slots if s.span is not None)
    total = sum(n_slots)
    stats = {
        "examples": n,
        "intents": len(intents),
        "slot_types": len(slot_names),
        "slots": total,
        "slots_per_example": round(total / n, 3) if n else 0.0,
        "examples_without_slots": sum(1 for k in n_slots if k == 0),
        "max_slots_per_example": max(n_slots, default=0),
        "slots_with_span": with_span,
        "slot_origins": dict(sorted(origins.items())),
        "domains": len({e.domain for e in examples if e.domain}),
    }
    multi = sum(1 for e in examples if len(e.meta.get("intents", [])) > 1)
    if multi:
        stats["multi_intent_examples"] = multi
    nested = sum(1 for e in examples if e.meta.get("has_nested"))
    if nested:
        stats["nested_examples"] = nested
    if any(e.history for e in examples):
        stats["dialogues"] = len({e.meta.get("dialogue_id") for e in examples})
        stats["mean_history_turns"] = round(sum(len(e.history) for e in examples) / n, 2) if n else 0.0
    return stats


def prepare(dataset: str, lang: str, raw_dir: str | Path | None = None,
            out_dir: str | Path = "data/processed", download: bool = False) -> dict:
    loader = get_loader(dataset, raw_dir)
    if download:
        if not hasattr(loader, "download"):
            raise SystemExit(f"{dataset} n'est pas téléchargeable automatiquement (voir la docstring du module)")
        loader.download()
    out = Path(out_dir) / dataset / lang
    out.mkdir(parents=True, exist_ok=True)

    files, stats = {}, {}
    for split in SPLITS:
        examples = list(loader.load(lang, split))
        path = out / f"{split}.jsonl"
        write_jsonl(path, examples)
        files[path.name] = {"sha256": sha256_file(path), "examples": len(examples)}
        stats[split] = split_stats(examples)

    onto = loader.ontology(lang)
    onto_path = out / "ontology.json"
    onto_path.write_text(onto.to_json() + "\n", encoding="utf-8")
    files[onto_path.name] = {"sha256": sha256_file(onto_path), "intents": len(onto.intents),
                             "slots": len(onto.slots)}

    sources = {}
    source_files = getattr(loader, "source_files", None)
    if source_files is not None:
        for p in source_files(lang):
            sources[Path(p).name if Path(p).name not in sources else str(p)] = {"sha256": sha256_file(p)}
    manifest = {
        "dataset": dataset,
        "lang": lang,
        "source_version": getattr(loader, "source_version", None),
        "source_files": sources,
        "files": files,
        "stats": stats,
        "skipped_lines": {s: n for (lg, s), n in getattr(loader, "skipped", {}).items() if lg == lang and n},
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    return manifest


def format_summary(manifest: dict) -> str:
    lines = [f"{manifest['dataset']} / {manifest['lang']}  ({manifest['source_version']})",
             (f"{'split':<6} {'exemples':>9} {'intentions':>10} {'types slot':>10} {'slots':>7} "
              f"{'slots/énoncé':>12} {'sans slot':>9} {'avec span':>9}")]
    for split, s in manifest["stats"].items():
        lines.append(f"{split:<6} {s['examples']:>9} {s['intents']:>10} {s['slot_types']:>10} {s['slots']:>7} "
                     f"{s['slots_per_example']:>12} {s['examples_without_slots']:>9} {s['slots_with_span']:>9}")
    onto = manifest["files"]["ontology.json"]
    lines.append(f"ontologie : {onto['intents']} intentions, {onto['slots']} slots")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["multiatis", "mtop", "multi3woz"])
    ap.add_argument("--lang", required=True)
    ap.add_argument("--raw-dir", default=None, help="répertoire brut (défaut : data/raw/<dataset>)")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--download", action="store_true", help="télécharge les données si le jeu est libre")
    args = ap.parse_args(argv)
    try:
        manifest = prepare(args.dataset, args.lang, args.raw_dir, args.out_dir, args.download)
    except FileNotFoundError as e:
        print(f"Données brutes introuvables : {e}", file=sys.stderr)
        print("Voir la docstring de laya_slots/data/<dataset>.py pour les obtenir.", file=sys.stderr)
        return 2
    print(format_summary(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
