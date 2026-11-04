"""Chargeurs des jeux (MultiATIS++, MTOP, Multi3WOZ) vers le format unifié de `laya_slots.schema`.

Usage :
    loader = get_loader("mtop")                 # données brutes sous data/raw/mtop/
    for ex in loader.load("fr", "dev"): ...
    onto = loader.ontology("fr")

Préparation des fichiers gelés : `python -m laya_slots.data.prepare --dataset mtop --lang fr`.
Les choix de conversion sont documentés dans la docstring de chaque module.
"""
from __future__ import annotations

from pathlib import Path

from ..interfaces import DatasetLoader
from .mtop import MtopLoader
from .multi3woz import Multi3WozLoader
from .multiatis import MultiAtisLoader

LOADERS: dict[str, type[DatasetLoader]] = {
    "multiatis": MultiAtisLoader,
    "mtop": MtopLoader,
    "multi3woz": Multi3WozLoader,
}


def get_loader(name: str, raw_dir: str | Path | None = None) -> DatasetLoader:
    try:
        cls = LOADERS[name]
    except KeyError:
        raise ValueError(f"jeu inconnu : {name!r} (connus : {sorted(LOADERS)})") from None
    return cls(raw_dir=raw_dir)


__all__ = ["LOADERS", "MtopLoader", "Multi3WozLoader", "MultiAtisLoader", "get_loader"]
