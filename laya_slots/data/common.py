"""Outils communs aux chargeurs : chemins, BIO -> spans, offsets, recherche de fichiers."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from ..schema import Split

# Répertoire des données brutes : $LAYA_DATA_RAW, sinon data/raw (relatif au répertoire courant).
DEFAULT_RAW_DIR = Path(os.environ.get("LAYA_DATA_RAW", "data/raw"))
SPLITS: tuple[Split, ...] = ("train", "dev", "test")


def raw_dir_for(dataset: str, raw_dir: str | Path | None = None) -> Path:
    """Répertoire brut d'un jeu : `raw_dir` s'il est donné, sinon <DEFAULT_RAW_DIR>/<dataset>."""
    return Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR / dataset


def find_file(root: Path, *candidates: str) -> Path:
    """Cherche récursivement sous `root` le premier fichier dont le chemin finit par un des `candidates`.

    Les archives officielles n'ont pas toutes la même arborescence une fois décompressées (dossier
    racine ou non, packaging LDC) : on cherche donc par suffixe de chemin, insensible à la casse.
    """
    root = Path(root)
    wanted = [c.lower().replace("\\", "/") for c in candidates]
    if root.is_dir():
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            rel = p.as_posix().lower()
            if any(rel.endswith("/" + w) or rel == w for w in wanted):
                return p
    raise FileNotFoundError(f"aucun de {list(candidates)} sous {root}")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def token_offsets(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    """Offsets caractères [début, fin) de chaque jeton, retrouvés séquentiellement dans `text`.

    Lève ValueError si un jeton n'est pas retrouvé (texte et jetons incohérents).
    """
    offsets, pos = [], 0
    for tok in tokens:
        start = text.find(tok, pos)
        if start < 0:
            raise ValueError(f"jeton {tok!r} introuvable après la position {pos} dans {text!r}")
        offsets.append((start, start + len(tok)))
        pos = start + len(tok)
    return offsets


def bio_to_spans(labels: list[str]) -> list[tuple[str, int, int]]:
    """Convertit des étiquettes BIO en spans (label, i_début, i_fin) en indices de JETONS, fin exclue.

    Règle tolérante (comme conlleval) : un I-x qui ne prolonge pas un x ouvert commence un nouveau span.
    Les étiquettes sans préfixe autre que « O » sont traitées comme B-.
    """
    spans: list[tuple[str, int, int]] = []
    cur: str | None = None
    start = 0
    for i, lab in enumerate(list(labels) + ["O"]):
        if lab == "O" or not lab:
            tag, name = "O", None
        elif lab[:2] in ("B-", "I-"):
            tag, name = lab[0], lab[2:]
        else:
            tag, name = "B", lab
        if cur is not None and (tag != "I" or name != cur):
            spans.append((cur, start, i))
            cur = None
        if tag == "B" or (tag == "I" and cur is None):
            cur, start = name, i
    return spans


def byte_to_char_offsets(text: str, b_start: int, b_end: int) -> tuple[int, int]:
    """Convertit des offsets en octets UTF-8 en offsets caractères."""
    enc = text.encode("utf-8")
    return len(enc[:b_start].decode("utf-8", errors="ignore")), len(enc[:b_end].decode("utf-8", errors="ignore"))


def humanize(name: str) -> str:
    """Nom d'étiquette -> texte lisible de secours (« fromloc.city_name » -> « fromloc city name »)."""
    for prefix in ("IN:", "SL:"):
        name = name.removeprefix(prefix)
    return " ".join(name.replace(".", " ").replace("_", " ").replace("-", " ").lower().split())
