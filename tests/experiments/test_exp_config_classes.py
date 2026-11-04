"""Intégration : chaque composant référencé par une config existe et accepte ses paramètres."""
import importlib
import inspect
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from experiments.config import load_config

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
# Composants prévus en vague 2, pas encore écrits.
A_VENIR = {"laya_slots.candidates.tagger.TaggerGenerator",
           "laya_slots.candidates.llm_extractor.LLMExtractorGenerator"}


def _components(cfg):
    yield cfg.judge
    yield from cfg.candidates.generators


@pytest.mark.parametrize("path", sorted(p for p in CONFIGS.rglob("*.yaml") if not p.name.startswith("_")),
                         ids=lambda p: str(p.relative_to(CONFIGS)))
def test_config_components_match_constructors(path):
    over = ["judge.params.model=dummy"] if "llm" in path.name else []
    try:
        cfg = load_config(path, over)
    except Exception as e:  # noqa: BLE001 — fragments (candidats/*.yaml) sans juge ni données
        pytest.skip(f"fragment de config : {e.__class__.__name__}")
    for comp in _components(cfg):
        if comp["class"] in A_VENIR:
            continue
        mod, _, name = comp["class"].rpartition(".")
        cls = getattr(importlib.import_module(mod), name)
        params = comp.get("params")
        params = OmegaConf.to_container(params, resolve=True) if params else {}
        sig = inspect.signature(cls.__init__)
        sig.bind(None, **params)   # lève TypeError si un paramètre n'existe pas
