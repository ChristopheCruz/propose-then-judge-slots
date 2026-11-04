"""Toutes les configs du dépôt se chargent et se résolvent."""
from pathlib import Path

import pytest

from experiments.config import load_config

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
ALL = sorted(p for p in CONFIGS.rglob("*.yaml") if not p.name.startswith("_") and p.parent.name != "candidats")
# valeurs obligatoires laissées à « ??? » dans les squelettes
FILL = {"c4_llm_juge.yaml": ["judge.params.model=x/y"], "c4_llm_juge_reels.yaml": ["judge.params.model=x/y"],
        "c5_llm_extracteur.yaml": ["candidates.generators.0.params.model=x/y"]}


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.name)
def test_config_loads(path):
    cfg = load_config(path, FILL.get(path.name, []))
    assert cfg.judge["class"].count(".") >= 2
    for g in cfg.candidates.generators:
        assert g["class"].startswith("laya_slots.candidates.")
    assert cfg.seed == 13


def test_pilote():
    cfg = load_config(CONFIGS / "pilote.yaml")
    assert (cfg.data.dataset, cfg.data.lang, cfg.data.split, cfg.data.limit) == ("multiatis", "en", "test", 200)
    p = cfg.candidates.generators[0].params
    assert (p.k, p.p_absent, p.seed) == (5, 0.0, 13)


def test_reels_variant_and_seed_interpolation():
    cfg = load_config(CONFIGS / "conditions" / "c1_laya_zeroshot_reels.yaml", ["seed=14", "data.lang=fr"])
    assert cfg.condition == "C1" and cfg.candidates.fit is True and len(cfg.candidates.generators) == 4
    c2 = load_config(CONFIGS / "conditions" / "c2_laya_ft.yaml", ["seed=14", "data.lang=fr"])
    assert c2.judge.finetuned == "checkpoints/c2/multiatis-fr-s14"
    assert c2.candidates.generators[0].params.seed == 14


def test_missing_mandatory_value():
    with pytest.raises(Exception, match="model"):
        load_config(CONFIGS / "conditions" / "c4_llm_juge.yaml")
