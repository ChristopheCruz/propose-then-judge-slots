"""experiments.train : préparation des items (sans torch ni Laya) et branchement des checkpoints C2/C3."""
import math
from collections import Counter
from pathlib import Path

import exp_fakes
import pytest
from omegaconf import OmegaConf

from experiments import run as R
from experiments import train as T
from experiments.config import load_config
from laya_slots.schema import write_jsonl

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


@pytest.fixture
def data_root(tmp_path):
    d = exp_fakes.write_fixture(tmp_path / "processed", n=24)
    write_jsonl(d / "dev.jsonl", exp_fakes.make_examples(12, "dev"))
    return tmp_path / "processed"


def _cfg(data_root, **train):
    cfg = OmegaConf.create({
        "name": "t", "condition": "C2", "seed": 7,
        "data": {"dataset": "multiatis", "lang": "en", "split": "test", "root": str(data_root)},
        "candidates": {"generators": [{"class": "laya_slots.candidates.oracle.OracleGenerator",
                                       "params": {"k": 3, "distractors": "hard"}}], "fit": True},
        "judge": {"class": "laya_slots.judges.laya_judge.LayaJudge",
                  "params": {"mode": "single", "candidate_order": "shuffle"}},
        "train": {"dev_limit": None, **train},
    })
    path = data_root / "train.yaml"
    path.write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    return load_config(path)


def test_prepare_counts_and_unexpressed_ratio(data_root):
    tr, dv, stats = T.prepare(_cfg(data_root, unexpressed_ratio=1.0))
    s = stats["train"]
    assert s["examples"] == 24 and s["requests_expressed"] > 0
    assert s["requests_unexpressed_kept"] <= s["requests_unexpressed"]
    assert s["items"] == len(tr) == s["requests_expressed"] + s["requests_unexpressed_kept"]
    # par exemple : au plus ceil(1.0 * max(1, exprimées)) requêtes non exprimées
    per_ex = Counter((it.meta["example_id"], it.expressed) for it in tr)
    for ex_id in {k[0] for k in per_ex}:
        assert per_ex[(ex_id, False)] <= math.ceil(max(1, per_ex[(ex_id, True)]))
    assert {it.kind for it in tr} == {"choice"}
    # dev : toutes les requêtes, sans sous-échantillonnage
    assert stats["dev"]["requests_unexpressed_kept"] == stats["dev"]["requests_unexpressed"]
    assert stats["dev"]["examples"] == 12 and dv


def test_prepare_keeps_all_unexpressed_when_ratio_is_null(data_root):
    _, _, stats = T.prepare(_cfg(data_root, unexpressed_ratio=None))
    assert stats["train"]["requests_unexpressed_kept"] == stats["train"]["requests_unexpressed"]


def test_prepare_is_deterministic(data_root):
    a, _, _ = T.prepare(_cfg(data_root))
    b, _, _ = T.prepare(_cfg(data_root))
    assert [(i.meta, i.target, i.question) for i in a] == [(i.meta, i.target, i.question) for i in b]


def test_targets_point_to_gold_or_none(data_root):
    tr, _, _ = T.prepare(_cfg(data_root, unexpressed_ratio=None))
    gold = {(e.id, s.name): s.value for e in exp_fakes.make_examples(24, "train") for s in e.slots}
    for it in tr:
        opts = list(it.question["criteria"].values())
        g = gold.get((it.meta["example_id"], it.meta["slot"]))
        if g is None:
            assert it.target == len(opts) - 1                    # « non précisé »
        else:
            assert opts[it.target] == g


def test_augment_varies_list_size(data_root):
    tr, dv, _ = T.prepare(_cfg(data_root, augment={"k": [2, 5], "p_absent": [0.0], "distractors": ["easy", "hard"]}))
    sizes = Counter(len(it.question["criteria"]) - 1 for it in tr)   # sans « non précisé »
    assert set(sizes) <= {1, 2, 3, 4, 5} and len(sizes) > 1
    assert {len(it.question["criteria"]) - 1 for it in dv} <= {1, 2, 3}  # dev : oracle de la config (k=3)


def test_two_step_items(data_root):
    cfg = _cfg(data_root)
    cfg.judge.params.mode = "two_step"
    tr, _, stats = T.prepare(cfg)
    assert {"expressed", "choice_given"} <= {it.kind for it in tr}
    assert stats["train"]["items_expressed"] == stats["train"]["requests_expressed"] + \
        stats["train"]["requests_unexpressed_kept"]


@pytest.mark.parametrize("name", ["c2.yaml", "c3.yaml"])
def test_train_configs_load_and_match_eval_judge(name):
    over = ["seed=14", "data.lang=fr"]
    cfg = load_config(CONFIGS / "train" / name, over)
    assert cfg.candidates.fit is True and cfg.condition == name[:2].upper()
    ev = load_config(CONFIGS / "conditions" / ("c2_laya_ft.yaml" if name == "c2.yaml" else "c3_laya_ft_aug.yaml"),
                     over)
    # questions d'entraînement = questions d'évaluation : même juge, même oracle
    assert OmegaConf.to_container(cfg.judge.params) == OmegaConf.to_container(ev.judge.params)
    assert OmegaConf.to_container(cfg.candidates) == OmegaConf.to_container(ev.candidates)
    assert "finetuned" not in cfg.judge
    known = set(T.TRAIN_DEFAULTS)
    from dataclasses import fields
    try:
        from laya_slots.train.trainer import TrainConfig
        known |= {f.name for f in fields(TrainConfig)}
    except ImportError:          # pragma: no cover
        pass
    assert set(cfg.train) <= known, set(cfg.train) - known
    if name == "c3.yaml":
        assert set(cfg.train.augment) == {"k", "p_absent", "distractors"}


# --- évaluation des checkpoints fine-tunés (judge.finetuned -> LayaJudge.model_path) --------------
def _finetuned_cfg(tmp_path, data_root, ckpt):
    cfg = exp_fakes.write_config(tmp_path / "c.yaml", data_root)
    raw = OmegaConf.load(cfg)
    raw.judge = {"class": "laya_slots.judges.laya_judge.LayaJudge", "finetuned": str(ckpt),
                 "params": {"mode": "single"}}
    OmegaConf.save(raw, cfg)
    return cfg


def test_run_refuses_missing_finetuned_checkpoint(tmp_path, data_root):
    cfg = _finetuned_cfg(tmp_path, data_root, tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="checkpoint introuvable"):
        R.main(["--config", str(cfg), "--output", str(tmp_path / "run")])


def test_run_passes_checkpoint_to_laya_judge(tmp_path, data_root, monkeypatch):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"")
    seen = {}
    real = R.instantiate

    def fake_instantiate(spec, seed=None):
        if isinstance(spec, dict) and spec.get("class", "").endswith("LayaJudge"):
            seen.update(spec)
            return exp_fakes.FirstJudge()
        return real(spec, seed)

    monkeypatch.setattr(R, "instantiate", fake_instantiate)
    cfg = _finetuned_cfg(tmp_path, data_root, ckpt)
    assert R.main(["--config", str(cfg), "--output", str(tmp_path / "run")]) == 0
    assert seen["params"]["model_path"] == str(ckpt.resolve())


def test_laya_judge_model_path_validation():
    from laya_slots.judges import LayaJudge
    assert LayaJudge(model_path="/x").name == "laya-single-multilingual-ft"
    with pytest.raises(ValueError, match="model_path"):
        LayaJudge(model_path="/x", checkpoint="lang")


def test_prepare_mix_caps_expands_and_builds_both_dev_kinds(data_root, tmp_path):
    import shutil
    shutil.copytree(data_root / "multiatis" / "en", data_root / "multiatis" / "fr")
    reels = tmp_path / "reels.yaml"
    reels.write_text("candidates:\n  generators:\n    - class: laya_slots.candidates.lexicon.LexiconGenerator\n"
                     "      params: {}\n  fit: true\n  max_candidates: 20\n", encoding="utf-8")
    cfg = _cfg(data_root, dev_candidates="both", reels_config=str(reels), unexpressed_ratio=1.0, mix=[
        {"dataset": "multiatis", "lang": "*", "exclude": ["fr"], "max_train": 10, "dev_limit": 6},
        {"dataset": "multiatis", "lang": "fr", "max_train": 5, "dev": False},
    ])
    assert [(e["dataset"], e["lang"]) for e in T.mix_entries(cfg)] == [("multiatis", "en"), ("multiatis", "fr")]
    tr, dv, stats = T.prepare(cfg)
    by = stats["train"]["by_entry"]
    assert by["multiatis-en"]["train"]["examples"] == 10 and by["multiatis-fr"]["train"]["examples"] == 5
    assert set(by["multiatis-en"]) == {"train", "dev_oracle", "dev_reels"} and set(by["multiatis-fr"]) == {"train"}
    assert by["multiatis-en"]["dev_oracle"]["examples"] == 6 and len(tr) == stats["train"]["items"] and dv
    a, _, _ = T.prepare(cfg)                                  # tirage déterministe (graine)
    assert [it.meta["example_id"] for it in a] == [it.meta["example_id"] for it in tr]
