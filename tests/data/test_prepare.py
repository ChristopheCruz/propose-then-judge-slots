import json
from pathlib import Path

import pytest

from laya_slots.data import prepare as prep
from laya_slots.data.common import sha256_file
from laya_slots.schema import Ontology, read_examples

FIX = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("dataset,lang,counts", [
    ("multiatis", "en", {"train": 3, "dev": 1, "test": 1}),
    ("mtop", "fr", {"train": 3, "dev": 1, "test": 1}),
    ("multi3woz", "fr", {"train": 3, "dev": 2, "test": 1}),
])
def test_prepare_writes_files_and_manifest(tmp_path, dataset, lang, counts):
    rc = prep.main(["--dataset", dataset, "--lang", lang, "--raw-dir", str(FIX / dataset),
                    "--out-dir", str(tmp_path)])
    assert rc == 0
    out = tmp_path / dataset / lang
    manifest = json.loads((out / "MANIFEST.json").read_text())
    assert manifest["dataset"] == dataset and manifest["lang"] == lang
    assert manifest["source_files"] and manifest["source_version"]
    for split, n in counts.items():
        path = out / f"{split}.jsonl"
        exs = list(read_examples(path))
        assert len(exs) == n == manifest["files"][path.name]["examples"]
        assert manifest["files"][path.name]["sha256"] == sha256_file(path)
        assert all(e.split == split and e.lang == lang and e.dataset == dataset for e in exs)
        assert manifest["stats"][split]["examples"] == n
    onto = Ontology.from_dict(json.loads((out / "ontology.json").read_text()))
    assert onto.dataset == dataset and onto.slots


def test_prepare_is_deterministic(tmp_path):
    for sub in ("a", "b"):
        prep.prepare("mtop", "en", FIX / "mtop", tmp_path / sub)
    a = (tmp_path / "a" / "mtop" / "en" / "MANIFEST.json").read_text()
    b = (tmp_path / "b" / "mtop" / "en" / "MANIFEST.json").read_text()
    assert a == b


def test_stats():
    from laya_slots.data import MultiAtisLoader
    exs = list(MultiAtisLoader(FIX / "multiatis").load("en", "train"))
    s = prep.split_stats(exs)
    assert s["examples"] == 3 and s["multi_intent_examples"] == 1
    assert s["slots"] == 11 and s["slots_per_example"] == round(11 / 3, 3)


def test_missing_raw_data_returns_error(tmp_path, capsys):
    rc = prep.main(["--dataset", "multiatis", "--lang", "en", "--raw-dir", str(tmp_path / "none"),
                    "--out-dir", str(tmp_path)])
    assert rc == 2
    assert "introuvables" in capsys.readouterr().err
