"""Vérifications sur les vraies données (réseau, téléchargement) : `uv run pytest -m integration`.

Les effectifs attendus sont ceux publiés (README MTOP, article Multi3WOZ). MultiATIS++ (LDC) n'est
testé que si les fichiers sont déjà présents sous data/raw/multiatis.
"""
from pathlib import Path

import pytest

from laya_slots.data import MtopLoader, Multi3WozLoader, MultiAtisLoader

pytestmark = pytest.mark.integration
RAW = Path("data/raw")


@pytest.mark.parametrize("lang,expected", [("en", (15667, 2235, 4386)), ("fr", (11814, 1577, 3193))])
def test_mtop_counts(lang, expected):
    loader = MtopLoader(RAW / "mtop")
    if not (RAW / "mtop" / "mtop.zip").exists():
        loader.download()
    got = tuple(sum(1 for _ in loader.load(lang, s)) for s in ("train", "dev", "test"))
    assert got == expected


@pytest.mark.parametrize("lang", ["en", "fr"])
def test_multi3woz_dialogue_counts(lang):
    loader = Multi3WozLoader(RAW / "multi3woz")
    if not (RAW / "multi3woz" / "data.zip").exists():
        loader.download()
    got = tuple(len({e.meta["dialogue_id"] for e in loader.load(lang, s)}) for s in ("train", "dev", "test"))
    assert got == (7440, 860, 860)


@pytest.mark.parametrize("lang", ["en", "fr"])
def test_multiatis_counts_if_present(lang):
    loader = MultiAtisLoader(RAW / "multiatis")
    try:
        got = tuple(sum(1 for _ in loader.load(lang, s)) for s in ("train", "dev", "test"))
    except FileNotFoundError:
        pytest.skip("MultiATIS++ absent (licence LDC2021T04)")
    # Copie LDC : dev prélevé dans le train (500) ; 13 lignes du train FR mal annotées sont ignorées.
    attendu = {"en": (4478, 500, 893), "fr": (4465, 500, 893)}[lang]
    assert got in ((4488, 490, 893), attendu)
