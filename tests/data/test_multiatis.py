from pathlib import Path

import pytest

from laya_slots.data import MultiAtisLoader, get_loader
from laya_slots.data.multiatis import slot_description

FIX = Path(__file__).parent / "fixtures" / "multiatis"


@pytest.fixture
def loader():
    return MultiAtisLoader(raw_dir=FIX)


def test_registry():
    assert isinstance(get_loader("multiatis", FIX), MultiAtisLoader)
    with pytest.raises(ValueError):
        get_loader("atis")


def test_load_en_spans_and_values(loader):
    exs = list(loader.load("en", "train"))
    assert len(exs) == 3 and loader.skipped[("en", "train")] == 1      # ligne 4 : étiquettes incohérentes
    ex = exs[0]
    assert ex.id == "multiatis:en:train:1" and ex.intent == "atis_flight" and ex.domain == "flights"
    by_name = {s.name: s for s in ex.slots}
    assert by_name["toloc.city_name"].surface == "new york"
    s = by_name["toloc.city_name"]
    assert ex.utterance[s.span[0]:s.span[1]] == "new york"
    assert by_name["depart_date.today_relative"].value == "tomorrow"
    for s in ex.slots:
        assert ex.utterance[s.span[0]:s.span[1]] == s.surface and s.origin == "utterance"


def test_multi_intent(loader):
    ex = list(loader.load("en", "train"))[2]
    assert ex.intent == "atis_flight"
    assert ex.meta["intents"] == ["atis_flight", "atis_airfare"]
    assert ex.meta["intent_raw"] == "atis_flight#atis_airfare"
    time = next(s for s in ex.slots if s.name == "depart_time.time")
    assert time.surface == "five pm" and time.value == "17:00"


def test_orphan_inside_label_and_double_space(loader):
    (ex,) = loader.load("en", "test")
    city = next(s for s in ex.slots if s.name == "toloc.city_name")
    assert city.surface == "san  francisco" and city.value == "san francisco"


def test_load_fr(loader):
    exs = list(loader.load("fr", "train"))
    assert len(exs) == 3
    t = next(s for s in exs[2].slots if s.name == "depart_time.time")
    assert t.surface == "dix-sept heures" and t.value == "17:00"
    assert exs[0].id == "multiatis:fr:train:1"


def test_ontology(loader):
    onto = loader.ontology("en")
    assert onto.dataset == "multiatis" and onto.lang == "en"
    assert set(onto.intents) == {"atis_flight", "atis_airfare", "atis_airline"}    # atomiques seulement
    assert "fromloc.city_name" in onto.intents["atis_airfare"].slots
    assert onto.slots["fromloc.city_name"].description == "departure city"
    assert onto.slots["fromloc.city_name"].values is None
    # valeurs fermées relevées sur le train seulement (« friday » n'est qu'en test)
    assert onto.slots["depart_date.day_name"].values == ["monday"]
    fr = loader.ontology("fr")
    assert fr.slots["fromloc.city_name"].description == "ville (départ)"
    assert fr.intents["atis_flight"].description == "recherche de vols"


def test_slot_description_fallback():
    assert slot_description("depart_date.day_name", "en") == "departure date: day of the week"
    assert slot_description("unknown_thing", "en") == "unknown thing"


def test_missing_files_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(MultiAtisLoader(raw_dir=tmp_path).load("en", "train"))


def test_dev_preleve_si_absent(tmp_path):
    """Copie LDC sans dev : le dev est prélevé dans le train, mêmes identifiants en EN et FR."""
    import shutil

    from laya_slots.data import multiatis

    for f in FIX.glob("*.tsv"):
        if not f.name.startswith("dev_"):
            shutil.copy(f, tmp_path / f.name)
    n_train = len(list(MultiAtisLoader(raw_dir=FIX).load("en", "train")))
    size = max(1, n_train // 3)
    old, multiatis.DEV_SIZE = multiatis.DEV_SIZE, size
    try:
        loader = MultiAtisLoader(raw_dir=tmp_path)
        dev = {lang: [ex.meta["source_id"] for ex in loader.load(lang, "dev")] for lang in ("en", "fr")}
        train = [ex.meta["source_id"] for ex in loader.load("en", "train")]
    finally:
        multiatis.DEV_SIZE = old
    assert dev["en"] == dev["fr"] and len(dev["en"]) == size
    assert not set(dev["en"]) & set(train) and len(train) + size == n_train
    assert all(ex.split == "dev" and ":dev:" in ex.id for ex in MultiAtisLoader(raw_dir=tmp_path).load("en", "dev"))


def test_espace_insecable_dans_un_jeton(tmp_path):
    for split in ("train", "dev", "test"):
        (tmp_path / f"{split}_FR.tsv").write_text(
            "id\tutterance\tslot_labels\tintent\n"
            "1\tvol avant 7\xa0heures du matin\tO B-depart_time.time_relative B-depart_time.time "
            "I-depart_time.time I-depart_time.time\tatis_flight\n", encoding="utf-8")
    (ex,) = MultiAtisLoader(raw_dir=tmp_path, langs=["fr"]).load("fr", "test")
    assert [s.surface for s in ex.slots] == ["avant", "7\xa0heures du matin"]
