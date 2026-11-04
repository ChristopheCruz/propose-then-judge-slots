import pytest

from laya_slots.data.common import bio_to_spans, byte_to_char_offsets, humanize, token_offsets


def test_bio_to_spans():
    labels = ["O", "B-a", "I-a", "B-a", "I-b", "O", "I-c", "I-c"]
    assert bio_to_spans(labels) == [("a", 1, 3), ("a", 3, 4), ("b", 4, 5), ("c", 6, 8)]
    assert bio_to_spans([]) == []
    assert bio_to_spans(["B-x"]) == [("x", 0, 1)]


def test_token_offsets_handles_extra_spaces():
    text = "flights to san  francisco"
    offs = token_offsets(text, text.split())
    assert [text[a:b] for a, b in offs] == ["flights", "to", "san", "francisco"]
    with pytest.raises(ValueError):
        token_offsets("abc", ["x"])


def test_byte_to_char_offsets():
    text = "météo à Orléans"
    b0 = len("météo à ".encode())
    b1 = b0 + len("Orléans".encode())
    s, e = byte_to_char_offsets(text, b0, b1)
    assert text[s:e] == "Orléans"


def test_humanize():
    assert humanize("SL:DATE_TIME") == "date time"
    assert humanize("fromloc.city_name") == "fromloc city name"
