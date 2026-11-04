from experiments.train_tagger import align_labels, label_list, span_f1, tag_examples
from laya_slots.schema import Example, SlotValue


def _ex(utt, slots):
    return Example(id=f"multiatis:en:train:{utt}", dataset="multiatis", lang="en", split="train",
                   utterance=utt, intent="atis_flight", slots=slots)


def test_tag_examples_et_etiquettes():
    seqs, st = tag_examples([_ex("to boston", [SlotValue("toloc", "boston", "boston", (3, 9))]),
                             _ex("from new york", [SlotValue("fromloc", "new york", "new york", (5, 13))])])
    assert seqs[1] == (["from", "new", "york"], ["O", "B-fromloc", "I-fromloc"])
    assert st["spans"] == 2 and st["examples"] == 2
    assert label_list(seqs) == ["O", "B-fromloc", "I-fromloc", "B-toloc", "I-toloc"]


def test_align_labels_premier_sous_mot():
    # [CLS] new ##york york [SEP] : mots 0, 0, 1
    assert align_labels([None, 0, 0, 1, None], [3, 4]) == [-100, 3, -100, 4, -100]


def test_span_f1_exact():
    gold = [["B-a", "I-a", "O"], ["B-b"]]
    pred = [["B-a", "O", "O"], ["B-b"]]
    m = span_f1(gold, pred)
    assert (m["n_gold"], m["n_pred"]) == (2, 2)
    assert m["precision"] == m["recall"] == m["f1"] == 0.5
    assert span_f1([["O"]], [["O"]])["f1"] == 0.0
