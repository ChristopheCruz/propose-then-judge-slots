import json
import math
import sys
import types
from typing import ClassVar

import pytest

from laya_slots.judges import LLMJudge, OpenAICompatBackend
from laya_slots.judges.llm_judge import (
    PROMPT_VERSION,
    as_completion_text,
    as_messages,
    letter_probs,
    parse_chat_logprobs,
    parse_completion_logprobs,
    prompt_hash,
    resolve_chat_template_kwargs,
    thinking_off_kwargs,
)

from .helpers import request


class FakeBackend:
    """Imite un top-k de logprobs : favorise la lettre de l'option qui apparaît dans l'énoncé."""

    def __init__(self, raw=None):
        self.raw = raw
        self.prompts = []

    def top_logprobs(self, prompts):
        self.prompts += prompts
        if self.raw is not None:
            return [dict(self.raw) for _ in prompts]
        out = []
        for p in prompts:
            user = p["user"]
            msg = user.split("User's last message")[1].split("\n")[0].lower()
            opts = [l for l in user.split("Candidates:\n")[1].split("\n\n")[0].split("\n")]
            pairs = [line.split(". ", 1) for line in opts]
            any_hit = any(t.lower() in msg for _, t in pairs[:-1])
            top = {}
            for letter, text in pairs:
                hit = text.lower() in msg or (text.startswith("not specified") and not any_hit)
                top[" " + letter] = math.log(0.8) if hit else math.log(0.02)
            top[" The"] = math.log(0.01)
            out.append(top)
        return out


def test_letter_probs_variants_and_renormalization():
    top = {" A": math.log(0.5), "A": math.log(0.1), "▁B": math.log(0.2), "C.": math.log(0.1),
           " D": math.log(0.05), "Hello": math.log(0.05)}
    probs, missing = letter_probs(top, 3)
    assert not missing
    assert probs == pytest.approx([0.6 / 0.9, 0.2 / 0.9, 0.1 / 0.9])
    probs, missing = letter_probs({"Hello": -0.1}, 2)
    assert missing and probs == [0.5, 0.5]


def test_llm_judge_with_fake_backend():
    backend = FakeBackend()
    j = LLMJudge(backend=backend, candidate_order="given")
    out = j.judge(request(utterance="i want to fly to denver"))
    assert out.choice == 1 and len(out.probs) == 4 and abs(sum(out.probs) - 1) < 1e-6
    assert out.raw["prompt_version"] == PROMPT_VERSION and out.raw["no_letter"] is False
    user = backend.prompts[0]["user"]
    assert "A. boston\nB. denver\nC. dallas\nD. not specified" in user
    assert "Slot: toloc.city_name" in user and "arrival city" in user


def test_french_utterance_in_english_prompt():
    backend = FakeBackend()
    out = LLMJudge(backend=backend, candidate_order="given").judge(
        request(values=("paris", "lyon"), utterance="je veux aller à lyon", lang="fr"))
    assert out.choice == 1
    assert "User's last message (French): je veux aller à lyon" in backend.prompts[0]["user"]


def test_none_and_empty():
    backend = FakeBackend()
    j = LLMJudge(backend=backend, candidate_order="given")
    assert j.judge(request(utterance="show me fares")).choice is None
    outs = j.judge_batch([request(values=()), request(utterance="to dallas please")])
    assert outs[0].choice is None and outs[0].probs == [1.0]
    assert outs[1].choice == 2 and len(backend.prompts) == 2       # pas de prompt pour la liste vide


def test_truncation_and_shuffle_mapping():
    backend = FakeBackend()
    vals = [f"city{i}" for i in range(30)] + ["denver"]
    scores = [0.0] * 30 + [1.0]
    out = LLMJudge(backend=backend, max_options=5, candidate_order="shuffle", seed=2).judge(
        request(values=vals, scores=scores, utterance="to denver"))
    assert out.choice == 30 and out.raw["dropped"] == 26 and len(out.probs) == 32


def test_few_shot_rendering():
    shot = {"slot": "date", "description": "travel date", "utterance": "tomorrow please",
            "candidates": ["2026-10-02", "2026-10-03"], "answer": 0}
    backend = FakeBackend()
    j = LLMJudge(backend=backend, few_shot=[shot, dict(shot, answer=None)])
    j.judge(request(utterance="to denver"))
    p = backend.prompts[0]
    assert [s["assistant"] for s in p["few_shot"]] == ["A", "C"]
    msgs = as_messages(p)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    text = as_completion_text(p)
    assert text.endswith("Answer:") and "Answer: A" in text
    assert j.name.endswith("fewshot")


def test_prompt_hash_stable():
    assert prompt_hash() == prompt_hash() and len(prompt_hash()) == 12


def test_openai_parsers():
    comp = {"index": 0, "logprobs": {"top_logprobs": [{" A": -0.1, " B": -2.5}]}}
    assert parse_completion_logprobs(comp) == {" A": -0.1, " B": -2.5}
    chat = {"logprobs": {"content": [{"token": "B", "logprob": -0.2,
                                      "top_logprobs": [{"token": "B", "logprob": -0.2},
                                                       {"token": "A", "logprob": -1.8}]}]}}
    assert parse_chat_logprobs(chat) == {"B": -0.2, "A": -1.8}


def test_openai_backend_completions_and_chat(monkeypatch):
    sent = []

    def fake_post(self, path, body):
        sent.append((path, body))
        if path == "/completions":
            n = len(body["prompt"])
            return {"choices": [{"index": i, "logprobs": {"top_logprobs": [{" B": -0.1, " A": -3.0}]}}
                                for i in reversed(range(n))]}
        return {"choices": [{"logprobs": {"content": [{"top_logprobs": [{"token": "A", "logprob": -0.1}]}]}}]}

    monkeypatch.setattr(OpenAICompatBackend, "_post", fake_post)
    j = LLMJudge(backend="openai", model="m", candidate_order="given",
                 backend_kwargs={"base_url": "http://x/v1", "batch_size": 2})
    outs = j.judge_batch([request(id=str(i)) for i in range(3)])
    assert [o.choice for o in outs] == [1, 1, 1]
    assert [len(b["prompt"]) for _, b in sent] == [2, 1]
    assert sent[0][1]["max_tokens"] == 1 and sent[0][1]["logprobs"] == 20
    json.dumps(sent[0][1])                                            # sérialisable
    sent.clear()
    j = LLMJudge(backend="openai", model="m", candidate_order="given",
                 backend_kwargs={"endpoint": "chat"})
    assert j.judge(request()).choice == 0
    assert sent[0][0] == "/chat/completions" and sent[0][1]["top_logprobs"] == 20


def test_vllm_backend_requires_model():
    with pytest.raises(ValueError):
        LLMJudge(backend="vllm")


# ---------------------------------------------------------------------------
# Gabarit de chat par famille (chat_template_kwargs: auto) ; faux module vllm, sans GPU
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model, expected", [
    ("Qwen/Qwen3-32B", {"enable_thinking": False}),
    ("Qwen/Qwen3.8-27B", {"enable_thinking": False}),
    ("Qwen/Qwen3.8-Flash-Next-FP8", {"enable_thinking": False}),
    ("google/gemma-4-31B-it", {"enable_thinking": False}),
    ("mistralai/Mistral-Small-4-119B-2603", {"reasoning_effort": "none"}),
    ("/home/ubuntu/hf-cache/hub/models--mistralai--Mistral-Small-4-119B-2603/snapshots/x",
     {"reasoning_effort": "none"}),
    ("meta-llama/Llama-4-Scout", {"enable_thinking": False}),          # défaut, sans effet
])
def test_thinking_off_kwargs_by_family(model, expected):
    assert thinking_off_kwargs(model) == expected


def test_resolve_chat_template_kwargs():
    assert resolve_chat_template_kwargs(None, "Qwen/Qwen3-32B") is None
    assert resolve_chat_template_kwargs({}, "Qwen/Qwen3-32B") is None
    explicit = {"enable_thinking": False, "x": 1}
    assert resolve_chat_template_kwargs(explicit, "mistralai/M") == explicit     # dict : inchangé
    assert resolve_chat_template_kwargs("auto", "mistralai/M") == {"reasoning_effort": "none"}
    with pytest.raises(ValueError):
        resolve_chat_template_kwargs("thinking", "Qwen/Qwen3-32B")


class _FakeLogprob:
    def __init__(self, token, logprob):
        self.decoded_token, self.logprob = token, logprob


class _FakeLLM:
    """Imite vllm.LLM : note les arguments de chat, répond « B » avec une forte probabilité."""
    instances: ClassVar[list] = []

    def __init__(self, model, **kwargs):
        self.model, self.kwargs, self.calls = model, kwargs, []
        _FakeLLM.instances.append(self)

    def chat(self, messages, params, use_tqdm=False, **kw):
        self.calls.append((messages, params, kw))
        first = {1: _FakeLogprob("A", math.log(0.1)), 2: _FakeLogprob("B", math.log(0.8)),
                 3: _FakeLogprob("C", math.log(0.05)), 4: _FakeLogprob("D", math.log(0.05))}
        out = types.SimpleNamespace(logprobs=[first])
        return [types.SimpleNamespace(outputs=[out]) for _ in messages]


@pytest.fixture
def fake_vllm(monkeypatch):
    _FakeLLM.instances.clear()
    mod = types.ModuleType("vllm")
    mod.LLM = _FakeLLM
    mod.SamplingParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vllm", mod)
    return _FakeLLM


@pytest.mark.parametrize("model, sent", [
    ("Qwen/Qwen3.8-27B", {"enable_thinking": False}),
    ("google/gemma-4-31B-it", {"enable_thinking": False}),
    ("mistralai/Mistral-Small-4-119B-2603", {"reasoning_effort": "none"}),
])
def test_vllm_backend_auto_kwargs_sent_to_chat(fake_vllm, model, sent):
    j = LLMJudge(backend="vllm", model=model, candidate_order="given",
                 backend_kwargs={"chat_template_kwargs": "auto",
                                 "llm_kwargs": {"tensor_parallel_size": 2, "language_model_only": True}})
    out = j.judge(request())
    assert out.choice == 1                                       # « B » le plus probable
    llm = fake_vllm.instances[0]
    expected = {"tensor_parallel_size": 2, "language_model_only": True}
    if "mistral" in model.lower():                                # format HF imposé (params.json ignoré)
        expected |= {"config_format": "hf", "load_format": "safetensors", "tokenizer_mode": "hf"}
    assert llm.model == model and llm.kwargs == expected
    messages, params, kw = llm.calls[0]
    assert kw == {"chat_template_kwargs": sent}
    assert params == {"max_tokens": 1, "temperature": 0.0, "logprobs": 20}
    assert messages[0][0]["role"] == "system"


def test_vllm_backend_explicit_and_absent_kwargs(fake_vllm):
    j = LLMJudge(backend="vllm", model="mistralai/Mistral-Small-4-119B-2603", candidate_order="given",
                 backend_kwargs={"chat_template_kwargs": {"enable_thinking": False}})
    j.judge(request())
    assert fake_vllm.instances[-1].calls[0][2] == {"chat_template_kwargs": {"enable_thinking": False}}
    j = LLMJudge(backend="vllm", model="Qwen/Qwen3-32B", candidate_order="given")
    j.judge(request())
    assert fake_vllm.instances[-1].calls[0][2] == {}             # rien envoyé par défaut


def test_c4_config_uses_auto_and_text_only():
    from pathlib import Path

    import yaml
    cfg = yaml.safe_load((Path(__file__).parents[2] / "configs/conditions/c4_llm_juge.yaml").read_text())
    bk = cfg["judge"]["params"]["backend_kwargs"]
    assert bk["chat_template_kwargs"] == "auto"
    assert bk["llm_kwargs"]["language_model_only"] is True



def test_family_llm_kwargs_mistral_forces_hf_format():
    from laya_slots.judges.llm_judge import family_llm_kwargs
    k = family_llm_kwargs("mistralai/Mistral-Small-4-119B-2603", {"tensor_parallel_size": 2})
    assert k == {"config_format": "hf", "load_format": "safetensors", "tokenizer_mode": "hf",
                 "tensor_parallel_size": 2}
    assert family_llm_kwargs("mistralai/X", {"tokenizer_mode": "mistral"})["tokenizer_mode"] == "mistral"
    assert family_llm_kwargs("Qwen/Qwen3.8-27B", {"max_model_len": 4096}) == {"max_model_len": 4096}
