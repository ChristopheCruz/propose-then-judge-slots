"""LLM-juge : prompt fixe et versionné, options en lettres, réponse en un jeton.

La probabilité de chaque option est lue sur les logprobs du premier jeton généré (top-k), en
sommant les variantes d'un même jeton-lettre (« A », « ▁A », « Ġ A »…), puis renormalisée sur
les lettres proposées. Si aucune lettre n'est dans le top-k, la distribution est uniforme et
`raw["no_letter"]` vaut True.

Backends :
- `VLLMBackend` : vLLM hors ligne (GPU), batch complet en un appel `generate` ou `chat` ;
- `OpenAICompatBackend` : serveur compatible OpenAI (`vllm serve`, etc.), `/v1/completions`
  (prompts groupés) ou `/v1/chat/completions` (une requête par prompt), via urllib ;
- tout objet exposant `top_logprobs(prompts) -> list[dict[str, float]]` (faux backend des tests).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

from ..interfaces import Judge
from ..schema import JudgeRequest, Judgment
from ._common import (
    LANG_NAMES,
    MAX_LETTER_OPTIONS,
    candidate_text,
    clip,
    format_filled,
    format_history,
    judgment_from_probs,
    normalize,
    option_keys,
    order_candidates,
)

log = logging.getLogger(__name__)

PROMPT_VERSION = "slot-judge-v1"

SYSTEM_PROMPT = ("You are a slot-filling judge for a task-oriented dialogue system. "
                 "You answer with a single capital letter and nothing else.")

PROMPT_TEMPLATE = """\
Task: choose the value that the user's last message gives for the slot below.
Pick the candidate that the user expresses or clearly refers to in the last message (the history \
and the slots already filled are context only). If the last message gives no value for this slot, \
or none of the candidates is right, pick the "not specified" option.
The conversation may be in a language other than English; candidates are written as they appear.

Slot: {slot}
Slot description: {description}
Current intent: {intent}
Slots already filled: {filled}
Conversation history:
{history}

User's last message ({lang}): {utterance}

Candidates:
{options}

Answer with the letter of one candidate only."""

FEW_SHOT_ANSWER = "Answer: {letter}"
NONE_OPTION = "not specified (the last message gives no value for this slot)"


def prompt_hash() -> str:
    """Empreinte du prompt (version + gabarits), à écrire dans chaque run."""
    parts = (PROMPT_VERSION, SYSTEM_PROMPT, PROMPT_TEMPLATE, FEW_SHOT_ANSWER, NONE_OPTION)
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:12]


# Variables de gabarit qui coupent le raisonnement, par famille (motif cherché dans le nom du
# modèle, en minuscules ; première famille trouvée). Vérifié sur les gabarits du Hub (27/10/2026) :
# - Qwen3 / Qwen3.5 / Qwen3.8 : `enable_thinking: false` -> « <think>\n\n</think>\n\n » après
#   le tour assistant (Qwen3.8 : raisonnement actif par défaut, `reasoning_effort` ignoré quand
#   `enable_thinking` est faux) ;
# - Gemma 4 : `enable_thinking` vaut faux par défaut ; explicite pour ne pas en dépendre ;
# - Mistral Small 4 (et Magistral / Devstral) : `reasoning_effort: "none"` (défaut du gabarit HF ;
#   explicite car le tokenizer mistral_common de vLLM, s'il est choisi, ne transmet que cette
#   variable ; `enable_thinking` n'existe pas chez Mistral).
# Une variable inconnue du gabarit est écartée sans erreur par vLLM (gabarits HF) : le défaut
# {enable_thinking: false} est donc sans effet sur un modèle sans raisonnement.
THINKING_OFF: tuple[tuple[tuple[str, ...], dict], ...] = (
    (("mistral", "magistral", "devstral", "ministral"), {"reasoning_effort": "none"}),
    (("qwen", "gemma"), {"enable_thinking": False}),
)
THINKING_OFF_DEFAULT: dict = {"enable_thinking": False}


def thinking_off_kwargs(model: str | None) -> dict:
    """Variables du gabarit de chat qui désactivent le raisonnement pour `model` (id Hub ou chemin)."""
    name = (model or "").lower()
    for patterns, kwargs in THINKING_OFF:
        if any(p in name for p in patterns):
            return dict(kwargs)
    return dict(THINKING_OFF_DEFAULT)


# Réglages de chargement vLLM propres à une famille, ajoutés sauf si llm_kwargs les fixe déjà.
# Mistral : les dépôts récents publient les deux formats (params.json + consolidated*, et config.json
# + model-*). vLLM choisit le format Mistral dès qu'il voit params.json, alors que seuls les poids
# HF sont téléchargés (experiments/telecharger_modeles.py) : « No model architectures are specified ».
FAMILY_LLM_KWARGS: list[tuple[tuple[str, ...], dict]] = [
    (("mistral", "magistral", "devstral"), {"config_format": "hf", "load_format": "safetensors",
                                             "tokenizer_mode": "hf"}),
]


def family_llm_kwargs(model: str | None, llm_kwargs: dict) -> dict:
    """`llm_kwargs` complétés par les réglages de la famille de `model` (les valeurs données gagnent)."""
    name = (model or "").lower()
    for patterns, extra in FAMILY_LLM_KWARGS:
        if any(p in name for p in patterns):
            return {**extra, **llm_kwargs}
    return dict(llm_kwargs)


def resolve_chat_template_kwargs(value: dict | str | None, model: str | None) -> dict | None:
    """`chat_template_kwargs` du backend : dict tel quel, "auto" -> `thinking_off_kwargs(model)`."""
    if value is None or isinstance(value, dict):
        return dict(value) if value else None
    if isinstance(value, str) and value == "auto":
        return thinking_off_kwargs(model)
    if hasattr(value, "items"):                       # DictConfig / Mapping
        return dict(value.items()) or None
    raise ValueError(f"chat_template_kwargs : dict, \"auto\" ou None attendu, reçu {value!r}")


class LogprobBackend(Protocol):
    def top_logprobs(self, prompts: list[dict]) -> list[dict[str, float]]:
        """Pour chaque prompt {"system", "user", "few_shot"}, logprobs du premier jeton : {texte: lp}."""
        ...


# ---------------------------------------------------------------------------
# Rendu du prompt
# ---------------------------------------------------------------------------
def render_user_prompt(slot: str, description: str, utterance: str, lang: str, options: list[str],
                       intent: str = "", filled: str = "", history: str = "") -> str:
    lines = "\n".join(f"{k}. {t}" for k, t in zip(option_keys(len(options)), options))
    return PROMPT_TEMPLATE.format(
        slot=slot, description=description or slot, intent=intent or "unknown",
        filled=filled or "none", history=history or "(none)", utterance=utterance,
        lang=LANG_NAMES.get(lang, lang), options=lines)


def as_completion_text(prompt: dict) -> str:
    """Prompt « texte brut » pour /v1/completions sans gabarit de chat."""
    parts = [prompt["system"], ""]
    for shot in prompt.get("few_shot", []):
        parts += [shot["user"], FEW_SHOT_ANSWER.format(letter=shot["assistant"]), ""]
    parts += [prompt["user"], "Answer:"]
    return "\n".join(parts)


def as_messages(prompt: dict) -> list[dict]:
    msgs = [{"role": "system", "content": prompt["system"]}]
    for shot in prompt.get("few_shot", []):
        msgs += [{"role": "user", "content": shot["user"]},
                 {"role": "assistant", "content": shot["assistant"]}]
    msgs.append({"role": "user", "content": prompt["user"]})
    return msgs


def letter_probs(top: dict[str, float], k: int) -> tuple[list[float], bool]:
    """Probabilités des k premières lettres à partir du top-k du premier jeton."""
    keys = option_keys(k)
    mass = dict.fromkeys(keys, 0.0)
    for tok, lp in top.items():
        t = tok.replace("▁", " ").replace("Ġ", " ").strip().rstrip(".):")
        if t in mass and lp is not None and lp > -math.inf:
            mass[t] += math.exp(lp)
    found = any(v > 0 for v in mass.values())
    return normalize([mass[key] for key in keys]), not found


# ---------------------------------------------------------------------------
# Juge
# ---------------------------------------------------------------------------
class LLMJudge(Judge):
    """LLM-juge open-weight (condition C4).

    backend : "vllm" (hors ligne), "openai" (serveur compatible) ou un objet LogprobBackend.
    model : nom ou chemin du modèle (vLLM) / nom servi (serveur).
    max_options : nombre maximal de candidats par question (hors « non précisé »), ≤ 25 ;
        au-delà, troncature (meilleurs scores générateur s'il y en a, sinon les premiers).
    candidate_order : "shuffle" (défaut, graine), "given" ou "score".
    few_shot : liste d'exemples {"slot", "description", "utterance", "lang", "candidates": [...],
        "answer": index ou None, "intent"?, "history"?} rendus avec le même gabarit.
    history_turns, history_max_chars, utterance_max_chars, option_max_chars : troncatures.
    backend_kwargs : options du backend (voir VLLMBackend, OpenAICompatBackend).
    """
    name = "llm"

    def __init__(self, backend: Any = "vllm", model: str | None = None, max_options: int = 25,
                 candidate_order: str = "shuffle", seed: int = 0, few_shot: list[dict] | None = None,
                 history_turns: int = 4, history_max_chars: int = 1500,
                 utterance_max_chars: int = 1000, option_max_chars: int = 200,
                 backend_kwargs: dict | None = None):
        if not 1 <= max_options <= MAX_LETTER_OPTIONS:
            raise ValueError(f"max_options doit être entre 1 et {MAX_LETTER_OPTIONS}")
        self.max_options, self.candidate_order, self.seed = max_options, candidate_order, seed
        self.few_shot = [self._render_shot(s) for s in (few_shot or [])]
        self.history_turns, self.history_max_chars = history_turns, history_max_chars
        self.utterance_max_chars, self.option_max_chars = utterance_max_chars, option_max_chars
        self.model = model
        if isinstance(backend, str):
            kw = dict(backend_kwargs or {})
            if backend == "vllm":
                backend = VLLMBackend(model=model, **kw)
            elif backend == "openai":
                backend = OpenAICompatBackend(model=model, **kw)
            else:
                raise ValueError(f"backend inconnu : {backend!r}")
        self.backend = backend
        self.name = f"llm-{PROMPT_VERSION}-{'fewshot' if self.few_shot else 'zeroshot'}"

    def _render_shot(self, s: dict) -> dict:
        opts = list(s["candidates"]) + [NONE_OPTION]
        answer = len(opts) - 1 if s.get("answer") is None else int(s["answer"])
        user = render_user_prompt(s["slot"], s.get("description", ""), s["utterance"],
                                  s.get("lang", "en"), opts, s.get("intent", ""),
                                  s.get("filled", ""), s.get("history", ""))
        return {"user": user, "assistant": option_keys(len(opts))[answer]}

    def build_prompt(self, req: JudgeRequest) -> tuple[dict, list[int], int]:
        idx, dropped = order_candidates(req, self.candidate_order, self.seed, self.max_options)
        opts = [clip(candidate_text(req.candidates[i]), self.option_max_chars) for i in idx]
        opts.append(NONE_OPTION)
        user = render_user_prompt(
            req.slot.name, req.slot.description, clip(req.example.utterance, self.utterance_max_chars),
            req.example.lang, opts, req.intent or "", format_filled(req.filled),
            format_history(req, self.history_turns, self.history_max_chars))
        return {"system": SYSTEM_PROMPT, "user": user, "few_shot": self.few_shot}, idx, dropped

    def judge_batch(self, requests: list[JudgeRequest]) -> list[Judgment]:
        built = [self.build_prompt(r) for r in requests]
        todo = [i for i, r in enumerate(requests) if r.candidates]
        tops: dict[int, dict] = {}
        per_req = 0.0
        if todo:
            t0 = time.perf_counter()
            out = self.backend.top_logprobs([built[i][0] for i in todo])
            per_req = (time.perf_counter() - t0) * 1000.0 / len(todo)
            tops = dict(zip(todo, out))
        results = []
        for i, req in enumerate(requests):
            n = len(req.candidates)
            raw = {"judge": self.name, "prompt_version": PROMPT_VERSION, "prompt_hash": prompt_hash()}
            if n == 0:
                raw["skipped"] = "no_candidates"
                results.append(Judgment(choice=None, proba=1.0, probs=[1.0], latency_ms=0.0, raw=raw))
                continue
            _, idx, dropped = built[i]
            q, no_letter = letter_probs(tops[i], len(idx) + 1)
            probs = [0.0] * (n + 1)
            for pos, ci in enumerate(idx):
                probs[ci] = q[pos]
            probs[n] = q[-1]
            raw.update(order=idx, dropped=dropped, no_letter=no_letter,
                       top_logprobs={k: round(v, 4) for k, v in tops[i].items()})
            results.append(judgment_from_probs(probs, latency_ms=per_req, raw=raw))
        return results


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class VLLMBackend:
    """vLLM hors ligne. Chargé à la première requête.

    chat : applique le gabarit de chat du modèle (`LLM.chat`) ; sinon prompt texte brut.
    chat_template_kwargs : variables du gabarit de chat, ex. {"enable_thinking": False} ; "auto" :
        celles qui coupent le raisonnement pour la famille du modèle (voir `thinking_off_kwargs`).
    top_k_logprobs : taille du top-k lu (vLLM plafonne par défaut à 20).
    llm_kwargs : passés à `vllm.LLM` (tensor_parallel_size, dtype, max_model_len, seed…).
    """

    def __init__(self, model: str | None, chat: bool = True, top_k_logprobs: int = 20,
                 chat_template_kwargs: dict | str | None = None, llm_kwargs: dict | None = None):
        if not model:
            raise ValueError("VLLMBackend : préciser `model`")
        self.model, self.chat, self.top_k = model, chat, top_k_logprobs
        self.chat_template_kwargs = resolve_chat_template_kwargs(chat_template_kwargs, model)
        self.llm_kwargs = dict(llm_kwargs or {})
        self._llm = None

    def _get(self):
        if self._llm is None:
            # CUDA est déjà initialisé dans ce processus (graines torch) : le moteur vLLM doit
            # démarrer par spawn, sinon « Cannot re-initialize CUDA in forked subprocess ».
            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
            from vllm import LLM  # dépendance optionnelle : uv sync --extra llm
            kwargs = family_llm_kwargs(self.model, self.llm_kwargs)
            log.info("vLLM %s : llm_kwargs=%s chat_template_kwargs=%s", self.model, kwargs,
                     self.chat_template_kwargs)
            self._llm = LLM(model=self.model, **kwargs)
        return self._llm

    def top_logprobs(self, prompts: list[dict]) -> list[dict[str, float]]:
        from vllm import SamplingParams
        llm = self._get()
        params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=self.top_k)
        if self.chat:
            kw = {"chat_template_kwargs": self.chat_template_kwargs} if self.chat_template_kwargs else {}
            outs = llm.chat([as_messages(p) for p in prompts], params, use_tqdm=False, **kw)
        else:
            outs = llm.generate([as_completion_text(p) for p in prompts], params, use_tqdm=False)
        res = []
        for o in outs:
            first = o.outputs[0].logprobs[0] if o.outputs and o.outputs[0].logprobs else {}
            res.append({(lp.decoded_token or str(tid)): lp.logprob for tid, lp in first.items()})
        return res


class OpenAICompatBackend:
    """Serveur compatible OpenAI (ex. `vllm serve <modèle>`).

    endpoint : "completions" (prompt texte brut, prompts groupés par requête HTTP) ou "chat"
        (`/v1/chat/completions`, gabarit appliqué par le serveur, une requête par prompt).
    base_url : ex. "http://localhost:8000/v1". api_key : facultatif.
    extra_body : champs ajoutés à chaque requête (ex. {"chat_template_kwargs": {...}}).
    """

    def __init__(self, model: str | None, base_url: str = "http://localhost:8000/v1",
                 endpoint: str = "completions", api_key: str | None = None, top_k_logprobs: int = 20,
                 batch_size: int = 64, concurrency: int = 16, timeout: float = 120.0,
                 extra_body: dict | None = None):
        if endpoint not in ("completions", "chat"):
            raise ValueError(f"endpoint inconnu : {endpoint!r}")
        self.model, self.base_url, self.endpoint = model, base_url.rstrip("/"), endpoint
        self.api_key, self.top_k, self.batch_size = api_key, top_k_logprobs, batch_size
        self.concurrency, self.timeout = concurrency, timeout
        self.extra_body = dict(extra_body or {})

    def _post(self, path: str, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url + path, data=json.dumps(body).encode(),
                                     headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def top_logprobs(self, prompts: list[dict]) -> list[dict[str, float]]:
        if self.endpoint == "completions":
            out: list[dict[str, float]] = []
            for s in range(0, len(prompts), self.batch_size):
                part = prompts[s:s + self.batch_size]
                body = {"model": self.model, "prompt": [as_completion_text(p) for p in part],
                        "max_tokens": 1, "temperature": 0.0, "logprobs": self.top_k, **self.extra_body}
                choices = sorted(self._post("/completions", body)["choices"], key=lambda c: c["index"])
                out += [parse_completion_logprobs(c) for c in choices]
            return out
        with ThreadPoolExecutor(self.concurrency) as pool:
            return list(pool.map(self._chat_one, prompts))

    def _chat_one(self, prompt: dict) -> dict[str, float]:
        body = {"model": self.model, "messages": as_messages(prompt), "max_tokens": 1,
                "temperature": 0.0, "logprobs": True, "top_logprobs": self.top_k, **self.extra_body}
        return parse_chat_logprobs(self._post("/chat/completions", body)["choices"][0])


def parse_completion_logprobs(choice: dict) -> dict[str, float]:
    """Format /v1/completions : choices[i].logprobs.top_logprobs[0] = {jeton: logprob}."""
    lp = choice.get("logprobs") or {}
    tops = lp.get("top_logprobs") or [{}]
    return dict(tops[0] or {})


def parse_chat_logprobs(choice: dict) -> dict[str, float]:
    """Format /v1/chat/completions : choices[0].logprobs.content[0].top_logprobs = [{token, logprob}]."""
    content = (choice.get("logprobs") or {}).get("content") or [{}]
    return {t["token"]: t["logprob"] for t in content[0].get("top_logprobs", [])}
