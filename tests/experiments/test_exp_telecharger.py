"""experiments.telecharger_modeles : choix des LLM et des fichiers, sans réseau."""
from experiments.telecharger_modeles import MISTRAL_FORMAT, default_llms, ignore_patterns


def test_default_llms_reads_all_variables_in_order_without_duplicates():
    env = {"LLM_MODEL": "Qwen/Qwen3.8-Flash-Next-FP8", "LLM_MODEL_DEV": "Qwen/Qwen3.8-27B",
           "LLM_MODEL_ALT": "mistralai/Mistral-Small-4-119B-2603", "LLM_MODEL_ALT2": "google/gemma-4-31B-it"}
    assert default_llms(env) == list(env.values())
    assert default_llms({"LLM_MODEL": "a/b", "LLM_MODEL_DEV": "", "LLM_MODEL_ALT": "a/b"}) == ["a/b"]
    assert default_llms({"LLM_MODEL_ALT2": " g/x "}) == ["g/x"]
    assert default_llms({}) == []


def test_ignore_patterns_dual_format_mistral_repo():
    dual = ["config.json", "params.json", "tekken.json", "consolidated.safetensors.index.json",
            "consolidated-00001-of-00007.safetensors", "model-00001-of-00003.safetensors",
            "model.safetensors.index.json"]
    assert ignore_patterns(dual) == MISTRAL_FORMAT
    # format Mistral seul : on garde tout ; dépôts HF ordinaires : rien d'ignoré
    assert ignore_patterns(["params.json", "consolidated.safetensors"]) is None
    assert ignore_patterns(["config.json", "model-00001-of-00002.safetensors"]) is None
    assert ignore_patterns(["sub/model.safetensors", "README.md"]) is None
