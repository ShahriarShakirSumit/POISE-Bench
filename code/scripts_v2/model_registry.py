# The list of evaluated models, shared by every script.
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ["qwen2.5-7b", "qwen3-8b", "llama3.1-8b", "olmo2-7b", "mistral-7b", "qwen3-14b", "phi4-14b", "qwen3-32b"]
NEW = ["phi4-mini", "gemma4-e4b", "gemma4-12b", "mistral-24b", "gemma4-26b-a4b", "gemma4-31b"]
NAME = {"qwen2.5-7b": "Qwen2.5-7B", "qwen3-8b": "Qwen3-8B", "llama3.1-8b": "Llama-3.1-8B", "olmo2-7b": "OLMo-2-7B",
        "mistral-7b": "Mistral-7B", "qwen3-14b": "Qwen3-14B", "phi4-14b": "Phi-4", "qwen3-32b": "Qwen3-32B",
        "phi4-mini": "Phi-4-mini", "gemma4-e4b": "Gemma-4-E4B", "gemma4-12b": "Gemma-4-12B",
        "mistral-24b": "Mistral-Small-24B", "gemma4-26b-a4b": "Gemma-4-26B-A4B", "gemma4-31b": "Gemma-4-31B"}


def enabled_new():
    f = ROOT / "results" / "new_models_enabled.json"
    return json.load(open(f)) if f.exists() else {}


def model_order():
    en = enabled_new()
    return ORIGINAL + [t for t in NEW if t in en]
