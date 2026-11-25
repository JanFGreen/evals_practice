"""promptfoo custom provider: calls the real app entry point instead of re-implementing the prompt."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragapp.app import answer  # noqa: E402


def call_api(prompt: str, options: dict, context: dict) -> dict:
    config = options.get("config", {})
    test_vars = context.get("vars", {})
    r = answer(test_vars.get("question", prompt), test_vars.get("workspace", "nordwind"),
               prompt_version=config.get("prompt_version", "v2"), k=int(config.get("k", 4)))
    usage = r.usage or {}
    return {
        "output": r.answer,
        "tokenUsage": {"prompt": usage.get("input_tokens", 0), "completion": usage.get("output_tokens", 0),
                       "total": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)},
        "cost": r.cost_usd,
        "cached": r.cached,
        "metadata": {"retrieved_doc_ids": r.retrieved_doc_ids, "citations": r.citations, "model": r.model},
    }
