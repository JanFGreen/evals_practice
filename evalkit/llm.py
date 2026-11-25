"""All model calls go through here. EVAL_LLM_MODE: fake (default, no network) | cache | replay | live."""

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from evalkit import config

_client = None


class LLMError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(f"[{kind}] {message}")
        self.kind = kind


@dataclass
class LLMResult:
    text: str
    model: str
    usage: dict = field(default_factory=dict)
    stop_reason: str | None = None
    latency_ms: float = 0.0
    content: list[dict] = field(default_factory=list)
    cached: bool = False
    fake: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _get_client():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic()
    return _client


def _cache_key(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_path(key: str):
    return config.cache_dir() / key[:2] / f"{key}.json"


def complete(
    *,
    model: str,
    system: str,
    messages: list[dict],
    max_tokens: int = 4096,
    effort: str | None = None,
    json_schema: dict | None = None,
    tools: list[dict] | None = None,
    rep: int = 0,
    fake: Callable[[], str | LLMResult] | None = None,
) -> LLMResult:
    """`rep` is part of the cache key, so repetitions (reps) give separate samples."""
    mode = config.llm_mode()

    if mode == "fake":
        if fake is None:
            raise LLMError("fake_unavailable", "this call has no fake-mode implementation")
        out = fake()
        if isinstance(out, LLMResult):
            out.fake = True
            return out
        return LLMResult(text=out, model=f"fake:{model}", stop_reason="end_turn", fake=True,
                         content=[{"type": "text", "text": out}])

    output_config: dict = {}
    if effort:
        output_config["effort"] = effort
    if json_schema:
        output_config["format"] = {"type": "json_schema", "schema": json_schema}

    payload = {"model": model, "system": system, "messages": messages, "max_tokens": max_tokens,
               "output_config": output_config, "tools": tools, "rep": rep}
    key = _cache_key(payload)
    path = _cache_path(key)

    if mode in ("cache", "replay") and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        data["cached"] = True
        return LLMResult(**data)
    if mode == "replay":
        raise LLMError("replay_miss", f"no cached response for key {key[:12]}")

    result = _call_api(model, system, messages, max_tokens, output_config, tools)
    if mode == "cache":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def _call_api(model, system, messages, max_tokens, output_config, tools) -> LLMResult:
    import anthropic

    kwargs = {"model": model, "max_tokens": max_tokens, "system": system, "messages": messages}
    if output_config:
        kwargs["output_config"] = output_config
    if tools:
        kwargs["tools"] = tools

    started = time.perf_counter()
    try:
        response = _get_client().messages.create(**kwargs)
    except anthropic.AuthenticationError as e:
        raise LLMError("auth", "missing or invalid API key (set ANTHROPIC_API_KEY)") from e
    except anthropic.RateLimitError as e:
        raise LLMError("rate_limit", str(e)) from e
    except anthropic.APITimeoutError as e:
        raise LLMError("timeout", str(e)) from e
    except anthropic.APIConnectionError as e:
        raise LLMError("connection", str(e)) from e
    except anthropic.BadRequestError as e:
        raise LLMError("bad_request", e.message) from e
    except anthropic.APIStatusError as e:
        raise LLMError(f"api_{e.status_code}", e.message) from e
    latency_ms = (time.perf_counter() - started) * 1000

    usage = response.usage
    text = "".join(b.text for b in response.content if b.type == "text")
    return LLMResult(
        text=text,
        model=response.model,
        usage={
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
            "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
        },
        stop_reason=response.stop_reason,
        latency_ms=latency_ms,
        content=[b.to_dict(mode="json", exclude_none=True) for b in response.content],
    )


def complete_json(*, schema: dict, **kwargs) -> tuple[dict, LLMResult]:
    result = complete(json_schema=schema, **kwargs)
    if result.stop_reason == "max_tokens":
        raise LLMError("truncated", "JSON response truncated at max_tokens")
    try:
        return json.loads(result.text), result
    except json.JSONDecodeError as e:
        raise LLMError("bad_json", f"cannot parse JSON: {result.text[:200]}") from e
