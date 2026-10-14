"""Pi's per-session model override and the translation of its token vocabulary to ours."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wikitrace.harness.llm.openrouter import provider_routing
from wikitrace.harness.terminal.rpc import _rpc_roundtrip


# The only supported way to change a model's output budget: the CLI has no --max-tokens flag.
_PI_AGENT_DIR_VAR = "PI_CODING_AGENT_DIR"
_PI_AGENT_DIR_NAME = ".pi-agent"
# Pi clamps the request to 32000 tokens, so asking for more is silently capped.
PI_MAX_OUTPUT_TOKENS_CEILING = 32000
# OpenRouter models absent from the bundled Pi registry, where Pi would drop their modelOverrides.
_PI_UNLISTED_MODELS: dict[str, dict[str, Any]] = {
    "z-ai/glm-5.3-flash": {
        "contextWindow": 1048575,
        "cost": {"input": 0.15, "output": 0.5, "cacheRead": 0.03, "cacheWrite": 0},
    },
    "qwen/qwen3.8-flash": {
        "contextWindow": 1000000,
        "cost": {"input": 0.15, "output": 0.47, "cacheRead": 0.016, "cacheWrite": 0.2},
    },
}


def _write_pi_agent_dir(
    work: Path, model: str, max_output_tokens: int, providers: list[str] | None = None,
) -> Path:
    """Write ``<work>/.pi-agent/models.json`` carrying this model's output budget and routing."""
    # The bundled Pi ships a stale 4096 for deepseek-v4-flash, the whole completion budget REASONING INCLUDED.
    override: dict[str, Any] = {"maxTokens": max_output_tokens}
    if providers:
        override["compat"] = {"openRouterRouting": provider_routing(providers)}
    provider: dict[str, Any] = {"modelOverrides": {model: override}}
    if model in _PI_UNLISTED_MODELS:
        provider = {
            "baseUrl": "https://openrouter.ai/api/v1",
            "apiKey": "OPENROUTER_API_KEY",
            "api": "openai-completions",
            "models": [{
                "id": model, "reasoning": True, "input": ["text"],
                **_PI_UNLISTED_MODELS[model], "maxTokens": max_output_tokens,
            }],
        }
        if providers:
            provider["compat"] = {"openRouterRouting": provider_routing(providers)}
    d = work / _PI_AGENT_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    (d / "models.json").write_text(
        json.dumps({"providers": {"openrouter": provider}}, indent=2), encoding="utf-8",
    )
    return d


def _usage_fields(tokens: Any, cost: Any) -> dict[str, Any]:
    """Pi's token vocabulary -> ``LLMCallRecord``'s. Empty dict if unusable."""
    # Pi's ``input`` EXCLUDES cache traffic while ``cached_tokens`` is a SUBSET of ``prompt_tokens`` here, so sum the input side.
    if not isinstance(tokens, dict):
        return {}
    nums = {
        k: v for k, v in tokens.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    if not {"input", "output"} <= nums.keys():
        return {}
    out = {
        "prompt_tokens": nums["input"] + nums.get("cacheRead", 0) + nums.get("cacheWrite", 0),
        "completion_tokens": nums["output"],
        # A SUBSET of completion_tokens; without pi-patch 0003 Pi folds reasoning into `output`.
        "reasoning_tokens": nums.get("reasoning", 0),
        "cached_tokens": nums.get("cacheRead", 0),
        "cache_write_tokens": nums.get("cacheWrite", 0),
    }
    out["total_tokens"] = nums.get("total", nums.get("totalTokens")) or (
        out["prompt_tokens"] + out["completion_tokens"]
    )
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        out["cost"] = cost
    return out


def collect_call_usage(
    proc: Any, timeout: float = 30.0, buf: bytearray | None = None,
) -> list[dict[str, Any]]:
    """Per-REQUEST usage for a live Pi session: one entry per provider round-trip."""
    # ``get_messages`` would pull a whole transcript through the pipe; empty on anything unexpected, so the caller falls back to totals.
    data = _rpc_roundtrip(proc, "get_usage", timeout, buf)
    messages = data.get("calls") if isinstance(data, dict) else None
    if not isinstance(messages, list):
        return []
    calls: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        usage = msg.get("usage")
        cost = (usage or {}).get("cost") if isinstance(usage, dict) else None
        fields = _usage_fields(usage, (cost or {}).get("total") if isinstance(cost, dict) else None)
        if not fields:
            continue
        if not fields["prompt_tokens"] and not fields["completion_tokens"]:
            # A message with NO tokens is not a round-trip (Pi synthesizes one for a cut-short run); counting it inflates the call count.
            continue
        fields["model_returned"] = msg.get("model") or ""
        fields["generation_id"] = msg.get("responseId") or ""
        fields["finish_reason"] = msg.get("stopReason") or ""
        calls.append(fields)
    return calls


def _sum_usage(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll per-call usage up into one phase total (numeric fields only)."""
    total: dict[str, Any] = {}
    for call in calls:
        for key, value in call.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total[key] = total.get(key, 0) + value
    return total


def collect_session_stats(
    proc: Any, timeout: float = 15.0, buf: bytearray | None = None,
) -> dict[str, Any]:
    """The session's token/cost TOTALS, the fallback when per-call usage is unavailable."""
    data = _rpc_roundtrip(proc, "get_session_stats", timeout, buf)
    if not isinstance(data, dict):
        return {}
    return _usage_fields(data.get("tokens"), data.get("cost"))
