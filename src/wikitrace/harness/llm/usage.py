"""Token/cost extraction from OpenRouter responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0


def parse_usage(response_json: dict[str, Any]) -> Usage:
    """Extract usage from an OpenRouter chat-completion response."""
    # `cost` arrives only when `usage: {include: true}` is requested.
    usage = response_json.get("usage") or {}
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    cost = float(usage.get("cost", 0.0) or 0.0)

    completion_details = usage.get("completion_tokens_details") or {}
    reasoning = int(completion_details.get("reasoning_tokens", 0) or 0)

    prompt_details = usage.get("prompt_tokens_details") or {}
    cached = int(prompt_details.get("cached_tokens", 0) or 0)

    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        reasoning_tokens=reasoning,
        cached_tokens=cached,
        cost=cost,
    )
