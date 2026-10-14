"""Preserve final HTTP requests and provider responses at the transport boundary."""

from copy import deepcopy
import json
from pathlib import Path

import httpx
import pytest

from wikitrace.harness.llm.openrouter import OpenRouterClient, OpenRouterError
from wikitrace.harness.llm.usage import parse_usage

REQUESTS = json.loads(
    (Path(__file__).parents[1] / "fixtures/prompts/model_requests.json").read_text()
)
MODEL = "deepseek/deepseek-v4-flash"
RAW = {
    "id": "fixture-generation",
    "model": MODEL,
    "provider": "fixture-provider",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": '{"ok":true}', "reasoning": "trace"},
        "finish_reason": "stop",
    }],
    "usage": {
        "prompt_tokens": 19,
        "completion_tokens": 7,
        "prompt_tokens_details": {"cached_tokens": 5},
        "completion_tokens_details": {"reasoning_tokens": 3},
        "cost": 0.002,
    },
}


def capture_http(monkeypatch, *, raw=RAW, status=200):
    requests = []

    def send(self, request, **kwargs):
        assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
        assert request.method == "POST"
        assert request.headers["Authorization"] == "Bearer fixture-key"
        requests.append(json.loads(request.content))
        return httpx.Response(status, json=deepcopy(raw), request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    return requests


@pytest.mark.parametrize("call", REQUESTS)
def test_frozen_requests_reach_http_unchanged(monkeypatch, call):
    requests = capture_http(monkeypatch)
    result = OpenRouterClient(api_key="fixture-key").chat(**deepcopy(call))
    expected = {key: deepcopy(value) for key, value in call.items() if key != "reasoning_effort"}
    expected["usage"] = {"include": True}
    effort = call.get("reasoning_effort")
    if effort:
        expected["reasoning"] = (
            {"enabled": False} if effort.lower() in ("none", "off") else {"effort": effort}
        )
    assert requests == [expected]
    assert result.raw == RAW
    assert result.reasoning == "trace"
    assert result.model_returned == MODEL
    assert result.generation_id == "fixture-generation"
    usage = parse_usage(result.raw)
    assert (usage.prompt_tokens, usage.completion_tokens, usage.cached_tokens,
            usage.reasoning_tokens, usage.cost) == (19, 7, 5, 3, 0.002)


@pytest.mark.parametrize("effort,reasoning", [
    (None, None), ("off", {"enabled": False}), ("NONE", {"enabled": False}),
    ("high", {"effort": "high"}),
])
def test_april_model_and_reasoning_reach_http(monkeypatch, effort, reasoning):
    requests = capture_http(monkeypatch)
    OpenRouterClient(api_key="fixture-key").chat(
        model=MODEL, messages=[{"role": "user", "content": "Fixed question.\n"}],
        reasoning_effort=effort,
    )
    assert requests[0] == {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Fixed question.\n"}],
        "temperature": 0.0, "max_tokens": 8192, "usage": {"include": True},
        **({"reasoning": reasoning} if reasoning is not None else {}),
    }


@pytest.mark.parametrize("status,raw,error", [
    (429, {"error": "rate limited"}, httpx.HTTPStatusError),
    (200, {"choices": []}, OpenRouterError),
])
def test_transport_does_not_retry_or_substitute(monkeypatch, status, raw, error):
    requests = capture_http(monkeypatch, raw=raw, status=status)
    with pytest.raises(error):
        OpenRouterClient(api_key="fixture-key").chat(model=MODEL, messages=[])
    assert len(requests) == 1
    assert requests[0]["model"] == MODEL


def test_missing_provider_cost_is_not_estimated(monkeypatch):
    raw = deepcopy(RAW)
    del raw["usage"]["cost"]
    capture_http(monkeypatch, raw=raw)
    result = OpenRouterClient(api_key="fixture-key").chat(model=MODEL, messages=[])
    assert "cost" not in result.raw["usage"]
    assert parse_usage(result.raw).cost == 0.0
