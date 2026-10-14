"""Provider pinning reaches both LLM paths with one routing dict, and only when configured."""

from __future__ import annotations

import json

import httpx

from wikitrace.config import LLMConfig
from wikitrace.harness.llm.openrouter import OpenRouterClient
from wikitrace.harness.terminal.usage import _write_pi_agent_dir

RAW = {
    "id": "gen",
    "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "x"},
                 "finish_reason": "stop"}],
}


def _capture(monkeypatch) -> list[dict]:
    requests: list[dict] = []

    def send(self, request, **kwargs):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=RAW, request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    return requests


def test_llm_config_default_has_no_providers():
    assert LLMConfig().openrouter_providers == []


def test_chat_payload_carries_provider_only_when_configured(monkeypatch):
    requests = _capture(monkeypatch)
    OpenRouterClient(api_key="k", providers=["baidu"]).chat(model="m", messages=[])
    assert requests[0]["provider"] == {"order": ["baidu"], "only": ["baidu"]}


def test_chat_payload_has_no_provider_when_empty(monkeypatch):
    requests = _capture(monkeypatch)
    OpenRouterClient(api_key="k").chat(model="m", messages=[])
    OpenRouterClient(api_key="k", providers=[]).chat(model="m", messages=[])
    assert all("provider" not in r for r in requests)


def test_pi_models_json_adds_compat_only_when_configured(tmp_path):
    plain = _write_pi_agent_dir(tmp_path / "a", "m", 16384) / "models.json"
    empty = _write_pi_agent_dir(tmp_path / "b", "m", 16384, []) / "models.json"
    pinned = _write_pi_agent_dir(tmp_path / "c", "m", 16384, ["baidu"]) / "models.json"
    assert plain.read_bytes() == empty.read_bytes()
    assert json.loads(plain.read_text()) == {
        "providers": {"openrouter": {"modelOverrides": {"m": {"maxTokens": 16384}}}}
    }
    assert json.loads(pinned.read_text()) == {
        "providers": {"openrouter": {"modelOverrides": {"m": {
            "maxTokens": 16384,
            "compat": {"openRouterRouting": {"order": ["baidu"], "only": ["baidu"]}},
        }}}}
    }


REGISTRY_MODELS_JSON = """{
  "providers": {
    "openrouter": {
      "modelOverrides": {
        "%s": {
          "maxTokens": 16384,
          "compat": {
            "openRouterRouting": {
              "order": [
                "alibaba"
              ],
              "only": [
                "alibaba"
              ]
            }
          }
        }
      }
    }
  }
}"""


def test_pi_models_json_for_registry_models_is_unchanged(tmp_path):
    for model in ("deepseek/deepseek-v4-flash", "openai/gpt-5.4-mini"):
        d = _write_pi_agent_dir(tmp_path / model.replace("/", "_"), model, 16384, ["alibaba"])
        assert (d / "models.json").read_text() == REGISTRY_MODELS_JSON % model


def test_pi_models_json_defines_unlisted_models_with_routing_and_context(tmp_path):
    for model, context in (("z-ai/glm-5.3-flash", 1048575), ("qwen/qwen3.8-flash", 1000000)):
        d = _write_pi_agent_dir(tmp_path / model.replace("/", "_"), model, 16384, ["together"])
        provider = json.loads((d / "models.json").read_text())["providers"]["openrouter"]
        routing = {"order": ["together"], "only": ["together"]}
        assert provider["compat"] == {"openRouterRouting": routing}
        assert (provider["apiKey"], provider["api"]) == ("OPENROUTER_API_KEY", "openai-completions")
        [entry] = provider["models"]
        assert (entry["id"], entry["contextWindow"], entry["maxTokens"]) == (model, context, 16384)
        assert entry["reasoning"] is True
