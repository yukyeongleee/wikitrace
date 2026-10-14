"""Every call records the provider that served it, on the chat path and the Pi path."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from pydantic import BaseModel

from wikitrace.config import ModelSpec, ModelsConfig
from wikitrace.config import EventLogger, provider_counts
from wikitrace.config import LLMCallRecord
from wikitrace.run.evaluate import _record_traverse_cost
from wikitrace.harness.llm import openrouter
from wikitrace.harness.llm.client import LLMClient
from wikitrace.harness.llm.openrouter import ChatResponse, OpenRouterClient


class _Out(BaseModel):
    x: int


class _Chat:
    def __init__(self, content: str) -> None:
        self.content = content

    def chat(self, **kw):
        return ChatResponse(content=self.content, raw={}, generation_id="g", model_returned="m",
                            finish_reason="stop", provider="Baidu")


def test_a_record_without_a_provider_still_parses():
    assert LLMCallRecord.model_validate({"call_id": "c"}).provider == ""


def test_the_chat_transport_reads_the_provider_off_the_completion(monkeypatch):
    raw = {"id": "gen", "model": "m", "provider": "Alibaba",
           "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}
    monkeypatch.setattr(httpx.Client, "send",
                        lambda self, request, **kw: httpx.Response(200, json=raw, request=request))
    assert OpenRouterClient(api_key="k").chat(model="m", messages=[]).provider == "Alibaba"


def test_the_generation_lookup_retries_until_the_metadata_lands(monkeypatch):
    answers = iter([httpx.Response(404), httpx.Response(200, json={"data": {"provider_name": "Baidu"}})])
    seen: list[str] = []

    def send(self, request, **kw):
        seen.append(str(request.url))
        response = next(answers)
        response.request = request
        return response

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(openrouter.time, "sleep", lambda s: None)
    assert OpenRouterClient(api_key="k").generation_provider("gen-1") == "Baidu"
    assert len(seen) == 2 and seen[0].endswith("/generation?id=gen-1")


def test_the_generation_lookup_raises_after_its_last_attempt(monkeypatch):
    monkeypatch.setattr(httpx.Client, "send",
                        lambda self, request, **kw: httpx.Response(404, request=request))
    monkeypatch.setattr(openrouter.time, "sleep", lambda s: None)
    with pytest.raises(httpx.HTTPStatusError):
        OpenRouterClient(api_key="k").generation_provider("gen-1", attempts=2)


def test_llm_client_records_carry_the_provider():
    records: list[LLMCallRecord] = []
    models = ModelsConfig(default=ModelSpec(model="m"))
    LLMClient(models, _Chat('{"x": 1}'), on_call=records.append).generate_structured(
        "solver", _Out, system="s", user="u",
    )
    assert [r.provider for r in records] == ["Baidu"]


def _traverse(tmp_path, transport):
    records: list[LLMCallRecord] = []
    llm = SimpleNamespace(on_call=records.append, transport=transport)
    events = EventLogger(tmp_path / "events.jsonl")
    log = SimpleNamespace(
        phase="solver", usage={}, stopped_by="", truncated_calls=0, turns=[],
        usage_calls=[{"generation_id": "g1", "prompt_tokens": 1, "completion_tokens": 1},
                     {"generation_id": "g2", "prompt_tokens": 1, "completion_tokens": 1}],
    )
    _record_traverse_cost(SimpleNamespace(phase_logs=[log]), llm, events, sample_id="s",
                          phase="test", step_n=0, model_requested="m")
    events.close()
    names = [json.loads(line)["event"] for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    return [r.provider for r in records], names


def test_pi_calls_take_the_looked_up_provider_and_a_failed_lookup_is_an_event(tmp_path):
    def lookup(gid: str) -> str:
        if gid == "g2":
            raise httpx.ConnectError("down")
        return "Baidu"

    providers, names = _traverse(tmp_path, SimpleNamespace(generation_provider=lookup))
    assert providers == ["Baidu", ""]
    assert names == ["provider_lookup_failed"]


def test_a_transport_without_the_lookup_leaves_the_provider_empty(tmp_path):
    providers, names = _traverse(tmp_path, object())
    assert providers == ["", ""] and names == []


def test_provider_counts_sum_every_log_and_name_the_unknown(tmp_path):
    a, b = tmp_path / "llm_calls.jsonl", tmp_path / "missing.jsonl"
    a.write_text("\n".join(json.dumps(r) for r in (
        {"provider": "Baidu"}, {"provider": "Alibaba"}, {"provider": "Baidu"}, {"provider": ""},
    )) + "\n")
    assert provider_counts(a, b) == {"(unknown)": 1, "Alibaba": 1, "Baidu": 2}
