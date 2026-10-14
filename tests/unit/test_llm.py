"""LLM client: catalog validation, structured parsing/retry, usage logging."""

from __future__ import annotations

import json

import pytest

from wikitrace.harness.llm import LLMClient, ModelNotInCatalogError, StructuredParseError
from wikitrace.harness.llm.openrouter import ChatResponse
from wikitrace.harness.llm.structured_output import extract_json, parse_into
from tests.fixtures.structured_model import SelectorOutput
from wikitrace.config import ModelSpec, ModelsConfig


def _models() -> ModelsConfig:
    return ModelsConfig(default=ModelSpec(model="fake/model"))


# ---------------------------------------------------------------------------
# Catalog validation
# ---------------------------------------------------------------------------


def test_validate_models_raises_on_unknown(fake_transport_factory):
    transport = fake_transport_factory(validate_ok=False)
    with pytest.raises(ModelNotInCatalogError):
        transport.validate_models(["unknown/model"])


def test_validate_models_passes(fake_transport_factory):
    transport = fake_transport_factory(validate_ok=True)
    result = transport.validate_models(["fake/model"])
    assert result["validated"] is True


# ---------------------------------------------------------------------------
# Structured output parsing
# ---------------------------------------------------------------------------


def test_extract_json_strips_fences_and_extracts_balanced_object():
    noisy = (
        "Sure, here is the answer:\n"
        "```json\n"
        '{"schema_selection": {}, "decision_summary": "x"}\n'
        "```\n"
        "hope that helps"
    )
    extracted = extract_json(noisy)
    payload = json.loads(extracted)
    assert payload["decision_summary"] == "x"


def test_parse_into_validates_model():
    content = '{"schema_selection": {"Patient": "keep_all", "Lab": ["PT"]}}'
    parsed = parse_into(SelectorOutput, content)
    assert isinstance(parsed, SelectorOutput)
    assert parsed.schema_selection == {"Patient": "keep_all", "Lab": ["PT"]}


def test_malformed_json_retries_then_raises():
    """A transport returning junk should trigger retries then StructuredParseError."""

    class JunkTransport:
        def __init__(self) -> None:
            self.n_calls = 0

        def chat(self, model, messages, temperature, max_tokens, response_format=None, reasoning_effort=None):
            self.n_calls += 1
            raw = {
                "id": "x",
                "model": model,
                "choices": [{"message": {"content": "not json at all"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
            }
            return ChatResponse(
                content="not json at all",
                raw=raw,
                generation_id="x",
                model_returned=model,
                finish_reason="stop",
            )

    transport = JunkTransport()
    client = LLMClient(_models(), transport, max_retries=1)
    with pytest.raises(StructuredParseError):
        client.generate_structured(
            role="selector",
            model_cls=SelectorOutput,
            system="sys",
            user="usr",
        )
    # max_retries=1 -> initial attempt + 1 retry == 2 chat calls.
    assert transport.n_calls == 2


def _chat_response(content: str, model: str, finish_reason: str = "stop") -> ChatResponse:
    raw = {
        "id": "x", "model": model,
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
    }
    return ChatResponse(
        content=content, raw=raw, generation_id="x",
        model_returned=model, finish_reason=finish_reason,
    )


class _EmptyTransport:
    """Returns empty/error responses for the first ``n_empty`` calls, then valid JSON."""

    def __init__(self, n_empty: int, finish_reason: str = "error", content: str = "") -> None:
        self.n_empty = n_empty
        self.finish_reason = finish_reason
        self.empty_content = content
        self.msg_lens: list[int] = []

    def chat(self, model, messages, temperature, max_tokens, response_format=None, reasoning_effort=None):
        self.msg_lens.append(len(messages))
        if len(self.msg_lens) <= self.n_empty:
            return _chat_response(self.empty_content, model, self.finish_reason)
        return _chat_response('{"schema_selection": {}, "decision_summary": "ok"}', model, "stop")


def test_empty_response_is_transient_and_recovers(monkeypatch):
    """An empty/error response is retried (with backoff) and recovers, without nudging."""
    import wikitrace.harness.llm.client as client_mod

    monkeypatch.setattr(client_mod.time, "sleep", lambda *_: None)  # no real backoff in tests
    transport = _EmptyTransport(n_empty=2, finish_reason="error")
    client = LLMClient(_models(), transport, max_retries=3)

    result = client.generate_structured(
        role="selector", model_cls=SelectorOutput, system="sys", user="usr"
    )
    assert isinstance(result.parsed, SelectorOutput)
    # 2 empty + 1 valid = 3 calls; the empty path never grows the prompt (stays system+user).
    assert len(transport.msg_lens) == 3
    assert transport.msg_lens == [2, 2, 2]


def test_blank_content_also_transient(monkeypatch):
    """Whitespace-only content with a non-error finish is treated as transient too."""
    import wikitrace.harness.llm.client as client_mod

    monkeypatch.setattr(client_mod.time, "sleep", lambda *_: None)
    transport = _EmptyTransport(n_empty=1, finish_reason="stop", content="   \n")
    client = LLMClient(_models(), transport, max_retries=3)
    result = client.generate_structured(
        role="selector", model_cls=SelectorOutput, system="sys", user="usr"
    )
    assert isinstance(result.parsed, SelectorOutput)
    assert transport.msg_lens == [2, 2]


def test_persistent_empty_raises_without_prompt_growth(monkeypatch):
    import wikitrace.harness.llm.client as client_mod

    monkeypatch.setattr(client_mod.time, "sleep", lambda *_: None)
    transport = _EmptyTransport(n_empty=99, finish_reason="error")
    client = LLMClient(_models(), transport, max_retries=2)
    with pytest.raises(StructuredParseError):
        client.generate_structured(
            role="selector", model_cls=SelectorOutput, system="sys", user="usr"
        )
    # initial + 2 retries == 3 calls, prompt never grew.
    assert transport.msg_lens == [2, 2, 2]


# ---------------------------------------------------------------------------
# Usage / cost logging
# ---------------------------------------------------------------------------


def test_generate_structured_records_usage_and_on_call_sink(fake_transport_factory):
    sink: list = []
    transport = fake_transport_factory()
    client = LLMClient(_models(), transport, max_retries=2, on_call=sink.append)

    result = client.generate_structured(
        role="selector",
        model_cls=SelectorOutput,
        system="sys",
        user="usr",
        sample_id="s0",
    )

    rec = result.record
    assert rec.prompt_tokens == 1
    assert rec.completion_tokens == 1
    assert rec.role == "selector"
    assert rec.sample_id == "s0"
    # on_call sink received the same record.
    assert len(sink) == 1
    assert sink[0] is rec
