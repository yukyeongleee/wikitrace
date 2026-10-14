"""High-level LLM client: role routing, structured output, retry, logging; the transport is injectable."""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from wikitrace.harness.llm.openrouter import ChatResponse, OpenRouterClient
from wikitrace.harness.llm.structured_output import (
    StructuredParseError,
    parse_into,
    response_format_for,
)
from wikitrace.harness.llm.usage import parse_usage
from wikitrace.config import LLMCallRecord
from wikitrace.config import ModelSpec, ModelsConfig

T = TypeVar("T", bound=BaseModel)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class StructuredResult:
    parsed: BaseModel
    record: LLMCallRecord
    messages: list[dict[str, str]]
    raw_response: dict[str, Any]


Transport = OpenRouterClient


class LLMClient:
    def __init__(
        self,
        models: ModelsConfig,
        transport: Transport,
        max_retries: int = 3,
        on_call: Optional[Callable[[LLMCallRecord], None]] = None,
    ):
        self.models = models
        self.transport = transport
        self.max_retries = max_retries
        self.on_call = on_call

    def generate_structured(
        self,
        role: str,
        model_cls: Type[T],
        system: str,
        user: str,
        phase: str = "",
        sample_id: str = "",
        spec: Optional[ModelSpec] = None,
    ) -> StructuredResult:
        """Call the role's model, parse into `model_cls`, retry on failure."""
        spec = spec or self.models.for_role(role)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        response_format = response_format_for(model_cls)

        call_id = uuid.uuid4().hex[:16]
        start = time.time()
        last_err = ""
        last_content = ""
        response: Optional[ChatResponse] = None
        parsed: Optional[BaseModel] = None
        retry_count = 0

        for attempt in range(self.max_retries + 1):
            retry_count = attempt
            try:
                response = self.transport.chat(
                    model=spec.model,
                    messages=messages,
                    temperature=spec.temperature,
                    max_tokens=spec.max_tokens,
                    response_format=response_format,
                    reasoning_effort=spec.reasoning_effort,
                )
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2 ** attempt, 8))
                continue

            # A TRANSIENT hiccup: retry the SAME prompt with no nudge, since a grown history only makes failure likelier.
            last_content = response.content or ""
            if not last_content.strip() or response.finish_reason == "error":
                last_err = f"empty/error response (finish_reason={response.finish_reason!r})"
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2 ** attempt, 8))
                continue

            try:
                parsed = parse_into(model_cls, response.content)
                break
            except (ValidationError, ValueError) as e:
                last_err = f"{type(e).__name__}: {e}"
                messages = messages + [
                    {"role": "assistant", "content": last_content},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response did not match the required "
                            f"JSON schema ({model_cls.__name__}). Error: {last_err}. "
                            "Return ONLY a single valid JSON object, no markdown."
                        ),
                    },
                ]

        latency = time.time() - start

        if parsed is None or response is None:
            record = LLMCallRecord(
                call_id=call_id,
                role=role,
                phase=phase,
                sample_id=sample_id,
                model_requested=spec.model,
                latency=latency,
                retry_count=retry_count,
                finish_reason="error",
            )
            if self.on_call:
                self.on_call(record)
            raise StructuredParseError(
                f"Structured generation failed for role={role} after "
                f"{retry_count} retries: {last_err}",
                last_content=last_content,
            )

        usage = parse_usage(response.raw)
        record = LLMCallRecord(
            call_id=call_id,
            generation_id=response.generation_id,
            role=role,
            phase=phase,
            sample_id=sample_id,
            model_requested=spec.model,
            model_returned=response.model_returned,
            prompt_hash=_hash(system + "\n" + user),
            response_hash=_hash(response.content),
            latency=latency,
            retry_count=retry_count,
            finish_reason=response.finish_reason,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            cached_tokens=usage.cached_tokens,
            cost=usage.cost,
            provider=getattr(response, "provider", "") or "",
        )
        if self.on_call:
            self.on_call(record)

        return StructuredResult(
            parsed=parsed,
            record=record,
            messages=messages,
            raw_response=response.raw,
        )
