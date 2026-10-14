"""JSON structured-output parsing with bounded retry and a corrective instruction on failure."""

from __future__ import annotations

import json
import re
from typing import Any, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def strip_fences(text: str) -> str:
    """Remove a leading/trailing markdown code fence if present."""
    text = text.strip()
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text)
    return text.strip()


def extract_json(text: str) -> str:
    """Best-effort extraction of the first balanced JSON object from text."""
    text = strip_fences(text)
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def response_format_for(model_cls: Type[BaseModel]) -> dict[str, Any]:
    """Build an OpenAI/OpenRouter json_schema response_format for a model."""
    schema = model_cls.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model_cls.__name__,
            "strict": False,
            "schema": schema,
        },
    }


def parse_into(model_cls: Type[T], content: str) -> T:
    """Parse a completion string into a validated model (raises on failure)."""
    payload = json.loads(extract_json(content))
    return model_cls.model_validate(payload)


class StructuredParseError(RuntimeError):
    def __init__(self, message: str, last_content: str = ""):
        super().__init__(message)
        self.last_content = last_content
