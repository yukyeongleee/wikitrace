"""Sentinel-file reading and free-text JSON fallback; the caller validates against its schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def read_sentinel(out_dir: Path, name: str) -> tuple[Optional[dict], str]:
    """Return ``(parsed_dict_or_None, raw_text)`` for the sentinel file."""
    path = Path(out_dir) / name
    if not path.exists():
        return None, ""
    # ``errors="replace"``: one binary byte from the agent's shell redirection must not cost a finished trajectory.
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        obj = json.loads(text)
        return (obj if isinstance(obj, dict) else None), text
    except json.JSONDecodeError:
        return extract_json_fallback(text), text


def extract_json_fallback(text: str) -> Optional[dict]:
    """Extract the last top-level balanced ``{...}`` JSON object from `text`."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    best: Optional[dict] = None
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    chunk = text[start : i + 1]
                    try:
                        obj = json.loads(chunk)
                        if isinstance(obj, dict):
                            best = obj
                    except json.JSONDecodeError:
                        pass
    return best
