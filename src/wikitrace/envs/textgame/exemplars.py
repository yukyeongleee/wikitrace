"""The authors' few-shot episodes, shipped as data files outside the prompt fingerprint."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache

from wikitrace.envs.textgame.manifest import DATA_DIR


def exemplar_path(name: str):
    return DATA_DIR / f"{name}_react_prompts.json"


@lru_cache(maxsize=None)
def load_exemplars(name: str) -> dict[str, str]:
    """``{key: episode transcript}`` as the harness's own prompt file states it."""
    return json.loads(exemplar_path(name).read_text(encoding="utf-8"))


def data_file_hashes(name: str) -> dict[str, str]:
    """sha256 per data file the benchmark reads at run time (exemplars, manifest), by name."""
    paths = (exemplar_path(name), DATA_DIR / f"{name}.manifest.json")
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}


def strip_think(transcript: str) -> str:
    """Drop the `> think:` turns and their `OK.` replies; the agent reasons in its own channel."""
    # Line-based: a think line may lack its `OK.` reply or carry trailing whitespace on it.
    out: list[str] = []
    skip_ok = False
    for line in transcript.split("\n"):
        if line.startswith("> think:"):
            skip_ok = True
            continue
        if skip_ok and line.strip() == "OK.":
            skip_ok = False
            continue
        skip_ok = False
        out.append(line)
    return "\n".join(out)
