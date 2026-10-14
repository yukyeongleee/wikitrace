"""Unit tests for the analysis-side trajectory helpers in traverse.turns."""

from __future__ import annotations

import json

from wikitrace.harness.terminal.turns import (
    call_text,
)


def test_call_text_bash_vs_read():
    assert call_text("bash", {"command": "cat /work/corpus/x"}) == "cat /work/corpus/x"
    assert call_text("read", {"path": "/work/corpus/knowledge/document-0001.md"}) == \
        "read /work/corpus/knowledge/document-0001.md"
    assert call_text("weird", {"foo": 1}).startswith("weird ")


def _write(p, turns):
    p.write_text("\n".join(json.dumps(t) for t in turns), encoding="utf-8")


