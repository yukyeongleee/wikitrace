"""Protect frozen manifests from direct-test execution."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_direct_prompt_check_cannot_overwrite_a_drifting_manifest(tmp_path, monkeypatch):
    path = ROOT / "tests/regression/test_prompt_render_matrix.py"
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.If) and "__name__" in ast.unparse(n.test))
    fixture = tmp_path / "hashes.json"
    fixture.write_text("original")

    def fail_validation():
        raise AssertionError("prompt drift")

    monkeypatch.setattr("sys.argv", [str(path)])
    with pytest.raises(AssertionError, match="prompt drift"):
        exec(compile(ast.Module(body=main.body, type_ignores=[]), str(path), "exec"),
             {"MANIFEST": fixture, "test_every_traverse_protocol_combination_is_pinned": fail_validation})
    assert fixture.read_text() == "original"

