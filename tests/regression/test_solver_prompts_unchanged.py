"""The environment solvers' own prompts do not move; only the memory block they include.
The fixture holds each one as the base commit rendered it under `access="none"`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import wikitrace.envs  # noqa: F401  (registers the generator prompt roles)
from wikitrace.harness.prompts import render_system, render_user

GOLDEN = Path(__file__).parents[1] / "fixtures/prompts/no_memory_solver_prompts.json"

_SUPERVISOR = {"first_name": "Joyce", "last_name": "Weaver", "email": "j@example.com",
               "phone_number": "+1-123-456-7890", "password": "abc"}

CASES = {
    "textgame|solver.system": lambda: render_system(
        "textgame_solver", intro="<INTRO>", exemplars="<EXEMPLARS>", access="none"),
    "textgame|solver.user": lambda: render_user(
        "textgame_solver", task="<TASK>", memory_documents=""),
    "appworld|solver.traverse.system": lambda: render_system(
        "appworld_solver", "traverse", exemplar="<EXEMPLAR>", access="none"),
    "appworld|solver.traverse.user": lambda: render_user(
        "appworld_solver", "traverse", instruction="<TASK>", main_user=_SUPERVISOR,
        observation="", memory_documents=""),
}


@pytest.mark.parametrize("key", sorted(CASES))
def test_the_solver_prompt_is_byte_identical_to_the_base_commit(key):
    assert CASES[key]() == json.loads(GOLDEN.read_text())[key]
