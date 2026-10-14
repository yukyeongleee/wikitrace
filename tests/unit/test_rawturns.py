"""The mounted rawturns script numbers a past run by the turns of its raw log."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wikitrace.memory import rawturns


def _raw(tmp_path: Path) -> None:
    rows = [
        {"type": "message", "role": "system", "content": "SYS", "turn": "0"},
        {"type": "message", "role": "user", "content": "Here is the task.", "turn": "1"},
        {"turn": "2", "type": "function_call", "role": "assistant", "name": "take_action",
         "arguments": "{'action': 'search', 'argument': 'slab'}", "reasoning": "Find it.",
         "result": "recipe 1:\nstone at [A1]", "is_error": "False"},
        {"turn": "3", "type": "function_call", "role": "assistant", "name": "bash",
         "arguments": {"command": "ls /work"}, "reasoning": "", "result": "x" * 9000, "is_error": "False"},
        {"turn": "4", "type": "function_call", "role": "assistant", "name": "take_action",
         "arguments": "{'action': 'move', 'argument': 'from [I1] to [A1] with quantity 1'}",
         "reasoning": "", "result": "y" * 9000, "is_error": "False"},
        {"turn": "5", "type": "message", "role": "assistant", "content": "done"},
        {"turn": "6", "type": "episode_over"},
    ]
    step = tmp_path / "raw" / "step-0007"
    step.mkdir(parents=True)
    step.joinpath("solver.full.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    step.joinpath("metadata.md").write_text("# s\n\n- instruction: Craft a slab\n- success: True\n")


def test_actions_numbers_every_turn_by_the_raw_log(tmp_path, monkeypatch, capsys):
    _raw(tmp_path)
    monkeypatch.setattr(rawturns, "RAW", tmp_path / "raw")
    rawturns.main(["actions", "step-0007"])
    assert capsys.readouterr().out == (
        "step-0007 | outcome: success | turns 0..5\ntask: Craft a slab\n"
        "[2] take_action: search: slab\n[3] bash: ls /work\n"
        "[4] take_action: move: from [I1] to [A1] with quantity 1\n[5] message: done\n")


def test_turn_prints_whole_turns_up_to_the_limit(tmp_path, monkeypatch, capsys):
    _raw(tmp_path)
    monkeypatch.setattr(rawturns, "RAW", tmp_path / "raw")
    rawturns.main(["turn", "step-0007", "2"])
    out = capsys.readouterr().out
    assert "[turn 2] take_action\nreasoning:\nFind it.\naction: take_action\nsearch: slab\n" \
           "observation:\nrecipe 1:\nstone at [A1]\n" in out
    rawturns.main(["turn", "step-0007", "2", "5"])
    out = capsys.readouterr().out
    assert "[turn 3] bash" in out and "[turn 4]" not in out
    assert out.rstrip("\n").splitlines()[-1].startswith("turns 4..5 left out")
    with pytest.raises(SystemExit) as stop:
        rawturns.main(["turn", "step-0007", "9"])
    assert stop.value.code == 1
    assert capsys.readouterr().out == "No turn 9; this run has turns 0..5.\n"
