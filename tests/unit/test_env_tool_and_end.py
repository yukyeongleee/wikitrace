"""The environment tool name and the end phrase as benchmark attributes, and how they read back."""

from __future__ import annotations

from wikitrace.config import ToolStep
from wikitrace.envs.textgame import adapter as tg_adapter
from wikitrace.envs.textgame.benchmarks.alfworld import AlfworldBenchmarkAdapter
from wikitrace.envs.textgame.benchmarks.base import TextGameBenchmark
from wikitrace.envs.textgame.benchmarks.scienceworld import ScienceworldBenchmarkAdapter
from wikitrace.envs.textgame.models import TextGameResult
from wikitrace.envs.textgame.workflow import _command
from wikitrace.harness.environment import EnvTool, step_input


def test_env_tool_defaults_to_act_for_the_existing_benchmarks():
    assert TextGameBenchmark.env_tool == "act"
    assert AlfworldBenchmarkAdapter.env_tool == ScienceworldBenchmarkAdapter.env_tool == "act"


def test_step_input_joins_take_action_and_passes_act_through():
    assert step_input("take_action", {"action": "search", "argument": "iron_ingot"}) == "search: iron_ingot"
    assert step_input("act", {"command": "go to desk 1"}) == "go to desk 1"


def test_command_reads_the_benchmarks_tool_only():
    turn = {"name": "take_action", "arguments": {"action": "move", "argument": "from [I1] to [A1] with quantity 1"}}
    assert _command(turn, "take_action") == "> move: from [I1] to [A1] with quantity 1"
    act = {"name": "act", "arguments": {"command": "look"}}
    assert _command(act) == "> look"
    # Under the other tool's name the call is an ordinary tool call, not a game step.
    assert not _command(turn).startswith("> ")


def test_blocked_matches_the_named_tools_refusal():
    s = ToolStep(command="> search: x", result="one take_action per turn: read this observation",
                 is_error=True)
    assert tg_adapter._blocked(s, "take_action") and not tg_adapter._blocked(s)


def test_steps_text_names_the_benchmarks_tool(tmp_path):
    class Crafting(TextGameBenchmark):
        env_tool = "take_action"

    r = TextGameResult(sample_id="s", db_id="d", question="q", steps=3, ended_by="done")
    assert Crafting().steps_text(r) == "3 take_action calls"
    assert TextGameBenchmark().steps_text(r) == "3 act calls"
    assert EnvTool("take_action", tmp_path).extension.name == "take_action.ts"


def test_the_end_phrase_is_the_benchmarks_only_for_an_end_command_episode():
    class Crafting(TextGameBenchmark):
        env_tool = "take_action"
        end_phrase = "ended by the agent: it gave up"

    r = TextGameResult(sample_id="s", db_id="d", question="q", steps=4, ended_by="agent_stopped",
                       ended_on_end_command=True)
    assert Crafting().steps_text(r) == "4 take_action calls (ended by the agent: it gave up)"
    # Every other benchmark keeps the shared phrase, end command or not.
    assert AlfworldBenchmarkAdapter().steps_text(r) == "4 act calls (ended by the agent)"
    # A session the agent ended with a reply and no tool call did not give up.
    r.ended_on_end_command = False
    assert Crafting().steps_text(r) == "4 take_action calls (ended by the agent)"


def test_the_server_flag_reaches_the_result_and_the_unserved_row_no_action():
    from wikitrace.envs.textgame.models import TextGameSample
    from wikitrace.envs.textgame.workflow import actions_from_trajectory, result_from_server

    trajectory = [
        {"step": 0, "role": "environment", "observation": "start", "info": {}},
        {"step": 1, "role": "agent", "command": "search: planks"},
        {"step": 1, "role": "environment", "observation": "recipe", "info": {}},
        {"step": 1, "role": "agent", "command": "impossible: no logs", "served": False},
    ]
    actions = actions_from_trajectory(trajectory)
    assert [(a.step, a.command) for a in actions] == [(1, "search: planks")]
    sample = TextGameSample(id="s", db_id="d", task_type="t", variant="v", task_text="q", spec={})
    served = {"ended_by": "agent_stopped", "end_command": True, "steps": 1}
    assert result_from_server(sample, served, actions).ended_on_end_command is True
    assert result_from_server(sample, {"ended_by": "agent_stopped"}, actions).ended_on_end_command is False
