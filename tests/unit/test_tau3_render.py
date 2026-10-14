"""The tau3 compiler side: source-labelled observation rows, the step metadata, evaluation.json."""

from __future__ import annotations

import json
from pathlib import Path

from wikitrace.config import append_result
from wikitrace.config import ToolStep
from wikitrace.envs import resolve
from wikitrace.envs.tau3.adapter import build_trajectory_text, render_conversation_steps
from wikitrace.envs.tau3.models import TauAction, TauBenchResult, TauMessage
from wikitrace.envs.tau3.workflow import (
    _episode_meta,
    actions_from_trajectory,
    command_text,
    messages_from_turns,
)
from wikitrace.envs.textgame.models import TextGameOutcome, TextGameSample
from wikitrace.envs.textgame.workflow import TextGamePhaseLog
from wikitrace.harness.environment.server import Episode
from wikitrace.memory.store import render_raw_index_from_steps

FIRST = "Hi, I want to return the shoes from order #W1234567."
LOOKUP = '{"action":"find_user_id_by_email","argument":{"email":"a@example.com"}}'
BAD = '{"action":"respond","argument":{"text":"hi"}}'
NOT_FOUND = "Error: Tool 'respond' not found."
BLOCKED = "one take_action per turn: read this observation before sending the next action"
ASK = "Could you give me your email?"


def test_command_text_reads_every_call_as_tool_and_argument_json_and_leaves_the_rest_alone() -> None:
    assert command_text(LOOKUP) == 'find_user_id_by_email {"email":"a@example.com"}'
    assert command_text(BAD) == 'respond {"text":"hi"}'
    assert command_text("not json") == "not json"


def test_actions_keep_the_servers_source_mark_and_skip_messages() -> None:
    lines = [{"step": 0, "role": "environment", "observation": FIRST, "source": "user"},
             {"step": 0, "role": "message", "message": ASK, "observation": "a@example.com", "source": "user"},
             {"step": 1, "role": "agent", "command": LOOKUP},
             {"step": 1, "role": "environment", "observation": "sara_doe_496", "source": "env", "info": {}}]
    acts = actions_from_trajectory(lines)
    assert [(a.step, a.command, a.observation, a.source) for a in acts] == [(1, LOOKUP, "sara_doe_496", "env")]


def _steps() -> list[ToolStep]:
    return [
        ToolStep(reasoning="", command=f"> {LOOKUP}", result=BLOCKED, is_error=True),
        ToolStep(reasoning="look the user up", command=f"> {LOOKUP}", result="sara_doe_496"),
        ToolStep(reasoning="", command=f"> {BAD}", result=NOT_FOUND),
        ToolStep(reasoning="", command="cat corpus/knowledge/_index.md", result="# index\nline"),
    ]


def _served() -> list[TauAction]:
    return [TauAction(step=1, command=LOOKUP, observation="sara_doe_496", source="env"),
            TauAction(step=2, command=BAD, observation=NOT_FOUND, source="env", invalid=True)]


def _messages() -> list[TauMessage]:
    return [TauMessage(after=0, text=ASK, reasoning="ask for the email", reply="a@example.com\nthanks"),
            TauMessage(after=4, text="Your return is filed.\nAnything else?", reply="No. ###STOP###")]


def test_messages_are_steps_of_their_own_among_the_labelled_tool_steps() -> None:
    assert render_conversation_steps(_steps(), _served(), _messages()) == (
        "[1]\n"
        "  reason      │ ask for the email\n"
        "  message     │ Could you give me your email?\n"
        "  obs(user)   │ a@example.com\n"
        "              │ thanks\n"
        "[2]\n"
        "  reason      │ look the user up\n"
        '  take_action │ find_user_id_by_email {"email":"a@example.com"}\n'
        "  obs(env)    │ sara_doe_496\n"
        "[3]\n"
        '  take_action │ respond {"text":"hi"}\n'
        "  obs(env)    │ Error: Tool 'respond' not found.\n"
        "[4]\n"
        "  call        │ cat corpus/knowledge/_index.md\n"
        "  result      │ # index line\n"
        "[5]\n"
        "  message     │ Your return is filed.\n"
        "              │ Anything else?\n"
        "  obs(user)   │ No. ###STOP###"
    )


def test_without_a_source_mark_the_row_is_a_bare_obs_and_a_message_without_reply_has_no_obs_row() -> None:
    text = render_conversation_steps(_steps()[1:2], None, [TauMessage(after=1, text=ASK)])
    assert text == ("[1]\n  reason      │ look the user up\n"
                    '  take_action │ find_user_id_by_email {"email":"a@example.com"}\n  obs         │ sara_doe_496\n'
                    "[2]\n  message     │ Could you give me your email?")


def test_message_rows_are_read_off_the_turn_log_and_the_final_reply_row_is_not_one() -> None:
    rows = [{"type": "message", "role": "system", "content": "SYS"},
            {"type": "message", "role": "user", "content": f"Here is the task.\n{FIRST}"},
            {"type": "message", "role": "assistant", "content": ASK, "reasoning": "ask for the email"},
            {"type": "message", "role": "user", "content": "a@example.com"},
            {"type": "function_call", "name": "take_action", "arguments": {}},
            {"type": "message", "role": "assistant", "content": "Done."},
            {"type": "episode_over"},
            {"type": "request_context", "messages": []},
            {"type": "message", "role": "assistant", "content": "Done.", "reasoning": ""}]
    assert messages_from_turns(rows) == [
        TauMessage(after=0, text=ASK, reasoning="ask for the email", reply="a@example.com"),
        TauMessage(after=1, text="Done.")]


def _result(tool_steps=None, actions=None, **kw) -> TauBenchResult:
    return TauBenchResult(sample_id="tau3_0000_10", db_id="retail", question=FIRST, task_type="retail",
                          variant="10", success=True, steps=3, ended_by="done",
                          tool_steps=tool_steps or [], actions=actions or [], **kw)


def test_the_compiler_trajectory_uses_the_served_marks_when_they_line_up() -> None:
    dom = resolve("tau3", "tau3")
    text = dom.render_trajectory_for_compiler(_result(_steps(), _served(), messages=_messages()))
    assert text == "## Episode (agent transcript)\n" + render_conversation_steps(_steps(), _served(), _messages())
    # A conversation of messages alone is still the agent's transcript.
    assert build_trajectory_text(_result(messages=_messages()[:1])).startswith(
        "## Episode (agent transcript)\n[1]\n  reason      │ ask for the email\n  message     │")
    # Without a Pi log the server's record stands, the commands rendered the same way.
    assert build_trajectory_text(_result(actions=_served()[:1])) == (
        '## Episode\n> find_user_id_by_email {"email":"a@example.com"}\nsara_doe_496')
    assert dom.render_outcome_for_compiler(_result()) == "## Outcome\noutcome: success\nsteps: 3 take_action calls"


def test_the_metadata_names_the_user_request_and_leaves_the_counters_to_the_evaluation(tmp_path: Path) -> None:
    dom = resolve("tau3", "tau3")
    result = _result(meta={"task_id": "10", "orchestrator_steps": 12, "messages": 3, "num_errors": 1,
                           "stop_token": "###STOP###", "judge_attempts": 1})
    sample = TextGameSample(id=result.sample_id, db_id="retail", task_type="retail", variant="10", task_text=FIRST)
    dom.persist_traverse_trajectory(tmp_path, sample, TextGameOutcome(result))
    assert (tmp_path / "metadata.md").read_text(encoding="utf-8") == (
        "# tau3_0000_10\n\n- task_type: retail\n"
        f"- user_request: {FIRST}\n- success: True\n- steps: 3 take_action calls\n")
    assert dom.raw_layer_shape.metadata_keys == ("task_type", "success", "steps")
    assert dom.raw_layer_shape.task_key == "user_request"
    # A mounted turn log drops its `request_context` row, as the textgame shape it extends does.
    assert dom.raw_layer_shape.mount_without_request_context
    ci = dom.compilation_inputs(result)
    assert ci.task_fields == (("task_type", "retail"), ("user_request", FIRST))
    assert dom.sample_row(result, "p")["stop_token"] == "###STOP###"


def test_the_episode_meta_takes_the_counters_and_the_per_role_calls_and_cost() -> None:
    evaluation = {"success": True, "reward": 1.0, "orchestrator_steps": 12, "messages": 3, "num_errors": 0,
                  "stop_token": "###STOP###", "judge_attempts": 2, "judge_failed": False,
                  "calls": {"user_simulator": 4, "judge": 2}, "cost": {"user_simulator": 0.01, "judge": 0.002}}
    assert _episode_meta(evaluation) == {
        "orchestrator_steps": 12, "messages": 3, "num_errors": 0, "stop_token": "###STOP###", "judge_attempts": 2,
        "judge_failed": False, "simulator_calls": 4, "simulator_cost": 0.01, "judge_calls": 2, "judge_cost": 0.002}


class _Scored:
    """An environment that scores at close, as the tau3 one does."""

    def __init__(self, attempts: int) -> None:
        self.attempts = attempts

    def reset(self):
        return FIRST, {"source": "user", "task": FIRST}

    def step(self, command):
        return "sara_doe_496", False, {"source": "env", "invalid_command": False}

    def close(self):
        pass

    def evaluation(self):
        failed = self.attempts >= 3
        return {"success": not failed, "score": 0.0 if failed else 1.0, "reward": 0.0 if failed else 1.0,
                "reward_basis": ["DB", "NL_ASSERTION"], "reward_breakdown": {"DB": 1.0, "NL_ASSERTION": 0.0 if failed else 1.0},
                "db_check": {"db_match": True}, "nl_assertions": None if failed else [{"met": True}],
                "judge_attempts": self.attempts, "calls": {"user_simulator": 1, "judge": self.attempts},
                "cost": {"user_simulator": 0.0, "judge": 0.0}}


def test_evaluation_json_holds_the_records_fields_including_judge_attempts(tmp_path: Path) -> None:
    for attempts, success in ((2, True), (3, False)):
        out = tmp_path / str(attempts)
        out.mkdir()
        ep = Episode(lambda: _Scored(attempts), out)
        ep.reset()
        ep.step(LOOKUP)
        ep.close("agent_stopped")
        ev = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
        assert ev["success"] is success and ev["ended_by"] == "agent_stopped"
        assert ev["steps"] == 1 and ev["invalid_commands"] == 0
        assert ev["judge_attempts"] == attempts and ev["calls"]["judge"] == attempts
        assert (ev["nl_assertions"] is None) is (attempts == 3)
        assert set(ev) >= {"reward", "reward_basis", "reward_breakdown", "db_check", "nl_assertions", "judge_attempts"}
        assert not {"action_checks", "communicate_checks", "env_assertions", "info"} & set(ev)


def _corpus(tmp_path: Path):
    dom = resolve("tau3", "tau3")
    sample = TextGameSample(id="tau3_0000_10", db_id="retail", task_type="retail", variant="10", task_text=FIRST)
    ask, bye = _messages()[0], _messages()[1].model_copy(update={"after": 3})
    result = _result(_steps()[:3], _served(), messages=[ask, bye],
                     meta={"orchestrator_steps": 5, "messages": 2, "num_errors": 0, "stop_token": "###STOP###"})
    step_dir = tmp_path / "corpus" / "raw" / "step-0000"
    dom.persist_traverse_trajectory(step_dir, sample, TextGameOutcome(result))
    rows = [{"type": "message", "role": "system", "content": "SYS"},
            {"type": "message", "role": "user", "content": f"Here is the task.\n{FIRST}"},
            {"type": "message", "role": "assistant", "content": ask.text, "reasoning": ask.reasoning},
            {"type": "message", "role": "user", "content": ask.reply}]
    for s in _steps()[:3]:
        rows.append({"type": "function_call", "name": "take_action", "arguments": json.loads(s.command[2:]),
                     "reasoning": s.reasoning, "result": s.result, "is_error": s.is_error})
    rows += [{"type": "message", "role": "assistant", "content": bye.text, "reasoning": ""},
             {"type": "message", "role": "user", "content": bye.reply}, {"type": "episode_over"},
             {"type": "request_context", "messages": []},
             {"type": "message", "role": "assistant", "content": bye.text, "reasoning": ""}]
    (step_dir / "solver.full.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    append_result(tmp_path / "corpus", "step-0000", result)
    return dom, tmp_path / "corpus"


MULTI = '{"action":"modify_user_address","argument":{"user_id":"sara_doe_496","address1":"1 Main St\\nApt 2"}}'
TWO_PARAGRAPHS = "I need to return two orders.\n\nCan the refunds go to my gift card?"


def test_the_raw_index_counts_the_messages_to_the_user_among_the_steps() -> None:
    opening = render_raw_index_from_steps([], raw_shape=resolve("tau3", "tau3").raw_layer_shape)
    assert "the assistant `message` rows, each a message to the user" in opening
    assert "The episode's steps are the" not in opening


def _phase_log(turns: list[dict], final_text: str) -> TextGamePhaseLog:
    return TextGamePhaseLog(phase="solver", turns=turns, read_set=[], sentinel=None, final_text=final_text,
                            final_context=[], system_prompt="SYS", user_prompt=f"Here is the task.\n{FIRST}")


def test_the_final_reply_row_is_written_only_when_the_channel_did_not_already_record_it(tmp_path: Path) -> None:
    dom = resolve("tau3", "tau3")
    sample = TextGameSample(id="tau3_0000_10", db_id="retail", task_type="retail", variant="10", task_text=FIRST)
    sent = [{"type": "message", "role": "assistant", "content": "Done."},
            {"type": "message", "role": "user", "content": "###STOP###"}, {"type": "episode_over"}]
    tails = {}
    for name, final in (("repeat", "Done."), ("after_episode_over", "The episode ended.")):
        step_dir = tmp_path / name
        outcome = TextGameOutcome(_result(), phase_logs=[_phase_log([dict(t) for t in sent], final)])
        dom.persist_traverse_trajectory(step_dir, sample, outcome)
        rows = [json.loads(ln) for ln in (step_dir / "solver.full.jsonl").read_text().splitlines()]
        tails[name] = [(r["type"], r.get("content")) for r in rows[rows.index(next(
            r for r in rows if r["type"] == "request_context")):]]
    assert tails["repeat"] == [("request_context", None), ("answer", None)]
    assert tails["after_episode_over"] == [("request_context", None), ("message", "The episode ended."),
                                           ("answer", None)]
