"""The tau3 episode loop around a scripted environment: the worker's key, the call rows, the failure paths."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from unittest.mock import Mock

import pytest

from wikitrace.envs import resolve
from wikitrace.envs.tau3.benchmarks.tau3 import Tau3BenchmarkAdapter
from wikitrace.envs.tau3.workflow import env_call_records, run_tau3_episode
from wikitrace.envs.textgame.models import TextGameSample
from wikitrace.harness.environment.protocol import request
from wikitrace.harness.terminal import FakeTraverseRunner, TerminalResult

KEY = "sk-or-v1-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
FIRST = "Hi, I want to return the shoes from order #W1234567."
LOOKUP = {"action": "get_user_details", "argument": {"user_id": "chen_silva_7485"}}
ROWS = [
    {"role": "user_simulator", "model_requested": "openrouter/openai/gpt-5.4-mini", "model_returned": "openai/gpt-5.4-mini",
     "generation_id": "gen-1", "provider": "OpenAI", "finish_reason": "stop", "latency": 1.5,
     "prompt_tokens": 400, "completion_tokens": 30, "reasoning_tokens": 12, "cached_tokens": 0, "cost": 0.0009},
    {"role": "judge", "model_requested": "openrouter/openai/gpt-5.4-mini", "model_returned": "openai/gpt-5.4-mini",
     "generation_id": "gen-2", "provider": "Azure", "finish_reason": "stop", "latency": 4.0,
     "prompt_tokens": 2000, "completion_tokens": 200, "reasoning_tokens": 150, "cached_tokens": 100, "cost": 0.004},
]

# A retail environment that reports how long a key the server saw, and the rows of two calls.
_FAKE_ENV = textwrap.dedent(f'''
    import os
    FIRST = {FIRST!r}
    ROWS = {ROWS!r}


    class Env:
        def __init__(self, spec):
            self.mode = spec.get("mode", "ok")

        def reset(self):
            if self.mode == "resetfail":
                raise RuntimeError("the simulator refused")
            return FIRST, {{"source": "user", "task": FIRST}}

        def step(self, command):
            return "sara_doe_496", False, {{"source": "env", "invalid_command": False}}

        def close(self):
            pass

        def evaluation(self):
            out = {{"success": True, "score": 1.0, "scored": True, "reward": 1.0, "orchestrator_steps": 3,
                    "messages": 0, "num_errors": 0, "stop_token": "", "judge_attempts": 1, "judge_failed": False,
                    "calls": {{"user_simulator": 1, "judge": 1}}, "cost": {{"user_simulator": 0.0009, "judge": 0.004}},
                    "key_length": len(os.environ.get("OPENROUTER_API_KEY", "")), "llm_calls": ROWS}}
            if self.mode == "genfail":
                out["generation_error"] = "user_simulator: IndexError: pop from empty list"
            return out


    def make_env(spec):
        return Env(spec)
''')


def _handle(*, env_tool, **_):
    reply = request(str(env_tool.socket_dir / "act.sock"),
                    {"op": "step", "input": json.dumps(LOOKUP, separators=(",", ":"))})
    turns = [{"type": "function_call", "name": "take_action", "arguments": LOOKUP, "call_id": "c0"}]
    ctx = [{"role": "assistant", "content": [{"type": "thinking", "thinking": "look the user up"},
                                             {"type": "toolCall", "id": "c0"}]},
           {"role": "toolResult", "toolCallId": "c0", "content": [{"type": "text", "text": reply["observation"]}]}]
    return TerminalResult(final_text="done", turns=turns, final_context=ctx, read_set=[],
                          usage={"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.05})


@pytest.fixture
def bench(tmp_path, monkeypatch):
    env = tmp_path / "fake_tau3_env.py"
    env.write_text(_FAKE_ENV, encoding="utf-8")
    monkeypatch.setattr(Tau3BenchmarkAdapter, "env_factory", f"{env}:make_env")
    monkeypatch.setattr(Tau3BenchmarkAdapter, "python", sys.executable)
    return Tau3BenchmarkAdapter()


def _sample(mode: str = "ok") -> TextGameSample:
    return TextGameSample(id="tau3_0000_10", db_id="retail", task_type="retail", variant="10", task_text="",
                          split_role="test", spec={"task_id": "10", "mode": mode})


def _runner() -> FakeTraverseRunner:
    runner = FakeTraverseRunner(_handle)
    runner.api_key = KEY
    return runner


def _files(*roots: Path) -> dict[str, str]:
    return {str(p): p.read_text(encoding="utf-8", errors="replace")
            for root in roots for p in root.rglob("*") if p.is_file()}


def test_the_workers_key_reaches_the_server_and_no_file_the_episode_writes(tmp_path, bench) -> None:
    sample, rows = _sample(), []
    outcome = run_tau3_episode(sample, _runner(), None, tmp_path / "sessions", bench=bench, on_call=rows.append)
    result = outcome.result
    assert result.success and result.score == 1.0 and not outcome.generation_failed
    assert result.meta["judge_failed"] is False and result.meta["simulator_cost"] == 0.0009
    step_dir = tmp_path / "raw" / "step-0000"
    resolve("tau3", "tau3").persist_traverse_trajectory(step_dir, sample, outcome)
    files = _files(tmp_path)
    ev = json.loads(files[str(tmp_path / "sessions" / sample.id / "solver" / "evaluation.json")])
    # The server saw the whole key; nothing written holds it.
    assert ev["key_length"] == len(KEY)
    assert {"evaluation.json", "env_server.log", "metadata.md"} <= {Path(p).name for p in files}
    assert [p for p, text in files.items() if KEY in text] == []
    assert KEY not in result.model_dump_json() and KEY not in json.dumps(outcome.phase_logs[0].__dict__, default=str)
    assert KEY not in json.dumps([r.model_dump() for r in rows])
    # The simulator's and the judge's calls, as rows of the run's llm_calls.jsonl.
    assert [r.call_id for r in rows] == [f"tau3-user_simulator:test:{sample.id}:0", f"tau3-judge:test:{sample.id}:1"]
    assert [(r.role, r.phase, r.sample_id, r.provider, r.generation_id) for r in rows] == [
        ("user_simulator", "test", sample.id, "OpenAI", "gen-1"), ("judge", "test", sample.id, "Azure", "gen-2")]
    assert (rows[1].prompt_tokens, rows[1].completion_tokens, rows[1].reasoning_tokens, rows[1].cached_tokens,
            rows[1].cost, rows[1].latency, rows[1].model_returned) == (2000, 200, 150, 100, 0.004, 4.0, "openai/gpt-5.4-mini")


def test_call_rows_are_built_from_the_evaluation_and_nothing_else() -> None:
    sample = _sample()
    assert env_call_records({}, sample) == []
    (row,) = env_call_records({"llm_calls": [ROWS[0], "junk"]}, sample)
    assert row.call_id == f"tau3-user_simulator:test:{sample.id}:0" and row.model_requested == ROWS[0]["model_requested"]


def test_a_generation_failure_is_never_a_success(tmp_path, bench) -> None:
    outcome = run_tau3_episode(_sample("genfail"), _runner(), None, tmp_path / "sessions", bench=bench)
    assert outcome.generation_failed and outcome.result.success is False and outcome.result.score == 0.0
    assert outcome.result.generation_error == "user_simulator: IndexError: pop from empty list"


def test_a_failed_reset_skips_the_memory_binding_and_records_it(tmp_path, bench) -> None:
    sample, bind = _sample("resetfail"), Mock()
    with pytest.raises(RuntimeError, match="the simulator refused"):
        run_tau3_episode(sample, _runner(), None, tmp_path / "sessions", bench=bench, bind_memory=bind)
    bind.assert_not_called()
    assert sample.task_text == "" and sample.meta["memory_binding"] == "skipped: the reset failed"


def test_a_provider_error_closes_as_the_harness_and_is_not_scored(tmp_path, bench) -> None:
    from wikitrace.run.evaluate import ProviderErrorGuard

    def failed(**kw):
        res = _handle(**kw)
        res.provider_error = "429 Provider returned error"
        return res

    runner = FakeTraverseRunner(failed)
    sample = _sample()
    outcome = run_tau3_episode(sample, runner, None, tmp_path / "sessions", bench=bench)
    # The fake evaluation says success; a solver whose provider failed is still never scored.
    assert outcome.generation_failed and outcome.result.success is False and outcome.result.score == 0.0
    assert outcome.result.generation_error == "provider_error: 429 Provider returned error"
    assert outcome.phase_logs[0].provider_error == "429 Provider returned error"
    ev = json.loads((tmp_path / "sessions" / sample.id / "solver" / "evaluation.json").read_text())
    assert ev["ended_by"] == "harness_stopped"
    guard = ProviderErrorGuard(limit=2)
    guard.record(outcome, phase="test", step_n=0, sample_id=sample.id)
    with pytest.raises(RuntimeError, match="consecutive episodes"):
        guard.record(outcome, phase="test", step_n=1, sample_id=sample.id)


def test_messages_reach_the_raw_turn_log_in_order_and_nothing_else_is_a_call(tmp_path, bench) -> None:
    seen = {}

    def conversation(*, env_tool, **kw):
        seen["messages"] = env_tool.messages
        res = _handle(env_tool=env_tool, **kw)
        res.turns = [
            {"turn": 0, "type": "message", "role": "assistant", "content": "Your email?", "reasoning": "authenticate"},
            {"turn": 1, "type": "message", "role": "user", "content": "a@example.com"},
            {**res.turns[0], "turn": 2},
            {"turn": 3, "type": "message", "role": "assistant", "content": "Done.", "reasoning": "wrap up"},
            {"turn": 4, "type": "message", "role": "user", "content": "###STOP###"},
            {"turn": 5, "type": "episode_over"},
        ]
        res.final_text, res.final_reasoning = "Done.", "wrap up"
        return res

    sample = _sample()
    outcome = run_tau3_episode(sample, FakeTraverseRunner(conversation), None, tmp_path / "sessions", bench=bench)
    assert seen["messages"] is True
    step_dir = tmp_path / "raw" / "step-0000"
    resolve("tau3", "tau3").persist_traverse_trajectory(step_dir, sample, outcome)
    rows = [json.loads(ln) for ln in (step_dir / "solver.full.jsonl").read_text().splitlines()]
    kinds = [(r["type"], r.get("role"), r.get("content")) for r in rows if r["type"] in ("message", "function_call")]
    assert kinds[2:] == [
        ("message", "assistant", "Your email?"), ("message", "user", "a@example.com"),
        ("function_call", None, None),
        ("message", "assistant", "Done."), ("message", "user", "###STOP###"),
    ]
    # The final reply is the channel's last message, so no second row repeats it.
    assert [r["type"] for r in rows[-2:]] == ["request_context", "answer"]
    assert next(r for r in rows if r.get("content") == "Done.")["reasoning"] == "wrap up" and [r["name"] for r in rows if r["type"] == "function_call"] == ["take_action"]
    assert [s.command for s in outcome.result.tool_steps] == [
        '> {"action":"get_user_details","argument":{"user_id":"chen_silva_7485"}}']
    assert [m.model_dump() for m in outcome.result.messages] == [
        {"after": 0, "text": "Your email?", "reasoning": "authenticate", "reply": "a@example.com"},
        {"after": 1, "text": "Done.", "reasoning": "wrap up", "reply": "###STOP###"}]
    trajectory = resolve("tau3", "tau3").render_trajectory_for_compiler(outcome.result)
    assert trajectory.count("  message     │") == 2 and "  obs(user)   │ ###STOP###" in trajectory


@pytest.mark.parametrize("mode", ["resetfail", "sessionfail"])
def test_the_call_rows_are_delivered_when_the_episode_raises(tmp_path, bench, mode) -> None:
    def broken(**_):
        raise RuntimeError("the session broke")

    runner = FakeTraverseRunner(broken) if mode == "sessionfail" else _runner()
    rows = []
    with pytest.raises(RuntimeError):
        run_tau3_episode(_sample(mode), runner, None, tmp_path / "sessions", bench=bench, on_call=rows.append)
    # The fake evaluation stands for the billed calls the closed episode reports, once each.
    assert [r.role for r in rows] == ["user_simulator", "judge"]


def test_evaluate_hands_on_call_to_an_environment_episode_only() -> None:
    from types import SimpleNamespace

    from wikitrace.run.evaluate import _generate

    seen = []

    def domain(kind):
        def run_traverse(*_, **kw):
            seen.append((kind, kw.get("on_call")))
            return SimpleNamespace(phase_logs=[], generation_failed=False, result=None)
        return SimpleNamespace(benchmark=SimpleNamespace(harness_spec=lambda _: SimpleNamespace(kind=kind)),
                               run_traverse=run_traverse)

    llm = SimpleNamespace(on_call=lambda record: None)
    cfg = SimpleNamespace(tools="read,bash", max_turns=5, sentinel_name="__out.json", max_retries=1,
                          prompt_strategy="top-down")
    for kind in ("environment", "terminal"):
        _generate(domain(kind), SimpleNamespace(id="s", db_id="retail"), llm=llm, memory_text="",
                  run_dir=Path("."), events=SimpleNamespace(emit=lambda *a, **k: None), phase="test", step_n=0,
                  traverse_runner=SimpleNamespace(model="m"), corpus_dir=None, session_root=Path("."),
                  traverse_cfg=cfg, repair_fn=None)
    assert seen == [("environment", llm.on_call), ("terminal", None)]


