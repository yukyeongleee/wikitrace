"""Gates that stop a HARNESS fault from being recorded as a wrong answer.
A terminal session that never starts ends with an empty prediction the metrics cannot
tell from a genuinely wrong one; these turn that silence into an abort."""

from __future__ import annotations

import pytest
from subprocess import TimeoutExpired

from wikitrace.run.evaluate import DeadSessionGuard, ProviderErrorGuard
from wikitrace.harness.terminal.docker import _require_docker_image
from wikitrace.harness.terminal.sentinel import read_sentinel


class _Log:
    """One phase log, as the guard reads it."""

    def __init__(self, turns=None, sentinel=None, stopped_by="", provider_error="") -> None:
        self.turns = turns or []
        self.sentinel = sentinel
        self.stopped_by = stopped_by
        self.provider_error = provider_error


class _TerminalOutcome:
    def __init__(self, logs: list[_Log]) -> None:
        self.phase_logs = logs


class _ChatOutcome:
    """A chat outcome carries no ``phase_logs`` at all."""


def _alive() -> _TerminalOutcome:
    return _TerminalOutcome([_Log(turns=[{"type": "message"}], sentinel={"answer": "B"})])


def _dead() -> _TerminalOutcome:
    """What a container that never started leaves."""
    return _TerminalOutcome([
        _Log(sentinel={"answer": "The request has been processed successfully."},
             stopped_by="eof")
    ])


def test_dead_session_guard_ignores_chat_outcomes():
    """A chat suite has no terminal session, so it can never trip the dead-session guard."""
    guard = DeadSessionGuard()
    for step in range(10):
        guard.record(_ChatOutcome(), phase="test", step_n=step, sample_id="s")


def test_dead_session_guard_aborts_on_the_first_dead_session():
    """The guard aborts on the FIRST dead session: a default limit of one."""
    guard = DeadSessionGuard()
    with pytest.raises(RuntimeError, match="a terminal session started but produced NO turns"):
        guard.record(_dead(), phase="test", step_n=0, sample_id="alfworld_0000")


def test_dead_session_guard_message_only_counts_when_a_streak_was_tolerated():
    """The streak count appears only when tolerating one was configured."""
    guard = DeadSessionGuard(limit=2)
    guard.record(_dead(), phase="test", step_n=0, sample_id="s")
    with pytest.raises(RuntimeError, match="2 consecutive terminal sessions produced NO turns"):
        guard.record(_dead(), phase="test", step_n=1, sample_id="s")


def test_dead_session_guard_resets_on_a_live_session():
    """The counter is CONSECUTIVE, not cumulative — a live session clears it."""
    guard = DeadSessionGuard(limit=2)
    for step in range(20):
        guard.record(_dead(), phase="test", step_n=step, sample_id="s")
        guard.record(_alive(), phase="test", step_n=step, sample_id="s")
    assert guard.streak == 0


def test_dead_session_guard_spares_a_fake_runner_that_answers_without_turns():
    """A fake runner that answers with no turns is spared; only ``stopped_by`` judges."""
    guard = DeadSessionGuard()
    fake = _TerminalOutcome([_Log(sentinel={"answer": "B"})])
    for step in range(10):
        guard.record(fake, phase="train", step_n=step, sample_id="s")


def test_dead_session_guard_spares_a_session_killed_after_doing_work():
    """A session killed after doing work is spared; only ``eof`` means it never started."""
    guard = DeadSessionGuard()
    for reason in ("wall", "idle", "command", "turn_cap"):
        for step in range(5):
            guard.record(
                _TerminalOutcome([_Log(turns=[{"t": 1}], stopped_by=reason)]),
                phase="test", step_n=step, sample_id="s",
            )
    # eof AFTER real turns is also spared: work was done, so it is not a startup fault.
    for step in range(5):
        guard.record(
            _TerminalOutcome([_Log(turns=[{"t": 1}], stopped_by="eof")]),
            phase="test", step_n=step, sample_id="s",
        )


def test_dead_session_guard_spares_an_outcome_with_no_phase_logs_at_all():
    """An outcome with no phase logs at all is spared: that is not evidence about the container."""
    guard = DeadSessionGuard()
    for step in range(10):
        guard.record(_TerminalOutcome([]), phase="corpus", step_n=step, sample_id="s")


def _provider_failed() -> _TerminalOutcome:
    return _TerminalOutcome([_Log(turns=[{"t": 1}], provider_error="402 Insufficient credits")])


def test_provider_error_guard_aborts_on_the_third_consecutive_provider_error():
    guard = ProviderErrorGuard()
    guard.record(_provider_failed(), phase="test", step_n=0, sample_id="a")
    guard.record(_provider_failed(), phase="test", step_n=1, sample_id="b")
    with pytest.raises(RuntimeError, match="3 consecutive episodes ended on a model provider error") as e:
        guard.record(_provider_failed(), phase="test", step_n=2, sample_id="c")
    assert "402 Insufficient credits" in str(e.value) and "'c'" in str(e.value)


def test_provider_error_guard_resets_on_an_episode_without_one():
    guard = ProviderErrorGuard()
    for step in range(10):
        guard.record(_provider_failed(), phase="test", step_n=step, sample_id="s")
        guard.record(_provider_failed(), phase="test", step_n=step, sample_id="s")
        guard.record(_alive(), phase="test", step_n=step, sample_id="s")
    assert guard.streak == 0
    for step in range(5):
        guard.record(_ChatOutcome(), phase="test", step_n=step, sample_id="s")
        guard.record(_TerminalOutcome([]), phase="test", step_n=step, sample_id="s")


def test_read_sentinel_survives_an_undecodable_byte(tmp_path):
    """``read_sentinel`` survives an undecodable byte rather than losing the run."""
    (tmp_path / "__out.json").write_bytes(b'{"answer": "B\xff", "decision_summary": ""}')

    parsed, raw = read_sentinel(tmp_path, "__out.json")

    assert parsed is not None and parsed["answer"].startswith("B")
    assert raw  # the undecodable byte became U+FFFD rather than an exception


def test_require_docker_image_rejects_a_missing_tag(monkeypatch):
    """The real validation gate must reject a failed local image inspection."""
    from types import SimpleNamespace
    from unittest.mock import Mock
    import wikitrace.harness.terminal.docker as runner
    inspect = Mock(return_value=SimpleNamespace(returncode=1, stderr="No such image"))
    monkeypatch.setattr(runner, "subprocess", SimpleNamespace(run=inspect, TimeoutExpired=TimeoutExpired))
    with pytest.raises(RuntimeError, match="not present locally"):
        _require_docker_image("missing-tag")
    inspect.assert_called_once_with(
        ["docker", "image", "inspect", "missing-tag"], capture_output=True, text=True, timeout=30
    )


@pytest.mark.parametrize("failure, message", [
    (FileNotFoundError("missing docker"), "needs the docker CLI"),
    (TimeoutExpired(["docker", "image", "inspect"], 30), "timed out"),
])
def test_require_docker_image_reports_execution_failures(monkeypatch, failure, message):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import wikitrace.harness.terminal.docker as runner
    monkeypatch.setattr(runner, "subprocess", SimpleNamespace(run=Mock(side_effect=failure), TimeoutExpired=TimeoutExpired))
    with pytest.raises(RuntimeError, match=message):
        _require_docker_image("local-image")


def test_require_docker_image_accepts_present_image(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import wikitrace.harness.terminal.docker as runner
    monkeypatch.setattr(runner, "subprocess", SimpleNamespace(run=Mock(return_value=SimpleNamespace(returncode=0)), TimeoutExpired=TimeoutExpired))
    _require_docker_image("local-image")


# --- the two silences: a failure that RETURNS, and a kill that censors ------------


class _Events:
    """Collect emitted events the way `events.jsonl` records them."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def emit(self, event_type: str, **fields) -> None:
        self.rows.append({"event": event_type, **fields})


class _FailedOutcome:
    """What a terminal generator RETURNS when its sentinel never parses."""

    generation_failed = True

    class result:  # noqa: D106 - a stand-in for a domain result
        generation_error = "traverse generation failed: solver sentinel unparseable"


def test_generation_failed_event_fires_for_a_failure_that_was_returned_not_raised():
    """The generation-failed event fires for a failure that was returned, not raised."""
    from wikitrace.run.evaluate import _emit_if_failed

    ev = _Events()
    _emit_if_failed(_FailedOutcome(), ev, phase="corpus", step_n=1, sample_id="s0082")
    assert [r["event"] for r in ev.rows] == ["generation_failed"]
    # `phase` is passed through, so a corpus failure is not mislabelled `test`.
    assert ev.rows[0]["phase"] == "corpus"
    assert "sentinel unparseable" in ev.rows[0]["error"]


def test_generation_failed_event_is_silent_for_a_healthy_outcome():
    from wikitrace.run.evaluate import _emit_if_failed

    class _Ok:
        generation_failed = False

    ev = _Events()
    _emit_if_failed(_Ok(), ev, phase="test", step_n=0, sample_id="s")
    assert ev.rows == []


def test_generation_failed_event_reads_either_domain_error_field():
    """A domain may record `generation_error` or `exec_error`."""
    # Assuming one of them exists on every domain is what a shared emitter must not do.
    from wikitrace.run.evaluate import _emit_if_failed

    class _ExecFailure:
        generation_failed = True

        class result:  # noqa: D106
            exec_error = "generation failed: boom"

    ev = _Events()
    _emit_if_failed(_ExecFailure(), ev, phase="train", step_n=2, sample_id="s")
    assert ev.rows[0]["error"] == "generation failed: boom"


def test_harness_stats_counts_kills_and_truncations(tmp_path):
    """The summary must show the kills that happened, so an uncounted one cannot read as none."""
    import json

    from wikitrace.run.storage import _harness_stats

    rows = [
        {"event": "session_cut_short", "stopped_by": "command", "truncated_calls": 0},
        {"event": "session_cut_short", "stopped_by": "command", "truncated_calls": 0},
        {"event": "session_cut_short", "stopped_by": "wall", "truncated_calls": 0},
        # survived its truncation: the cut turn still asked for a tool, so the loop ran on
        {"event": "session_cut_short", "stopped_by": "", "truncated_calls": 2,
         "last_stop_reason": "toolUse", "last_had_tool_call": True},
        # fatal: the last turn hit the budget with nothing to run, so Pi called it done
        {"event": "session_cut_short", "stopped_by": "", "truncated_calls": 1,
         "last_stop_reason": "length", "last_had_tool_call": False},
        {"event": "run_completed"},
    ]
    (tmp_path / "events.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    stats = _harness_stats(tmp_path)
    assert stats["sessions_killed"] == {"command": 2, "wall": 1}
    assert stats["truncated_calls"] == 3
    assert stats["truncated_sessions"] == 2
    assert stats["fatal_truncations"] == 1


def test_harness_stats_is_empty_without_events(tmp_path):
    from wikitrace.run.storage import _harness_stats

    assert _harness_stats(tmp_path) == {
        "sessions_killed": {}, "truncated_calls": 0,
        "truncated_sessions": 0, "fatal_truncations": 0,
    }


# --- a harness-stopped step is COMPILED but never LEARNED FROM -------------------


def _comp(step: str, outcome, *, generation_failed=False):
    from wikitrace.config import Compilation

    return Compilation(
        step_id=step, outcome=outcome, ex=0, soft_f1=0.0,
        generation_failed=generation_failed, title=f"[{outcome.value}] t", body="b",
    )


# --- per-suite hang guards -------------------------------------------------------


def test_other_benchmarks_keep_the_run_level_guards():
    """A benchmark that declares no guard override keeps the run-level values."""
    from wikitrace.envs import resolve

    for domain, benchmark, suite in (
        ("textgame", "alfworld", "valid-seen"),
        ("tau3", "tau3", "retail"),
        ("appworld", "appworld", "test-normal"),
    ):
        spec = resolve(domain, benchmark).benchmark.harness_spec(suite)
        assert (spec.cmd_timeout, spec.max_seconds) == (None, None)


def test_traverse_runner_prefers_the_suite_override_and_falls_back_to_config():
    from wikitrace.envs.base import HarnessSpec
    from wikitrace.run.evaluate import _default_traverse_runner

    class _TC:
        runner = "pi"
        pi_cli = "x"
        provider = "openrouter"
        base_url = ""
        cmd_timeout = 90.0
        max_cmd_kills = 5
        idle_timeout = 120.0
        max_seconds = 1500.0
        sandbox = "none"
        docker_image = "img"
        network_policy = "deny"
        context_level = "level3"

    class _Cfg:
        traverse = _TC()

        class llm:
            openrouter_providers: list[str] = []

        class models:
            class default:
                model = "m"
                reasoning_effort = "high"

    overridden = _default_traverse_runner(
        _Cfg(), None, HarnessSpec(kind="terminal", input_on_disk=False,
                                  cmd_timeout=300.0, max_seconds=3000.0),
    )
    assert (overridden.cmd_timeout, overridden.max_seconds) == (300.0, 3000.0)

    plain = _default_traverse_runner(
        _Cfg(), None, HarnessSpec(kind="terminal", input_on_disk=False),
    )
    assert (plain.cmd_timeout, plain.max_seconds) == (90.0, 1500.0)

    # No harness at all (an injected-runner path) must still construct.
    assert _default_traverse_runner(_Cfg(), None).cmd_timeout == 90.0
