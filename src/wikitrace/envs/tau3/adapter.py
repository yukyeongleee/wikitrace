"""tau3 domain adapter: the textgame persistence and renders over a tau3 conversation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from wikitrace.config import read_result_json
from wikitrace.config import CompilationInputs, PromptProfile, ToolStep
from wikitrace.envs.base import RawLayerShape, register_domain
from wikitrace.envs.tau3.benchmarks.tau3 import read_prompt_data
from wikitrace.envs.tau3.models import TauAction, TauBenchResult, TauMessage
from wikitrace.envs.tau3.workflow import (
    command_text,
    conversation,
    run_tau3_episode,
)
from wikitrace.envs.textgame.adapter import (
    SOLVER_LOG,
    TextGameDomainAdapter,
    _blocked,
    _executed,
    _flat,
    _is_act,
    _nested,
    _observation,
    _UNRESOLVED_PHRASE,
    _unserved_observation,
    _write_solver_log,
)
from wikitrace.envs.textgame.models import TextGameOutcome, TextGameSample
from wikitrace.harness.prompts import render_template

EXCERPT = "envs/tau3/prompts/solver/excerpt.j2"
TOOL = "take_action"


def _obs_label(source: str) -> str:
    """`obs(user)` for a simulator reply, `obs(env)` for a tool output or an error, bare `obs` unmarked."""
    return f"obs({source})" if source in ("user", "env") else "obs"


def render_conversation_steps(
    tool_steps: list[ToolStep], served: list[TauAction] | None, messages: Sequence[TauMessage] = (),
) -> str:
    """Reasoning, action and observation per step; a message to the user is a step of its own."""
    width = len(TOOL)

    def row(label: str, text: str) -> str:
        return f"  {label.ljust(width)} │ {text}"

    blocks: list[str] = []
    executed = 0
    for s in conversation(tool_steps, messages):
        if isinstance(s, TauMessage):
            lines = [f"[{len(blocks) + 1}]"]
            if s.reasoning.strip():
                lines.append(row("reason", _flat(s.reasoning)))
            lines.append(row("message", _observation(s.text, True, width)))
            if s.reply is not None:
                lines.append(row(_obs_label("user"), _observation(s.reply, True, width)))
            blocks.append("\n".join(lines))
            continue
        if _blocked(s, TOOL):
            continue
        lines = [f"[{len(blocks) + 1}]" + ("  ⚠ ERROR" if s.is_error and not _is_act(s) else "")]
        if s.reasoning.strip():
            lines.append(row("reason", _flat(s.reasoning)))
        if _is_act(s):
            lines.append(row(TOOL, _observation(command_text(s.command[2:]), True, width)))
            if served is not None and _executed(s):
                a = served[executed]
                lines.append(row(_obs_label(a.source), _observation(a.observation, True, width)))
            else:
                lines.append(row(_obs_label(""), _unserved_observation(s, True, width)))
        else:
            lines.append(row("call", _flat(s.command)))
            if s.unresolved:
                lines.append(row("result", f"⟨{_UNRESOLVED_PHRASE.get(s.unresolved, _UNRESOLVED_PHRASE['unknown'])}⟩"))
            else:
                lines.append(row("result", _flat(s.result) or "(no output)"))
        executed += _executed(s)
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _served(result: TauBenchResult, tool_steps: list[ToolStep]) -> list[TauAction] | None:
    """The server's actions when they line up one-to-one with the executed `take_action` calls."""
    n_acts = sum(1 for s in tool_steps if _executed(s))
    return result.actions if n_acts == len(result.actions) else None


def build_trajectory_text(result: TauBenchResult) -> str:
    """The conversation for the compiler: reasoning, action and observation per step."""
    if not result.messages and not [s for s in result.tool_steps if not _blocked(s, TOOL)]:
        lines = [f"> {command_text(a.command)}\n{a.observation}" for a in result.actions]
        return "## Episode\n" + ("\n".join(lines) or "(no actions recorded)")
    return "## Episode (agent transcript)\n" + render_conversation_steps(
        result.tool_steps, _served(result, result.tool_steps), result.messages
    )


def _repeats_last_message(log: Any) -> bool:
    """Whether the final reply is the last message the channel already recorded for the user."""
    sent = [t for t in log.turns if t.get("type") == "message" and t.get("role") == "assistant"]
    return bool(sent) and sent[-1].get("content") == log.final_text


def _metadata_text(result: TauBenchResult, steps: str) -> str:
    # Only an episode whose memory was bound at reset has these: the query and what it selected.
    memory = "".join(
        f"- {k}: {_nested(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))}\n"
        for k, v in result.meta.items() if k.startswith("memory_")
    )
    return (
        f"# {result.sample_id}\n\n"
        f"- task_type: {result.task_type}\n"
        f"- user_request: {_nested(result.question)}\n"
        f"- success: {result.success}\n"
        f"- steps: {steps}\n"
        f"{memory}"
    )


class TauBenchDomainAdapter(TextGameDomainAdapter):
    """One Pi agent per conversation, acting on the retail environment through `take_action`."""

    name = "tau3"

    @property
    def raw_layer_shape(self) -> RawLayerShape:
        return replace(
            super().raw_layer_shape,
            task_key="user_request",
            metadata_description=(
                "One conversation: the user's first message as the task, whether the authors' "
                "evaluator awarded the full reward, and the agent's turn log."
            ),
            metadata_artifacts=(
                f"- turn log: `{SOLVER_LOG}` (one JSON turn per line; a `take_action` call is a "
                "retail tool call and its `result` the tool output; an assistant `message` row is a "
                "message to the user, and the user `message` row after it the reply)."
            ),
            turn_log_text=(
                "Each `solver.full.jsonl` is one JSON row per line. The conversation's steps are "
                "the `function_call` rows, each with the call's `arguments`, the agent's `reasoning` "
                "and the tool's `result`, and the assistant `message` rows, each a message to the "
                "user with the agent's `reasoning`. The user `message` row after one holds the "
                "user's reply."
            ),
        )

    def prompt_profile(self, suite_id: str) -> PromptProfile:
        profile = super().prompt_profile(suite_id)
        return profile.model_copy(update={"solver_prompt": render_template(EXCERPT, prompt_data=read_prompt_data())})

    def run_traverse(self, sample: TextGameSample, runner: Any, corpus_dir: Path | None,
                     session_root: Path, **kwargs: Any) -> TextGameOutcome:
        # The turn cap is the suite's, not the config's: it must clear the conversation budget.
        kwargs["max_turns"] = self.benchmark.harness_spec(sample.db_id).max_turns
        return run_tau3_episode(sample, runner, corpus_dir, session_root, bench=self.benchmark, **kwargs)

    def persist_traverse_trajectory(
        self, step_dir: Path, sample: TextGameSample, outcome: TextGameOutcome
    ) -> None:
        step_dir.mkdir(parents=True, exist_ok=True)
        for log in outcome.phase_logs:
            _write_solver_log(step_dir, log, final_message=not _repeats_last_message(log))
        (step_dir / "metadata.md").write_text(
            _metadata_text(outcome.result, self.benchmark.steps_text(outcome.result)), encoding="utf-8"
        )

    def read_result(self, step_dir: Path) -> TauBenchResult:
        return TauBenchResult.model_validate_json(read_result_json(step_dir))

    def render_trajectory_for_compiler(self, result: TauBenchResult) -> str:
        return build_trajectory_text(result)

    def compilation_inputs(
        self, result: TauBenchResult, *, sample: TextGameSample | None = None
    ) -> CompilationInputs:
        ci = super().compilation_inputs(result, sample=sample)
        return replace(ci, task_fields=(("task_type", result.task_type), ("user_request", _nested(result.question))))


register_domain("tau3", TauBenchDomainAdapter)
