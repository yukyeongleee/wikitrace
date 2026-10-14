"""textgame domain adapter: one Pi agent playing a text environment, scored by the environment."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from wikitrace.config import read_result_json, write_turns
from wikitrace.config import CompilationInputs, Outcome, PromptProfile, ToolStep
from wikitrace.envs.base import BenchmarkAdapter, RawLayerShape, register_domain
from wikitrace.envs.textgame.models import Action, TextGameOutcome, TextGameResult, TextGameSample
from wikitrace.envs.textgame.workflow import ENV_TOOL, run_textgame_episode
from wikitrace.harness.environment import EPISODE_OVER, blocked_by_hook
from wikitrace.harness.prompts import render_template
from wikitrace.harness.terminal import (
    align_calls,
    last_model_returned,
    persisted_turns,
)
from wikitrace.memory.store import RawStepFacts
from wikitrace.memory.store import with_metric_vocabulary

SOLVER_LOG = "solver.full.jsonl"

# The layer STRUCTURE only: metric-naming fields come from the benchmark's `raw_metric_vocabulary`; the simulator's score stays in the run metrics.
_RAW_LAYER_SHAPE = RawLayerShape(
    agent_logs=(SOLVER_LOG,),
    optional_agent_logs=(),
    outcome_files=("metadata.md",),
    step_contents="",
    step_contents_outcome_only="",
    read_hint="Skip the `system` message and the `request_context` row: both are large",
    mount_without_request_context=True,
    prediction_file="metadata.md",
    metadata_description=(
        "One episode: the task, the initial observation, whether the environment reported "
        "it done, and the agent's turn log."
    ),
    metadata_keys=(),
    task_key="instruction",
    outcome_key="success",
    outcome_default="False",
    outcome_success_value="True",
    metadata_artifacts=(
        f"- turn log: `{SOLVER_LOG}` (one JSON turn per line; an `act` call is a command "
        "and its `result` the observation)."
    ),
    index_score="", index_score_pattern="", index_score_columns="", index_metric_names="",
    compilation_layer_desc="", compilation_task_hint="", compilation_outcome_hint="",
    compilation_correct_parts_hint="", compilation_description="",
    compilation_metric_keys=(), compilation_fm_metric_lines=(), writer_metric_header=(),
    env_compilation_sections=True,
    episode=True,
)

_UNRESOLVED_PHRASE = {
    "command_timeout": "no result — this call ran past the per-command time limit",
    "turn_cap": "no result — the session hit its turn limit while this ran",
    "wall_timeout": "no result — the session ran out of wall-clock time while this ran",
    "idle_timeout": "no result — the session went silent while this ran",
    "harness_eof": "no result — the agent process died while this ran; a harness fault",
    "output_truncated": "no result — the turn that issued this call was cut off by the output budget",
    "provider_error": "no result — the model provider's request failed and the session ended; a harness fault",
    "unknown": "no result — the reason was not recorded",
}


LABEL_WIDTH = 6


def _flat(text: str) -> str:
    """One line, whole: a compiler render shows every observation in full."""
    return " ".join((text or "").split())


def _is_act(s: ToolStep) -> bool:
    return s.command.startswith("> ")


def _blocked(s: ToolStep, tool: str = ENV_TOOL) -> bool:
    """An `act` call the one-per-turn hook refused: never a game step, so never rendered."""
    return _is_act(s) and s.is_error and blocked_by_hook(s.result, tool)


def _executed(s: ToolStep) -> bool:
    """An `act` call the server ran: answered, not with an error, and before the episode ended."""
    return _is_act(s) and not s.unresolved and not s.is_error and s.result != EPISODE_OVER


def _served_observations(result: TextGameResult) -> list[Action] | None:
    """The server's actions when they line up one-to-one with the agent's executed `act` calls."""
    n_acts = sum(1 for s in result.tool_steps if _executed(s))
    return result.actions if n_acts == len(result.actions) else None


def _observation(text: str, multiline: bool, width: int = LABEL_WIDTH) -> str:
    """An `obs` row's text: one line, or the observation's own lines under the row's gutter."""
    if not multiline:
        return _flat(text)
    gutter = "\n" + " " * (width + 3) + "│ "
    return gutter.join(line.rstrip() for line in (text or "").strip("\n").splitlines())


def _unserved_observation(s: ToolStep, multiline: bool = False, width: int = LABEL_WIDTH) -> str:
    """What an `act` call the server did not run returned: the reply, or why none came."""
    if s.unresolved:
        return f"⟨{_UNRESOLVED_PHRASE.get(s.unresolved, _UNRESOLVED_PHRASE['unknown'])}⟩"
    return _observation(s.result, multiline, width) or "(no output)"


def render_episode_steps(
    tool_steps: list[ToolStep], served: list[Action] | None, tool: str = ENV_TOOL,
    multiline: bool = False,
) -> str:
    """Reasoning, command and observation per step, every field entire."""
    steps = [s for s in tool_steps if not _blocked(s, tool)]
    width = max(LABEL_WIDTH, len(tool))

    def row(label: str, text: str) -> str:
        return f"  {label.ljust(width)} │ {text}"

    blocks: list[str] = []
    for i, s in enumerate(steps):
        lines = [f"[{i + 1}]" + ("  ⚠ ERROR" if s.is_error and not _is_act(s) else "")]
        if s.reasoning.strip():
            lines.append(row("reason", _flat(s.reasoning)))
        if _is_act(s):
            lines.append(row(tool, s.command[2:]))
            if served is not None and _executed(s):
                a = served[sum(1 for t in steps[:i] if _executed(t))]
                lines.append(row("obs", _observation(a.observation, multiline, width)))
            else:
                lines.append(row("obs", _unserved_observation(s, multiline, width)))
        else:
            lines.append(row("call", _flat(s.command)))
            if s.unresolved:
                lines.append(row("result", f"⟨{_UNRESOLVED_PHRASE.get(s.unresolved, _UNRESOLVED_PHRASE['unknown'])}⟩"))
            else:
                lines.append(row("result", _flat(s.result) or "(no output)"))
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def build_trajectory_text(result: TextGameResult, tool: str = ENV_TOOL, multiline: bool = False) -> str:
    """The episode transcript for the compiler: reasoning, command, observation per step."""
    if not [s for s in result.tool_steps if not _blocked(s, tool)]:
        # No Pi log survived; the server's own record still says what happened.
        lines = [f"> {a.command}\n{a.observation}" for a in result.actions]
        return "## Episode\n" + ("\n".join(lines) or "(no commands recorded)")
    return "## Episode (agent transcript)\n" + render_episode_steps(
        result.tool_steps, _served_observations(result), tool, multiline
    )


def build_outcome_text(result: TextGameResult, steps: str) -> str:
    """The binary outcome and the step count, stated once."""
    lines = [
        f"outcome: {'success' if result.success else 'failure'}",
        f"steps: {steps}",
    ]
    if result.ended_by == "harness_stopped":
        lines.append(
            "note: the harness stopped the session before the episode ended. The agent "
            "did not choose to stop."
        )
    if result.generation_error:
        lines.append(
            f"note: the episode did not run to its end — {result.generation_error}. "
            "A harness outcome, not the agent's."
        )
    return "## Outcome\n" + "\n".join(lines)


def _nested(text: str) -> str:
    """A multi-line value's later lines, each one tab under the bullet or field it opens."""
    first, *rest = text.splitlines() or [""]
    return "\n".join([first, *(f"\t{ln.strip()}" for ln in rest if ln.strip())])


def _metadata_text(result: TextGameResult, steps: str) -> str:
    return (
        f"# {result.sample_id}\n\n"
        f"- task_type: {result.task_type}\n"
        f"- instruction: {result.question}\n"
        f"- initial_observation: {_nested(result.initial_observation)}\n"
        f"- success: {result.success}\n"
        f"- steps: {steps}\n"
    )


def _write_solver_log(step_dir: Path, log: Any, *, final_message: bool = True) -> None:
    """The Pi session as the shared turn-stream layout, so every wiki reader can open it."""
    fc_turns = [t for t in log.turns if t.get("type") == "function_call"]
    for t, p in zip(fc_turns, align_calls(fc_turns, log.final_context)):
        if p["reasoning"]:
            t["reasoning"] = p["reasoning"]
        if (out := p["result"]) is not None:
            t["result"] = out["text"]
            t["is_error"] = out["is_error"]
            if out["state"]:
                t["result_state"] = out["state"]
    turns: list[dict] = [
        {"type": "message", "role": "system", "content": log.system_prompt},
        {"type": "message", "role": "user", "content": log.user_prompt},
        *persisted_turns(log.turns),
        {"type": "request_context", "messages": log.final_context},
        *([{"type": "message", "role": "assistant", "content": log.final_text,
            "reasoning": log.final_reasoning}] if final_message else []),
        {"type": "answer", "role": "assistant", "model_requested": None,
         "model_returned": last_model_returned(log.usage_calls), "parsed": None,
         "usage": dict(log.usage), "read_set": log.read_set},
    ]
    for i, t in enumerate(turns):
        t["turn"] = i
    write_turns(step_dir / f"{log.phase}.full.jsonl", turns)


class TextGameDomainAdapter:
    """One Pi agent per episode, acting on a host-side simulator through the `act` tool."""

    name = "textgame"
    prediction_filename = "metadata.md"
    raw_entry_source = "question"
    compilation_entry_source = "title"

    def __init__(self, benchmark: BenchmarkAdapter) -> None:
        self.benchmark = benchmark

    @property
    def raw_layer_shape(self) -> RawLayerShape:
        shape = with_metric_vocabulary(_RAW_LAYER_SHAPE, self.benchmark.raw_metric_vocabulary)
        tool = self.benchmark.env_tool
        if tool == ENV_TOOL:
            return shape
        return replace(shape, metadata_artifacts=(
            f"- turn log: `{SOLVER_LOG}` (one JSON turn per line; a `{tool}` call is an action "
            "and its `result` the observation)."
        ))

    def compiler_world_reference(self, corpus_dir: Path) -> str:
        return ""

    def prompt_profile(self, suite_id: str) -> PromptProfile:
        b = self.benchmark
        return PromptProfile(
            task_kind=b.task_kind,
            solver_prompt=render_template(
                "envs/textgame/prompts/solver/excerpt.j2", intro=b.solver_intro,
                env_tool=b.env_tool,
            ),
            has_evidence=b.has_evidence,
        )


    def run_traverse(self, sample: TextGameSample, runner: Any, corpus_dir: Path | None,
                     session_root: Path, **kwargs: Any) -> TextGameOutcome:
        # The turn cap is the suite's, not the config's: it must clear the action budget.
        kwargs["max_turns"] = self.benchmark.harness_spec(sample.db_id).max_turns
        return run_textgame_episode(sample, runner, corpus_dir, session_root, bench=self.benchmark, **kwargs)


    def failed_traverse_outcome(
        self, sample: TextGameSample, error: str, phase_logs: Any = None,
    ) -> TextGameOutcome:
        result = TextGameResult(
            sample_id=sample.id, db_id=sample.db_id, question=sample.task_text,
            task_type=sample.task_type, variant=sample.variant, meta=dict(sample.meta),
            step_limit=self.benchmark.step_limit, generation_error=f"episode failed: {error}",
        )
        return TextGameOutcome(result=result, phase_logs=list(phase_logs or []), generation_failed=True)


    def persist_traverse_trajectory(
        self, step_dir: Path, sample: TextGameSample, outcome: TextGameOutcome
    ) -> None:
        step_dir.mkdir(parents=True, exist_ok=True)
        result = outcome.result
        for log in outcome.phase_logs:
            _write_solver_log(step_dir, log)
        (step_dir / "metadata.md").write_text(
            _metadata_text(result, self.benchmark.steps_text(result)), encoding="utf-8"
        )

    def persist_train_task_files(self, step_dir: Path, sample: TextGameSample) -> None:
        return None

    def render_trajectory_for_compiler(self, result: TextGameResult) -> str:
        return build_trajectory_text(result, self.benchmark.env_tool, self.benchmark.multiline_observations)

    def render_outcome_for_compiler(self, result: TextGameResult, *, sample: TextGameSample | None = None) -> str:
        return build_outcome_text(result, self.benchmark.steps_text(result))

    def compilation_inputs(
        self, result: TextGameResult, *, sample: TextGameSample | None = None
    ) -> CompilationInputs:
        return CompilationInputs(
            sample_id=result.sample_id, db_id=result.db_id, success=result.success,
            metrics={"success": float(result.success), "steps": float(result.steps),
                     "step_limit_hit": float(result.step_limit_hit)},
            question=result.question,
            prediction=f"success={int(result.success)}; steps={self.benchmark.steps_text(result)}",
            generation_failed=bool(result.generation_error),
            task_fields=(("task_type", result.task_type), ("instruction", result.question),
                         ("initial_observation", _nested(result.initial_observation))),
        )

    def sample_row(self, result: TextGameResult, pred_path: str) -> dict[str, Any]:
        return {
            "episode_path": pred_path,
            "task_type": result.task_type,
            "variant": result.variant,
            "success": result.success,
            "score": result.score,
            "failed": result.failed,
            "steps": result.steps,
            "step_limit_hit": result.step_limit_hit,
            "invalid_commands": result.invalid_commands,
            "ended_by": result.ended_by,
            **result.meta,
        }


    def read_result(self, step_dir: Path) -> TextGameResult:
        return TextGameResult.model_validate_json(read_result_json(step_dir))


    def read_raw_step(self, step_dir: Path) -> RawStepFacts:
        result = self.read_result(step_dir)
        return RawStepFacts(
            outcome=(Outcome.SUCCESS if result.success else Outcome.FAILURE).value,
            tail=result.question, metrics={"success": float(result.success)},
        )


register_domain("textgame", TextGameDomainAdapter)
