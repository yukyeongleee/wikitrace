"""appworld domain adapter: one Pi agent driving an AppWorld task shell through `exec`."""

from __future__ import annotations

import re

from pathlib import Path
from typing import Any

from wikitrace.config import read_result_json, write_turns
from wikitrace.config import CompilationInputs, Outcome, PromptProfile, ToolStep, TraverseOutcome
from wikitrace.envs.appworld.benchmarks.appworld import AppWorldResult, AppWorldSample
from wikitrace.envs.appworld.workflow import (
    ENV_TOOL, authors_instructions, run_appworld_episode,
)
from wikitrace.envs.base import BenchmarkAdapter, RawLayerShape, register_domain, steps_text
from wikitrace.envs.textgame.adapter import (
    _UNRESOLVED_PHRASE,
    _flat,
)
from wikitrace.harness.environment import blocked_by_hook
from wikitrace.harness.prompts import render_template
from wikitrace.harness.terminal import (
    align_calls,
    last_model_returned,
    persisted_turns,
)
from wikitrace.memory.store import RawStepFacts
from wikitrace.memory.store import with_metric_vocabulary

_STEP_DIR = "/work/corpus/raw/step-NNNN"
SOLVER_LOG = "solver.full.jsonl"

# A raw step exposes the binary outcome alone; the unit-test counts stay in the run metrics.
_RAW_LAYER_SHAPE = RawLayerShape(
    compilation_metric_keys=(),
    compilation_fm_metric_lines=(),
    writer_metric_header=(),
    compilation_task_hint="instruction, supervisor, world_datetime",
    agent_logs=(SOLVER_LOG,),
    optional_agent_logs=(),
    outcome_files=("metadata.md",),
    step_contents=(
        f"Each `{_STEP_DIR}/` holds: `metadata.md` (sample id, instruction, supervisor,\n"
        "world_datetime, success, steps); the agent turn log `solver.full.jsonl` (an `exec`\n"
        "call is a code chunk and its `result` the output)."
    ),
    step_contents_outcome_only=(
        f"Each `{_STEP_DIR}/` holds: `metadata.md` (sample id, instruction, supervisor,\n"
        "world_datetime, success, steps)."
    ),
    read_hint="Skip the `system` message and the `request_context` row: both are large",
    mount_without_request_context=True,
    prediction_file="metadata.md",
    metadata_description=(
        "One task's episode: the instruction, the supervisor and the in-world clock, "
        "whether the task was completed, and the agent's turn log."
    ),
    metadata_keys=("supervisor", "world_datetime", "success", "steps"),
    task_key="instruction",
    outcome_key="success",
    outcome_default="False",
    outcome_success_value="True",
    metadata_artifacts=(
        "- turn log: `solver.full.jsonl` (one JSON turn per line; an `exec` call is a code "
        "chunk and its `result` the output)."
    ),
    metadata_body_sections=(),
    index_score="",
    index_score_pattern="",
    index_score_columns="",
    index_metric_names="success",
    compilation_layer_desc="(outcome, one-line summary)",
    compilation_outcome_hint="what success does/doesn't establish (no partial credit)",
    compilation_correct_parts_hint=(
        "(failure only) parts of the target state the outputs show reached (no "
        "partial-credit score — judge from the outputs)"
    ),
    compilation_description="{outcome} analysis.",
    env_compilation_sections=True,
    episode=True,
)


def _outcome_label(result: AppWorldResult) -> str:
    return (Outcome.SUCCESS if result.success else Outcome.FAILURE).value


def _steps_text(result: AppWorldResult) -> str:
    return steps_text(result.steps, result.ended_by, tool="exec")


def _blocked(s: ToolStep) -> bool:
    """An `exec` call the one-per-turn hook refused: never reached the shell, so never a step."""
    return s.is_error and blocked_by_hook(s.result, ENV_TOOL)


def _render_tool_steps(steps: list[ToolStep]) -> str:
    """The reason/call/result triples of a whole episode, every field entire."""
    steps = [s for s in steps if not _blocked(s)]
    blocks: list[str] = []
    for i, s in enumerate(steps):
        lines = [f"[{i + 1}]" + ("  ⚠ ERROR" if s.is_error else "")]
        if s.reasoning.strip():
            lines.append(f"  reason │ {_flat(s.reasoning)}")
        lines.append(f"  call   │ {_flat(s.command)}")
        if s.unresolved:
            lines.append(f"  result │ ⟨{_UNRESOLVED_PHRASE.get(s.unresolved, _UNRESOLVED_PHRASE['unknown'])}⟩")
        else:
            lines.append(f"  result │ {_flat(s.result) or '(no output)'}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def build_trajectory_text(result: AppWorldResult) -> str:
    """The episode as the compiler reads it: the agent's tool calls with their outputs."""
    if not result.tool_steps:
        return "(no agent steps recorded)"
    return "## Solver (terminal)\n" + _render_tool_steps(result.tool_steps)


def build_outcome_text(result: AppWorldResult) -> str:
    """The outcome block: the binary outcome and the step count, never the tests."""
    lines = [
        "## Outcome",
        f"outcome: {_outcome_label(result)}",
        f"steps: {_steps_text(result)}",
    ]
    if result.ended_by == "harness_stopped":
        lines.append(
            "note: the harness stopped the session before the episode ended. The agent "
            "did not choose to stop."
        )
    if result.generation_error:
        # A provider error ends the session early, but the episode it left was still graded.
        unscored = not result.generation_error.startswith("provider_error:")
        lines.append(
            f"note: the episode {'was never scored' if unscored else 'did not run to its end'} "
            f"— {result.generation_error}. "
            "A harness outcome, not the agent's."
        )
    return "\n".join(lines)


def _supervisor_text(result: AppWorldResult) -> str:
    s = result.supervisor
    return (
        f"{s.get('first_name', '')} {s.get('last_name', '')} <{s.get('email', '')}>, "
        f"phone {s.get('phone_number', '')}"
    ) if s else ""


def _metadata_text(result: AppWorldResult) -> str:
    return (
        f"# {result.sample_id}\n\n"
        f"- instruction: {result.question}\n"
        f"- supervisor: {_supervisor_text(result)}\n"
        f"- world_datetime: {result.world_datetime}\n"
        f"- success: {result.success}\n"
        f"- steps: {_steps_text(result)}\n"
    )


def persist_traverse_trajectory(
    step_dir: Path, sample: AppWorldSample, outcome: TraverseOutcome
) -> None:
    """Dump one episode: the Pi turn stream and the metadata."""
    step_dir.mkdir(parents=True, exist_ok=True)
    result = outcome.result
    for log in outcome.phase_logs:
        fc_turns = [t for t in log.turns if t.get("type") == "function_call"]
        for t, p in zip(fc_turns, align_calls(fc_turns, log.final_context)):
            if p["reasoning"]:
                t["reasoning"] = p["reasoning"]
            out = p["result"]
            if out is not None:
                t["result"] = out["text"]
                t["is_error"] = out["is_error"]
                if out["state"]:
                    t["result_state"] = out["state"]
        turns: list[dict] = [
            {"type": "message", "role": "system", "content": log.system_prompt},
            {"type": "message", "role": "user", "content": log.user_prompt},
            *persisted_turns(log.turns),
            {"type": "request_context", "messages": log.final_context},
            {"type": "message", "role": "assistant", "content": log.final_text,
             "reasoning": log.final_reasoning},
            {"type": "answer", "role": "assistant",
             "model_requested": None, "model_returned": last_model_returned(log.usage_calls),
             "parsed": None, "usage": dict(log.usage), "read_set": log.read_set},
        ]
        for i, t in enumerate(turns):
            t["turn"] = i
        write_turns(step_dir / f"{log.phase}.full.jsonl", turns)
    (step_dir / "metadata.md").write_text(_metadata_text(result), encoding="utf-8")


_APP_CALL = re.compile(r"\bapis\.(\w+)\.")
# Every run calls these, so as a tag they would match everything.
_UNTAGGED_APPS = frozenset({"api_docs", "supervisor"})


class AppWorldDomainAdapter:
    """One Pi agent per AppWorld task, its shell reached through the `exec` tool."""

    name = "appworld"

    @staticmethod
    def called_apps(result: Any) -> tuple[str, ...]:
        """The apps a run's executed code called, sorted, without the ones every run calls."""
        commands = [s.command[2:] if s.command.startswith("> ") else s.command
                    for s in result.tool_steps if not _blocked(s)]
        found = {m for c in commands for m in _APP_CALL.findall(c)}
        return tuple(sorted(found - _UNTAGGED_APPS))
    prediction_filename = "metadata.md"
    raw_entry_source = "question"
    compilation_entry_source = "title"

    def __init__(self, benchmark: BenchmarkAdapter) -> None:
        self.benchmark = benchmark

    @property
    def raw_layer_shape(self) -> RawLayerShape:
        return with_metric_vocabulary(
            _RAW_LAYER_SHAPE, getattr(self.benchmark, "raw_metric_vocabulary", None)
        )

    def compiler_world_reference(self, corpus_dir: Path) -> str:
        return ""

    def prompt_profile(self, suite_id: str) -> PromptProfile:
        b = self.benchmark
        intro, key_instructions = authors_instructions()
        return PromptProfile(
            task_kind=b.task_kind,
            solver_prompt=render_template(
                "envs/appworld/prompts/solver/excerpt.j2",
                intro=intro, key_instructions=key_instructions,
            ),
            has_evidence=b.has_evidence,
        )


    def run_traverse(self, sample: AppWorldSample, runner: Any, corpus_dir: Path,
                     session_root: Path, **kwargs: Any) -> TraverseOutcome:
        # The turn cap is the suite's, not the config's: it must clear the action budget.
        kwargs["max_turns"] = self.benchmark.harness_spec(sample.db_id).max_turns
        return run_appworld_episode(
            sample, runner, corpus_dir, session_root,
            **kwargs,
        )

    def _failed(self, sample: AppWorldSample, error: str) -> AppWorldResult:
        return AppWorldResult(
            sample_id=sample.id, db_id=sample.db_id, question=sample.question,
            scenario_id=sample.scenario_id, difficulty=sample.difficulty,
            supervisor=dict(sample.supervisor), world_datetime=sample.world_datetime,
            generation_error=error,
        )


    def failed_traverse_outcome(
        self, sample: AppWorldSample, error: str, phase_logs: Any = None,
    ) -> TraverseOutcome:
        return TraverseOutcome(
            result=self._failed(sample, f"episode failed: {error}"),
            phase_logs=list(phase_logs or []), generation_failed=True,
        )


    def persist_traverse_trajectory(
        self, step_dir: Path, sample: AppWorldSample, outcome: TraverseOutcome
    ) -> None:
        persist_traverse_trajectory(step_dir, sample, outcome)

    def persist_train_task_files(self, step_dir: Path, sample: AppWorldSample) -> None:
        """No task files: the task is its instruction, and the apps live in the simulator."""
        return None

    def render_trajectory_for_compiler(self, result: AppWorldResult) -> str:
        return build_trajectory_text(result)

    def render_outcome_for_compiler(
        self, result: AppWorldResult, *, sample: AppWorldSample | None = None
    ) -> str:
        return build_outcome_text(result)

    def compilation_inputs(
        self, result: AppWorldResult, *, sample: AppWorldSample | None = None
    ) -> CompilationInputs:
        return CompilationInputs(
            sample_id=result.sample_id, db_id=result.db_id, success=result.success,
            metrics={"success": float(result.success), "steps": float(result.steps),
                     "step_limit_hit": float(result.step_limit_hit)},
            question=result.question,
            prediction=f"success={int(result.success)}; "
                       f"steps={_steps_text(result)}",
            generation_failed=bool(result.generation_error),
            task_fields=(("instruction", result.question), ("supervisor", _supervisor_text(result)),
                         ("world_datetime", result.world_datetime)),
        )

    def sample_row(self, result: AppWorldResult, pred_path: str) -> dict[str, Any]:
        return {
            "episode_path": pred_path,
            "scenario_id": result.scenario_id,
            "difficulty": result.difficulty,
            "tgc": result.tgc,
            "tests_passed": result.tests_passed,
            "tests_total": result.tests_total,
            "steps": result.steps,
            "step_limit_hit": result.step_limit_hit,
            "invalid_commands": result.invalid_commands,
            "ended_by": result.ended_by,
            # Full report, failing tests included; build_outcome_text keeps it from the compiler.
            "report": result.report,
        }


    def read_result(self, step_dir: Path) -> AppWorldResult:
        return AppWorldResult.model_validate_json(read_result_json(step_dir))


    def read_raw_step(self, step_dir: Path) -> RawStepFacts:
        result = self.read_result(step_dir)
        return RawStepFacts(
            outcome=_outcome_label(result),
            tail=result.question,
            metrics={"success": float(result.success)},
        )


register_domain("appworld", AppWorldDomainAdapter)
