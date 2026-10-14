"""Domain and benchmark adapter contracts and their registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from wikitrace.config import ConfigView
from wikitrace.config import CompilationInputs, PromptProfile

from wikitrace.memory.store import (
    RawLayerShape,
    RawMetricVocabulary,
    with_task_context,
    with_task_files,
)
from wikitrace.memory.store import RawStepFacts

__all__ = ["CompilationInputs", "PromptProfile", "RawLayerShape", "RawMetricVocabulary"]


# A Pi guard's kill reason in the compiler-facing vocabulary of ``ToolStep.unresolved``.
_KILL_TO_UNRESOLVED = {
    "command": "command_timeout",
    # Repeated per-command force-kills read to the compiler as the same cause as one.
    "cmd_kill_budget": "command_timeout",
    "wall": "wall_timeout",
    "idle": "idle_timeout",
    "turn_cap": "turn_cap",
    "eof": "harness_eof",
}


def _unresolved_reason(log: Any) -> str:
    """Why a call in this phase log would have no result. See ``ToolStep.unresolved``."""
    if reason := _KILL_TO_UNRESOLVED.get(getattr(log, "stopped_by", "")):
        return reason
    if getattr(log, "provider_error", ""):
        return "provider_error"
    if getattr(log, "last_stop_reason", "") == "length":
        return "output_truncated"
    return "unknown"


@dataclass(frozen=True)
class HarnessSpec:
    """Which machinery ONE SUITE needs, resolved per suite id."""
    # ``kind`` and ``input_on_disk`` are INDEPENDENT: an agent can need a shell while its input ships inline.

    kind: Literal["chat", "terminal", "environment"]
    input_on_disk: bool
    allowed_hosts: tuple[str, ...] = ()
    # Per-suite overrides of the run-level hang guards (``None`` keeps the config value): a cap that bites censors every method.
    cmd_timeout: float | None = None
    max_seconds: float | None = None
    # A turn is one provider request and may carry read/bash calls: a session guard, not the action budget.
    max_turns: int | None = None
    reference_dbs: bool = False
    # The task text is the environment's first message, so an method that queries memory by it binds inside the episode.
    task_text_at_reset: bool = False

    @property
    def terminal(self) -> bool:
        """The generator is the Pi agent, with or without an environment behind it."""
        return self.kind != "chat"

    @property
    def environment(self) -> bool:
        return self.kind == "environment"


@dataclass(frozen=True)
class MemoryBinding:
    """What an method hands one test episode: the blocks that open and close its task message."""

    memory_text: str = ""
    user_prefix: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class Metrics(BaseModel):
    """Benchmark-agnostic per-sample scores."""

    primary: Optional[float] = None
    secondary: Optional[float] = None
    success: bool = False
    refused: bool = False
    finished: bool = True
    error: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


def neutral_summary(metrics: list[Metrics]) -> dict[str, Any]:
    """The DOMAIN-NEUTRAL headline every benchmark's ``summarize`` carries."""
    # Three views, since a refusal is not a wrong answer; ``secondary_mean`` is None, not 0.0, without a secondary metric.
    n = len(metrics)
    n_refused = sum(1 for m in metrics if m.refused)
    n_correct = sum(1 for m in metrics if m.success)
    n_answered = n - n_refused
    secondary = [m.secondary for m in metrics if m.secondary is not None]
    return {
        "n": n,
        "success_rate": (n_correct / n) if n else 0.0,
        "success_rate_answered": (n_correct / n_answered) if n_answered else 0.0,
        "refusal_rate": (n_refused / n) if n else 0.0,
        "secondary_mean": (sum(secondary) / n) if (secondary and n) else None,
    }


WIKI_ROLES: tuple[str, ...] = ("compiler_success_nomem", "compiler_failure_nomem")


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """Data + gold + evaluator for one benchmark."""

    name: str
    domain: str
    primary_metric_name: str
    secondary_metric_name: str
    # Only these keys are recorded and hashed, so another benchmark's keys cannot move this one's cohort.
    config_dataset_keys: tuple[str, ...]
    config_roles: tuple[str, ...]
    has_evidence: bool

    def carries_task_context(self, suite_id: str) -> bool:
        """Whether this suite's frozen steps hold a ``context.md``."""
        return False

    def harness_spec(self, suite_id: str) -> HarnessSpec:
        """Which machinery ``suite_id`` needs (see :class:`HarnessSpec`)."""
        # Must be TOTAL: the runner validates delivery before ``load_suite`` sees the id, so an unknown suite gets the conservative default.
        ...

    def load_suite(self, dataset_cfg: Any, db_id: str) -> list[Any]:
        """Load one suite's samples (ordering is the split's contract)."""
        ...

    def score(self, sample: Any, prediction: str) -> Any:
        """Score one prediction against the hidden gold (domain-typed result)."""
        ...


    def grade(self, sample: Any, result: Any, llm: Any = None) -> None:
        """Score a generated result IN PLACE, where the workflow cannot score it."""
        # Most benchmarks score when the generator returns, so ``grade`` is a no-op; one needing a validator or judge grades here.
        ...

    def metrics_from_result(self, result: Any) -> Metrics:
        """Map a domain result onto generic :class:`Metrics`."""
        ...

    def summarize(self, metrics: list[Metrics]) -> dict[str, Any]:
        """The ``test`` section of ``metrics/summary.json``."""
        ...


@runtime_checkable
class DomainAdapter(Protocol):
    """Workflow + trajectory shape for one domain, bound to a benchmark."""

    name: str
    benchmark: BenchmarkAdapter
    prediction_filename: str
    # Which Compilation field each layer index uses as an entry's tail, a plain string because `wiki/` cannot import this package.
    raw_entry_source: Literal["question", "description", "title"]
    compilation_entry_source: Literal["question", "description", "title"]
    # Must agree with what persistence writes: the agent opens exactly the paths and metric keys named here.

    raw_layer_shape: RawLayerShape

    def prompt_profile(self, suite_id: str) -> PromptProfile:
        """This domain's prompt prose, composed with its benchmark's metric pieces."""
        ...

    def compiler_world_reference(self, corpus_dir: Path) -> str:
        """Standing REFERENCE MATERIAL the generator's agents were given, for the compiler."""
        ...


    def run_traverse(self, sample: Any, runner: Any, corpus_dir: Path,
                     session_root: Path, **kwargs: Any) -> Any:
        """Terminal-agent (traverse) run on one sample -> domain outcome."""
        ...


    def failed_traverse_outcome(
        self, sample: Any, error: str, phase_logs: Any = None,
    ) -> Any:
        """Degenerate zero-score traverse outcome."""
        ...


    def persist_traverse_trajectory(self, step_dir: Path, sample: Any, outcome: Any) -> None:
        """Dump one traverse trajectory's raw artifacts under ``step_dir``."""
        ...

    def persist_train_task_files(self, step_dir: Path, sample: Any) -> None:
        """Copy a TRAIN sample's on-disk task-input files into ``step_dir/files/``."""
        # Gated on the TERMINAL axis, not the mount flag, or a filed-but-unmounted suite points at a file the corpus never got.
        ...

    def render_trajectory_for_compiler(self, result: Any) -> str:
        """The agent-steps transcript block of the compiler prompt."""
        ...

    def render_outcome_for_compiler(self, result: Any, *, sample: Any = None) -> str:
        """The execution-outcome (+ env feedback / output preview) block."""
        ...

    def compilation_inputs(self, result: Any, *, sample: Any) -> CompilationInputs:
        """Map a domain result onto the compiler's generic input slots."""
        # REQUIRED, not defaulted: a domain that omitted it would satisfy the Protocol and silently render empty options.
        ...

    def sample_row(self, result: Any, pred_path: str) -> dict[str, Any]:
        """Prediction-specific fields of a ``metrics/samples.jsonl`` row."""
        ...

    def read_result(self, step_dir: Path) -> Any:
        """Parse a frozen ``raw/step-NNNN/result.json`` into the domain's result model."""
        ...


    def read_raw_step(self, step_dir: Path) -> RawStepFacts:
        """Read one frozen ``raw/step-NNNN/`` back into its index-entry facts."""

        ...


def train_group_for(bench: "BenchmarkAdapter", suite_id: str) -> str:
    """The train set ``suite_id`` draws from — the key everything shareable is keyed on."""
    group_of = getattr(bench, "train_group", None)
    return group_of(suite_id) if group_of is not None else suite_id


def config_view_for(bench: "BenchmarkAdapter") -> ConfigView:
    """The slice of the config ``bench``'s runs record and fingerprint."""
    return ConfigView(dataset_keys=bench.config_dataset_keys, roles=bench.config_roles,
                      omit_at_default=getattr(bench, "config_omit_at_default", ()))


# The wiki-facing phrase per `harness.environment.ENDED_BY` value the task did not end; it names who ended the episode, nothing the agent never saw.
_EPISODE_END_PHRASE = {
    "task_failed": "ended by the environment: an action failed the task",
    "budget_exhausted": "ended by the harness: {tool} budget ran out",
    "game_time_exhausted": "ended by the environment: its step clock ran out",
    "agent_stopped": "ended by the agent",
    "harness_stopped": "ended by the harness: session cut off",
    "simulator_error": "ended by a simulator crash, not the agent",
}


def steps_text(steps: int, ended_by: str, *, tool: str, end_phrase: str = "") -> str:
    """The `steps` bullet of an environment episode: tool calls, plus what ended it if not the task."""
    phrase = end_phrase or _EPISODE_END_PHRASE.get(ended_by)
    if phrase is None:
        return f"{steps} {tool} calls"
    return f"{steps} {tool} calls ({phrase.format(tool=tool)})"


def carries_task_files(harness: HarnessSpec) -> bool:
    """Whether this suite's frozen ``raw/step-NNNN/`` dirs carry a ``files/`` subdir."""
    # THE gate for whether a train step's files are copied into the corpus and whether its map says so; never the solver's mount flag.
    return harness.kind == "terminal"


def raw_shape_for(dom: "DomainAdapter", suite_id: str) -> RawLayerShape:
    """The raw-layer shape to build AND describe this suite's corpus with."""
    # Domain supplies the layer structure, benchmark whether steps carry files; one entry point, as an unmarked shape drops them silently.
    shape = dom.raw_layer_shape
    if carries_task_files(dom.benchmark.harness_spec(suite_id)):
        shape = with_task_files(shape)
    if dom.benchmark.carries_task_context(suite_id):
        shape = with_task_context(shape)
    return shape


_DOMAINS: dict[str, Callable[[BenchmarkAdapter], DomainAdapter]] = {}
_BENCHMARKS: dict[str, Callable[[], BenchmarkAdapter]] = {}


def register_domain(name: str, factory: Callable[[BenchmarkAdapter], DomainAdapter]) -> None:
    if name in _DOMAINS:
        raise ValueError(f"domain already registered: {name}")
    _DOMAINS[name] = factory


def register_benchmark(name: str, factory: Callable[[], BenchmarkAdapter]) -> None:
    if name in _BENCHMARKS:
        raise ValueError(f"benchmark already registered: {name}")
    _BENCHMARKS[name] = factory


def resolve(
    domain: str, benchmark: str, *, include_evidence: bool = True
) -> DomainAdapter:
    """Resolve ``(domain, benchmark)`` to a domain adapter bound to its benchmark."""
    # ``include_evidence=False`` flips the channel here, not in each consumer; a fresh benchmark instance keeps the mutation local.
    if benchmark not in _BENCHMARKS:
        raise KeyError(f"unknown benchmark: {benchmark!r} (have {sorted(_BENCHMARKS)})")
    if domain not in _DOMAINS:
        raise KeyError(f"unknown domain: {domain!r} (have {sorted(_DOMAINS)})")
    bench = _BENCHMARKS[benchmark]()
    if not include_evidence:
        bench.has_evidence = False
    if bench.domain != domain:
        raise ValueError(
            f"benchmark {benchmark!r} belongs to domain {bench.domain!r}, not {domain!r}"
        )
    return _DOMAINS[domain](bench)
