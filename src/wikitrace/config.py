"""Configuration, train/test splits, the shared schemas, run artifacts and logging sinks."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Phase(str, Enum):
    SOLVER = "solver"


class Outcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class StrictModel(BaseModel):
    """Base for structured-output models: forbid extra keys so a malformed
    completion fails validation and triggers a retry."""

    model_config = ConfigDict(extra="forbid")


@dataclass
class CompilationInputs:
    """The domain-agnostic facts the compiler needs from one finished trajectory."""

    sample_id: str
    db_id: str
    success: bool
    metrics: dict[str, float]
    question: str
    prediction: str
    # Set only by a domain that can abstain; orthogonal to ``success``, else an abstention reads as a wrong answer.
    refused: bool = False
    generation_failed: bool = False
    choices: str = ""
    task_context: str = ""
    task_context_preview: str = ""
    task_fields: tuple[tuple[str, str], ...] = ()


class PromptProfile(BaseModel):
    """What the compiler prompts say about one suite's world."""

    model_config = ConfigDict(frozen=True)

    task_kind: str = ""
    solver_prompt: str = ""
    has_evidence: bool
    compiler_world_reference: str = ""


class ConfidenceLevel(str, Enum):
    """How well the trajectory supports a failure attribution or a corrective insight.

    Anchored on what the trajectory can PROVE, not on how plausible the account
    feels. The compiler never sees the correct answer, so an unanchored scale is
    self-flattering: measured over suite02, the free-text field it replaces read
    ``high`` 49 times, ``medium`` 22 and ``low`` ONCE — including one ``high`` whose
    own text said it could not be sure without the hidden gold SQL.

    Failure-side only. A success compilation's attribution carries no comparable
    doubt (EX=1 is ground truth that the whole thing worked), and the risk that
    remains there is over-generalisation, which is the writer's ``applicability``
    problem rather than a confidence one.
    """


    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EnvKeyDecision(StrictModel):
    decision: str
    support: str


class EnvFailureCause(StrictModel):
    decision: str
    support: str


class EnvFailureCauseWithConfidence(StrictModel):
    decision: str
    support: str
    confidence: ConfidenceLevel


class EnvInsightItem(StrictModel):
    """A candidate insight. `title` names it in a few words, `guideline` says what to do."""

    title: str = Field(min_length=1)
    guideline: str = Field(min_length=1)
    applicability: str


class EnvCompilerSuccessOutput(StrictModel):
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    key_decisions: list[EnvKeyDecision] = Field(min_length=1)
    candidate_insights: list[EnvInsightItem] = Field(min_length=1)


class EnvCompilerFailureOutput(StrictModel):
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    failure_causes: list[EnvFailureCause] = Field(min_length=1)
    candidate_insights: list[EnvInsightItem] = Field(min_length=1)


class EnvCompilerFailureOutputWithConfidence(StrictModel):
    # ``uncertainties`` FIRST: the prompt grades a cause against it, so schema order must write it first.
    uncertainties: list[str]
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    failure_causes: list[EnvFailureCauseWithConfidence] = Field(min_length=1)
    candidate_insights: list[EnvInsightItem] = Field(min_length=1)


class Sample(BaseModel):
    """The fields a split and a run read off any benchmark's sample."""

    id: str
    db_id: str
    question: str
    evidence: str = ""
    stratum: str = ""
    split_role: str = ""


class Compilation(BaseModel):
    """A single-trajectory compilation document (compilation/step-NNNN.md)."""

    step_id: str
    outcome: Outcome
    metrics: dict[str, float] = Field(default_factory=dict)
    refused: bool = False
    generation_failed: bool = False
    title: str = ""         # `[<outcome>] <agent headline>` (runner-tagged)
    description: str = ""
    question: str = ""
    evidence: str = ""
    prediction: str = ""
    body: str = ""

    def metric(self, key: str, default: float = 0.0) -> float:
        """This compilation's value for a declared metric name."""
        return self.metrics.get(key, default)


class CitationRecord(BaseModel):
    """A generator phase's memory usage, per layer: what it cited and what it read."""

    sample_id: str
    step_id: str = ""
    phase: Phase
    cited: dict[str, list[str]] = Field(default_factory=dict)
    read: dict[str, list[str]] = Field(default_factory=dict)
    comment: str = ""


class ToolStep(BaseModel):
    """One terminal-agent tool call, as the compiler reads it: why, what, what came back."""

    reasoning: str = ""
    command: str = ""
    result: str = ""
    is_error: bool = False
    # Context-management state of the OUTPUT: "", "truncated", "erased" or "compacted"; tells a cut-off output from an empty one.
    state: str = ""
    # WHY this call has no result (distinct from ``state``); only `command_timeout` and `turn_cap` support reasoning about METHOD.
    unresolved: str = ""


class StructuredCall(Protocol):
    """One structured LLM call, as the layers above `harness/` read it."""

    record: LLMCallRecord


class PhaseLog(Protocol):
    """One terminal phase's log, as `experiments/` reads it for cost and liveness."""

    phase: str
    usage: dict
    usage_calls: list
    stopped_by: str


@dataclass
class WorkflowOutcome:
    """A completed structured generator run: the scored result plus its trajectories."""

    result: Any
    structured: list[StructuredCall] = field(default_factory=list)
    generation_failed: bool = False


@dataclass
class TraverseOutcome:
    """A completed traverse run: the scored result plus one log per agent phase."""

    result: Any
    phase_logs: list[PhaseLog] = field(default_factory=list)
    generation_failed: bool = False


class LLMCallRecord(BaseModel):
    call_id: str
    generation_id: str = ""
    role: str = ""
    phase: str = ""
    sample_id: str = ""
    model_requested: str = ""
    model_returned: str = ""
    prompt_hash: str = ""
    response_hash: str = ""
    latency: float = 0.0
    retry_count: int = 0
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0
    provider: str = ""


class ModelSpec(BaseModel):
    # Forbid extra keys so a typo'd field (e.g. ``max_token``) fails loudly, also in per-role overrides.
    model_config = ConfigDict(extra="forbid")

    provider: str = "openrouter"
    model: str
    temperature: float = 0.0
    max_tokens: int = 8192
    reasoning_effort: str = "high"


class ModelsConfig(BaseModel):
    default: ModelSpec
    roles: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def for_role(self, role: str) -> ModelSpec:
        override = self.roles.get(role)
        if not override:
            return self.default
        return ModelSpec(**{**self.default.model_dump(), **override})


class DatasetConfig(BaseModel):
    alfworld_root: str = "data/alfworld"
    scienceworld_root: str = "data/scienceworld"
    plancraft_root: str = "data/plancraft"
    appworld_root: str = "data/appworld"
    tau3_root: str = "data/tau3"
    tau3_user_model: Optional[str] = None
    tau3_user_guidelines: Literal["official", "clarified"] = "official"
    # AppWorld only: also count a null-gold answer-check-only failure as a success (`tgc_lenient`); TGC stays primary.
    appworld_lenient_answer: bool = False
    # False blanks the field and mentions the channel in no prompt, since even saying it is absent discloses it exists.
    include_evidence: bool = True
    limit_explicit_split: bool = False
    # ``null``: test takes every sample left after train; an integer fixes the size and fails loudly if unfundable.
    test_size: Optional[int] = 100
    train_size: int = 50
    train_order: Literal["reverse", "forward"] = "reverse"


class MemoryConfig(BaseModel):
    compiler_confidence: bool = False


class ExperimentConfig(BaseModel):
    output_root: str = "outputs"
    seed: int = 0


class LLMConfig(BaseModel):
    max_retries: int = 3
    request_timeout: int = 120
    base_url: str = "https://openrouter.ai/api/v1"
    openrouter_providers: list[str] = []


class TraverseConfig(BaseModel):
    """Stage 2 traverse (DCI terminal-agent generator) settings."""

    runner: Literal["pi", "fake"] = "pi"
    pi_cli: str = "tools/pi-mono/packages/coding-agent/dist/cli.js"
    # Pi's NATIVE openrouter provider; the openai provider hits /v1/responses and returns empty turns for deepseek.
    provider: str = "openrouter"
    base_url: str = "https://openrouter.ai/api/v1"  # only used by the openai provider
    tools: str = "read,bash"
    max_turns: int = 50
    context_level: str = "level3"
    sentinel_name: str = "__out.json"
    max_retries: int = 1
    cmd_timeout: float = 90.0
    # Force-kills allowed per session; on the last the agent is told to answer now. 5 x cmd_timeout is the worst-case waste, which max_seconds must cover.
    max_cmd_kills: int = 5
    # Hang guards that DO kill the session (no event for idle_timeout, or wall-clock > max_seconds); the episode scores zero.
    idle_timeout: float = 120.0
    max_seconds: float = 1000.0
    sandbox: Literal["docker", "none"] = "docker"
    # The terminal agent computes INSIDE this image (docker/sandbox/Dockerfile); pin by image id for a reproducible run.
    docker_image: str = "wikitrace-sandbox:node20"
    # The container is ALWAYS `--network none`; this decides only whether a host relay opens a scoped hole to the LLM endpoint.
    network_policy: Literal["deny", "allowlist"] = "allowlist"


class AppConfig(BaseModel):
    dataset: DatasetConfig
    models: ModelsConfig
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    traverse: TraverseConfig = Field(default_factory=TraverseConfig)

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins on scalars/lists)."""
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_raw(config_path: Path | str) -> dict[str, Any]:
    """Load a YAML config, resolving a single optional `extends` chain."""
    path = Path(config_path).resolve()
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    extends = raw.pop("extends", None)
    if extends:
        parent_path = (path.parent / extends).resolve()
        parent = load_raw(parent_path)
        raw = _deep_merge(parent, raw)
    return raw


def load_config(config_path: Path | str) -> AppConfig:
    """Load and validate a config file into a typed AppConfig."""
    return AppConfig.model_validate(load_raw(config_path))


@dataclass(frozen=True)
class ConfigView:
    """The `dataset` keys and `models.roles` one benchmark reads: what its runs record and hash."""
    dataset_keys: tuple[str, ...]
    roles: tuple[str, ...]
    omit_at_default: tuple[str, ...] = ()


def resolved_dict(config: AppConfig, view: ConfigView | None = None) -> dict[str, Any]:
    """Deterministic dict form of the resolved config; `None` keeps every key."""
    d = config.model_dump(mode="json")
    if view is not None:
        defaults = DatasetConfig().model_dump(mode="json")
        d["dataset"] = {k: v for k, v in d["dataset"].items() if k in view.dataset_keys
                        and not (k in view.omit_at_default and v == defaults[k])}
        d["models"]["roles"] = {
            k: v for k, v in d["models"]["roles"].items() if k in view.roles
        }
    return d


def config_hash(config: AppConfig, view: ConfigView | None = None) -> str:
    """Stable sha256 of the resolved config."""
    blob = json.dumps(resolved_dict(config, view), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def dump_resolved(config: AppConfig, dest: Path | str, view: ConfigView | None = None) -> None:
    """Write the resolved config as YAML (config.resolved.yaml)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_dict(config, view), f, sort_keys=True, allow_unicode=True)


@dataclass
class Split:
    db_id: str
    test: list[Sample]
    train: list[Sample]


def split_sizes(n: int, cfg: DatasetConfig, db_id: str) -> tuple[int, int]:
    """``(test_size, train_size)`` for a suite of ``n`` samples."""
    train_size = cfg.train_size
    if cfg.test_size is None:
        test_size = n - train_size
        if test_size <= 0:
            raise ValueError(
                f"suite '{db_id}' has n={n} samples, which cannot fund train_size="
                f"{train_size} plus a non-empty test set"
            )
        return test_size, train_size
    if n < cfg.test_size + train_size:
        raise ValueError(
            f"suite '{db_id}' has n={n} samples, short of test_size={cfg.test_size} "
            f"+ train_size={train_size}. Set `dataset.test_size: null` to let test "
            f"take the remainder, or pick a smaller test_size."
        )
    return cfg.test_size, train_size


def _draw(groups: "OrderedDict[str, list[Sample]]", k: int) -> list[Sample]:
    """Pop up to ``k`` samples round-robin across the strata (mutates ``groups``)."""
    out: list[Sample] = []
    while len(out) < k:
        progressed = False
        for group in groups.values():
            if not group:
                continue
            out.append(group.pop(0))
            progressed = True
            if len(out) == k:
                return out
        if not progressed:
            break
    return out


def _stratified_split(
    samples: list[Sample], test_size: int, train_size: int
) -> tuple[list[Sample], list[Sample]]:
    """Test first, then train, both drawn round-robin from what is left."""
    groups: OrderedDict[str, list[Sample]] = OrderedDict()
    for s in samples:
        groups.setdefault(getattr(s, "stratum", ""), []).append(s)
    test = _draw(groups, test_size)
    train = _draw(groups, train_size)
    return test, train


def make_split(samples: list[Sample], cfg: DatasetConfig, db_id: str) -> Split:
    n = len(samples)

    def role(s: Sample) -> str:
        return getattr(s, "split_role", "")

    # Explicit-role split: a benchmark pinning samples to "train"/"test" bypasses sizing; smoke only truncates each role.
    if any(role(s) for s in samples):
        train = [s for s in samples if role(s) == "train"]
        test = [s for s in samples if role(s) == "test"]
        if not train or not test:
            raise ValueError(
                f"suite '{db_id}': explicit-role split needs both train and test "
                f"(got train={len(train)}, test={len(test)})"
            )
        ids = {s.id for s in train} & {s.id for s in test}
        if ids:
            raise ValueError(f"suite '{db_id}': {len(ids)} samples marked both train and test")
        if cfg.limit_explicit_split:
            for name, size, available in (
                ("train_size", cfg.train_size, len(train)),
                ("test_size", cfg.test_size, len(test)),
            ):
                if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                    raise ValueError(f"limit_explicit_split requires positive integer {name}")
                if size > available:
                    raise ValueError(
                        f"suite '{db_id}': {name}={size} exceeds frozen role size {available}"
                    )
            train, test = train[:cfg.train_size], test[:cfg.test_size]
        return Split(db_id=db_id, test=test, train=train)

    test_size, train_size = split_sizes(n, cfg, db_id)

    if any(getattr(s, "stratum", "") for s in samples):
        test, train = _stratified_split(samples, test_size, train_size)
        if len(train) < train_size:
            raise ValueError(
                f"suite '{db_id}': only {len(train)} samples left for train after a "
                f"{len(test)}-sample test draw (n={n}, train_size={train_size})"
            )
    else:
        test = samples[:test_size]
        train = samples[max(0, n - train_size):]
    if cfg.train_order == "reverse":
        train = list(reversed(train))

    test_ids = {s.id for s in test}
    train_ids = {s.id for s in train}
    overlap = test_ids & train_ids
    if overlap:
        raise ValueError(
            f"Train/test overlap for db '{db_id}' ({len(overlap)} samples). "
            f"n={n}, test_size={test_size}, train_size={train_size}."
        )

    return Split(db_id=db_id, test=test, train=train)


def write_turns(path: Path, turns: list[dict]) -> None:
    """Write a multi-turn stream as JSONL, one turn per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(t, ensure_ascii=False) for t in turns]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _persist_failure(run_dir: Path, phase: str, ident: str, exc: Exception) -> None:
    """Save the raw model output that failed to parse, for inspection."""
    fail_dir = run_dir / "_failures"
    fail_dir.mkdir(parents=True, exist_ok=True)
    last_content = getattr(exc, "last_content", "")
    record = {
        "phase": phase,
        "ident": ident,
        "error": f"{type(exc).__name__}: {exc}",
        "raw_output": last_content,
    }
    (fail_dir / f"{phase}-{ident}.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# One line per step, appended, LAST WINS, at the CORPUS ROOT and not in `raw/`: a record there
# would carry `grade_reason` to the agent the compiler is kept from informing.
RESULTS_LOG = "results.jsonl"

#: The per-step spelling of the same record, read so corpora frozen with it still compile.
LEGACY_NAME = "result.json"


def reset(corpus_dir: Path | str) -> None:
    """Drop the log, so a rebuild keeps no record of a step outside its split."""
    (Path(corpus_dir) / RESULTS_LOG).unlink(missing_ok=True)


def append_result(corpus_dir: Path | str, step_id: str, result: Any) -> None:
    """Append one step's structured result."""
    # Its own key, so a domain field named ``step_id`` cannot overwrite the join key.
    payload = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
    line = json.dumps({"step_id": step_id, "result": payload}, ensure_ascii=False)
    path = Path(corpus_dir) / RESULTS_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def read_result_json(step_dir: Path | str) -> str:
    """The JSON text of this step's result, for the domain adapter to parse."""
    step_dir = Path(step_dir)
    step_id = step_dir.name
    log = step_dir.parent.parent / RESULTS_LOG
    if log.exists():
        found = None
        for line in log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("step_id") == step_id:
                found = rec["result"]      # last wins: a repaired step appends a second record
        if found is not None:
            return json.dumps(found, ensure_ascii=False)

    legacy = step_dir / LEGACY_NAME
    if legacy.exists():
        return legacy.read_text(encoding="utf-8")

    raise FileNotFoundError(
        f"no result for {step_id}: neither {log} (nor a record for that step in it) "
        f"nor the legacy {legacy}"
    )


def _to_dict(obj: Any) -> dict[str, Any]:
    """Coerce a pydantic model or mapping into a JSON-ready dict."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    return dict(obj)


class JsonlWriter:
    """Append-only JSON-lines writer; flushes after each line."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, record: dict[str, Any] | BaseModel) -> None:
        self._fh.write(json.dumps(_to_dict(record), ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class EventLogger:
    """Timestamped event stream over a :class:`JsonlWriter`."""

    def __init__(self, path: Path | str) -> None:
        self._writer = JsonlWriter(path)

    def emit(self, event_type: str, **fields: Any) -> None:
        self._writer.write(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": event_type,
                **fields,
            }
        )

    def close(self) -> None:
        self._writer.close()


def aggregate_llm_calls(path: Path | str) -> dict[str, Any]:
    """Aggregate ``llm_calls.jsonl`` into per-role and total token/cost stats."""
    path = Path(path)
    by_role: dict[str, dict[str, float]] = {}
    totals = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "cost": 0.0,
    }
    if not path.exists():
        return {"by_role": by_role, "totals": totals}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            role = rec.get("role", "") or "(unknown)"
            bucket = by_role.setdefault(
                role,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "reasoning_tokens": 0,
                    "cached_tokens": 0,
                    "cost": 0.0,
                },
            )
            bucket["calls"] += 1
            totals["calls"] += 1
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "reasoning_tokens",
                "cached_tokens",
                "cost",
            ):
                val = rec.get(key, 0) or 0
                bucket[key] += val
                totals[key] += val

    return {"by_role": by_role, "totals": totals}


def provider_counts(*paths: Path | str) -> dict[str, int]:
    """Calls per serving provider over ``llm_calls`` files; "" counts as ``(unknown)``."""
    counts: dict[str, int] = {}
    for path in map(Path, paths):
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    name = json.loads(line).get("provider") or "(unknown)"
                    counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))
