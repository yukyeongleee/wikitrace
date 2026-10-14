"""One test phase: run, grade and persist every sample through the terminal harness."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from wikitrace.config import (
    EventLogger, JsonlWriter, LLMCallRecord, Sample, Split, TraverseOutcome, WorkflowOutcome,
    _persist_failure,
)
from wikitrace.envs import DomainAdapter, MemoryBinding, Metrics
from wikitrace.harness.llm import LLMClient, StructuredParseError
from wikitrace.memory.citation import CITATION_ROLE
from wikitrace.memory.compiler import compile_trajectory
from wikitrace.run.resume import ResumePlan
from wikitrace.run.spec import RunPaths, RunSpec


def _generate(
    domain: DomainAdapter, sample: Sample, *, llm: LLMClient, memory_text: str, run_dir: Path,
    events: EventLogger, phase: str, step_n: int, traverse_runner: Any, corpus_dir: Path | None,
    session_root: Path, traverse_cfg: Any, repair_fn: Any, user_prefix: str = "",
    memory_kwargs: dict | None = None,
) -> Any:
    """Run one episode through the terminal harness, logging its cost and any failure."""
    environment = domain.benchmark.harness_spec(sample.db_id).kind == "environment"
    outcome = _safe_run_traverse(
        domain, sample, traverse_runner, corpus_dir, session_root, traverse_cfg, repair_fn=repair_fn,
        memory_text=memory_text, user_prefix=user_prefix, memory_kwargs=memory_kwargs,
        on_call=getattr(llm, "on_call", None) if environment else None,
    )
    _emit_if_failed(outcome, events, phase=phase, step_n=step_n, sample_id=sample.id)
    citation_expected = bool((memory_kwargs or {}).get("memory_layers")) and environment
    _record_traverse_cost(
        outcome, llm, events, sample_id=sample.id, phase=phase, step_n=step_n,
        model_requested=getattr(traverse_runner, "model", ""), citation_expected=citation_expected,
    )
    return outcome


def _safe_run_traverse(
    domain: DomainAdapter, sample: Sample, runner: Any, corpus_dir: Path | None, session_root: Path,
    traverse_cfg: Any, *, repair_fn: Any, memory_text: str, user_prefix: str,
    memory_kwargs: dict | None, on_call: Any,
) -> TraverseOutcome:
    """Run the episode; on any failure, degrade to a zero-score outcome that keeps the paid logs."""
    extra: dict[str, Any] = {}
    if user_prefix:
        extra["user_prefix"] = user_prefix
    if on_call is not None:
        extra["on_call"] = on_call
    try:
        return domain.run_traverse(
            sample, runner, corpus_dir, session_root, memory_text=memory_text,
            tools=traverse_cfg.tools, max_turns=traverse_cfg.max_turns,
            sentinel_name=traverse_cfg.sentinel_name, max_retries=traverse_cfg.max_retries,
            repair_fn=repair_fn, **extra, **(memory_kwargs or {}),
        )
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        return domain.failed_traverse_outcome(
            sample, f"{type(exc).__name__}: {exc}",
            phase_logs=getattr(exc, "traverse_phase_logs", None),
        )


def _by_phase(outcome: WorkflowOutcome, field: str) -> dict[str, dict[str, list[str]]]:
    return {c.phase.value: {L: list(ids) for L, ids in getattr(c, field).items()}
            for c in outcome.result.citations}


def evaluate_test_samples(
    spec: RunSpec, paths: RunPaths, *, split: Split, plan: ResumePlan, corpus_dir: Path | None,
    llm: LLMClient, events: EventLogger, method: str,
    memory_for: Callable[[Sample], MemoryBinding] | None = None, memory_kwargs: dict | None = None,
) -> tuple[list[Metrics], int]:
    """Generate, grade and persist every test sample; return its metrics and failure count."""
    dom = spec.dom
    bind_at_reset = memory_for is not None and spec.harness.task_text_at_reset
    if bind_at_reset:
        memory_kwargs = {**(memory_kwargs or {}), "bind_memory": memory_for}
    test_meta = paths.test_dir / "metadata"
    test_meta.mkdir(parents=True, exist_ok=True)
    paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    dead_guard, provider_guard = DeadSessionGuard(), ProviderErrorGuard()
    repair = _make_sentinel_repair(spec.transport, spec.cfg.models.default.model,
                                   spec.cfg.models.default.reasoning_effort)
    test_metrics: list[Metrics] = list(plan.seeded)
    n_test_failed = plan.n_failed
    verify_install(dom, spec.cfg.dataset)
    with JsonlWriter(test_meta / "generator-citations.jsonl") as test_citations, \
            JsonlWriter(paths.metrics_dir / "samples.jsonl") as samples_writer:
        for step_n, sample in enumerate(split.test):
            if step_n < len(plan.done):
                continue
            events.emit("step_started", phase="test", step=step_n, sample_id=sample.id)
            step_dir = paths.test_dir / "trajectories" / f"step-{step_n:04d}"
            binding = memory_for(sample) if memory_for and not bind_at_reset else None
            outcome = _generate(
                dom, sample, llm=llm, memory_text=binding.memory_text if binding else "",
                run_dir=paths.run_dir, events=events, phase="test", step_n=step_n,
                traverse_runner=spec.traverse_runner, corpus_dir=corpus_dir,
                session_root=paths.test_dir / "sessions", traverse_cfg=spec.cfg.traverse,
                repair_fn=repair, user_prefix=binding.user_prefix if binding else "",
                memory_kwargs=memory_kwargs,
            )
            dead_guard.record(outcome, phase="test", step_n=step_n, sample_id=sample.id)
            provider_guard.record(outcome, phase="test", step_n=step_n, sample_id=sample.id)
            _safe_grade(dom, sample, outcome.result, llm, paths.run_dir, events, "test", step_n)
            dom.persist_traverse_trajectory(step_dir, sample, outcome)
            for citation in outcome.result.citations:
                test_citations.write(citation)
            result = outcome.result
            failed = bool(getattr(outcome, "generation_failed", False))
            n_test_failed += failed
            m = dom.benchmark.metrics_from_result(result)
            test_metrics.append(m)
            pred_rel = (step_dir / dom.prediction_filename).relative_to(paths.run_dir).as_posix()
            samples_writer.write({
                "sample_id": result.sample_id,
                "benchmark": spec.benchmark,
                "suite": spec.db_id,
                "method": method,
                "trial": spec.trial,
                "outcome": int(bool(m.success)),
                "value": m.secondary if spec.benchmark == "scienceworld" else float(bool(m.success)),
                "question": result.question,
                "generation_failed": failed,
                **dom.sample_row(result, pred_rel),
                "metrics": {
                    "success": m.success, "refused": m.refused, "primary": m.primary,
                    "secondary": m.secondary, **({"extra": m.extra} if m.extra else {}),
                },
                "cited": _by_phase(outcome, "cited"),
                "read": _by_phase(outcome, "read"),
            })
    verify_install(dom, spec.cfg.dataset)
    return test_metrics, n_test_failed


def verify_install(dom: DomainAdapter, dataset_cfg: Any) -> None:
    """Run the benchmark's install check, if it has one: before an episode loop and after it."""
    check = getattr(dom.benchmark, "verify_install", None)
    if callable(check):
        check(dataset_cfg)


def _safe_grade(
    domain: DomainAdapter,
    sample: Sample,
    result: Any,
    llm: LLMClient,
    run_dir: Path,
    events: EventLogger,
    phase: str,
    step_n: int,
) -> None:
    """Score a result through the benchmark; record a grader failure and leave the score at zero."""
    try:
        domain.benchmark.grade(sample, result, llm=llm)
    except Exception as exc:  # noqa: BLE001
        _persist_failure(run_dir, "grader", sample.id, exc)
        events.emit(
            "grading_failed", phase=phase, step=step_n, sample_id=sample.id,
            error=f"{type(exc).__name__}: {exc}",
        )
        setattr(result, "grade_reason", f"grading failed: {type(exc).__name__}: {exc}")


def _emit_if_failed(
    outcome: Any, events: EventLogger, *, phase: str, step_n: int, sample_id: str,
) -> None:
    """Emit ``generation_failed`` for any failed outcome, raised or returned."""
    if not getattr(outcome, "generation_failed", False):
        return
    result = getattr(outcome, "result", None)
    error = (
        getattr(result, "generation_error", "")
        or getattr(result, "exec_error", "")
        or "generation failed"
    )
    events.emit(
        "generation_failed", phase=phase, step=step_n, sample_id=sample_id, error=error,
    )


def _serving_providers(
    llm: Any, logs: list, events: EventLogger, *, sample_id: str, phase: str, step_n: int,
) -> dict[str, str]:
    """Each Pi call's serving provider by generation id, looked up after the episode."""
    # Pi's usage names no upstream provider; a failed lookup leaves it empty and never fails the run.
    lookup = getattr(getattr(llm, "transport", None), "generation_provider", None)
    ids = sorted({
        str(u.get("generation_id") or "")
        for log in logs for u in (getattr(log, "usage_calls", None) or [])
        if isinstance(u, dict)
    } - {""})
    if lookup is None or not ids:
        return {}

    def one(gid: str) -> tuple[str, str, str]:
        try:
            name = lookup(gid)
            return gid, name if isinstance(name, str) else "", ""
        except Exception as e:  # noqa: BLE001
            return gid, "", f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=min(8, len(ids))) as pool:
        found = list(pool.map(one, ids))
    for gid, _, error in found:
        if error:
            events.emit(
                "provider_lookup_failed", phase=phase, step=step_n, sample_id=sample_id,
                generation_id=gid, error=error[:200],
            )
    return {gid: name for gid, name, _ in found if name}


def _record_traverse_cost(
    outcome: Any, llm: LLMClient, events: EventLogger, *,
    sample_id: str, phase: str, step_n: int, model_requested: str,
    citation_expected: bool = False,
) -> None:
    """Log the TERMINAL generator's API cost into the run's ``llm_calls.jsonl``."""
    # The only record of a terminal phase's cost; never fails a run.
    on_call = getattr(llm, "on_call", None)
    if on_call is None:
        return
    logs = getattr(outcome, "phase_logs", None) or []
    if not logs:
        # No logs at all: a run whose cost is understated looks exactly like a cheap one.
        events.emit(
            "traverse_cost_missing", phase=phase, step=step_n,
            sample_id=sample_id, role="(no phase logs)",
        )
        return
    providers = _serving_providers(
        llm, logs, events, sample_id=sample_id, phase=phase, step_n=step_n,
    )
    for log in logs:
        # `stopped_by` is a kill by one of our guards; `truncated_calls` counts output-budget cuts the session may have survived.
        stopped_by = getattr(log, "stopped_by", "")
        truncated = int(getattr(log, "truncated_calls", 0) or 0)
        if stopped_by or truncated:
            events.emit(
                "session_cut_short", phase=phase, step=step_n, sample_id=sample_id,
                role=log.phase, stopped_by=stopped_by,
                truncated_calls=truncated,
                last_stop_reason=getattr(log, "last_stop_reason", ""),
                last_had_tool_call=bool(getattr(log, "last_had_tool_call", False)),
            )
        if provider_error := getattr(log, "provider_error", ""):
            events.emit(
                "provider_error", phase=phase, step=step_n, sample_id=sample_id,
                role=log.phase, error=provider_error[:500],
                retries=sum(t.get("type") == "provider_retry" for t in log.turns or []),
            )
        if stopped_by:
            # The generation streaming at the kill was billed but has no id to look up.
            events.emit(
                "traverse_cost_truncated", phase=phase, step=step_n,
                sample_id=sample_id, role=log.phase, reason=stopped_by,
            )
        citation = next(
            (t for t in (getattr(log, "turns", None) or []) if t.get("type") == "citation"),
            None,
        )
        if citation is not None and citation.get("parsed") is None:
            events.emit(
                "citation_unparsed", phase=phase, step=step_n, sample_id=sample_id,
                role=log.phase, text=str(citation.get("text", ""))[:200],
            )
        elif citation is None and citation_expected:
            events.emit(
                "citation_skipped", phase=phase, step=step_n, sample_id=sample_id,
                role=log.phase, stopped_by=stopped_by,
            )
        # The citation calls are those whose provider ids the citation turn recorded.
        citation_ids = {
            str(i or "") for i in
            [citation.get("response_id"), *citation.get("missed_response_ids", [])]
        } - {""} if citation else set()
        calls = list(getattr(log, "usage_calls", None) or [])
        kind = "call"
        if not calls:
            kind = "session-total"
            calls = [u] if (u := getattr(log, "usage", None)) else []
        if not calls:
            events.emit(
                "traverse_cost_missing", phase=phase, step=step_n,
                sample_id=sample_id, role=log.phase,
            )
            continue
        for i, usage in enumerate(calls):
            try:
                on_call(LLMCallRecord(
                    # train/test included so a bare sample id cannot collide across the splits.
                    call_id=f"traverse-{kind}:{phase}:{sample_id}:{log.phase}:{i}",
                    generation_id=str(usage.get("generation_id", "")),
                    role=(
                        CITATION_ROLE
                        if kind == "call" and str(usage.get("generation_id", "")) in citation_ids
                        else log.phase
                    ),
                    phase=phase,
                    sample_id=sample_id,
                    model_requested=model_requested,
                    model_returned=str(usage.get("model_returned", "")),
                    finish_reason=str(usage.get("finish_reason", "")),
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    cached_tokens=int(usage.get("cached_tokens", 0)),
                    cost=float(usage.get("cost", 0.0)),
                    provider=providers.get(str(usage.get("generation_id", "")), ""),
                ))
            except Exception:  # noqa: BLE001
                continue


def _make_sentinel_repair(transport: Any, model: str, reasoning_effort: str | None):
    """One no-tool call that rewrites a phase's final answer as the strict JSON sentinel."""

    def repair(*, role, final_context, final_text, required_keys) -> str:
        keys = ", ".join(required_keys)
        messages = list(final_context or [])
        if not any((m or {}).get("role") == "system" for m in messages):
            messages.insert(
                0, {"role": "system", "content": "You output only a single strict JSON object."}
            )
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous answer (below) was not saved as a valid JSON "
                    f"sentinel.\n\n---\n{final_text}\n---\n\nOutput ONE JSON object with "
                    f"EXACTLY these top-level keys: [{keys}]. No other keys, no markdown, "
                    "no prose — only the JSON object."
                ),
            }
        )
        try:
            resp = transport.chat(
                model, messages,
                response_format={"type": "json_object"},
                reasoning_effort=reasoning_effort,
            )
            return getattr(resp, "content", "") or ""
        except Exception:
            return ""

    return repair


def _default_traverse_runner(cfg: Any, api_key: str | None, harness: Any = None) -> Any:
    """The Pi runner the config describes."""
    from wikitrace.harness.terminal import PiTraverseRunner

    tc = cfg.traverse
    if tc.runner != "pi":
        raise ValueError(
            f"traverse.runner={tc.runner!r} requires an injected traverse_runner; "
            "only 'pi' is auto-constructed."
        )
    effort = (cfg.models.default.reasoning_effort or "high").lower()
    thinking = "off" if effort in ("none", "off", "") else effort
    extra_args = [
        "--context-management-level", tc.context_level,
        "--thinking", thinking,
    ]
    def _guard(name: str) -> float:
        override = getattr(harness, name, None)
        return getattr(tc, name) if override is None else override

    return PiTraverseRunner(
        pi_cli=tc.pi_cli,
        model=cfg.models.default.model,
        provider=tc.provider,
        base_url=tc.base_url,
        api_key=api_key,
        extra_args=extra_args,
        cmd_timeout=_guard("cmd_timeout"),
        max_cmd_kills=tc.max_cmd_kills,
        idle_timeout=tc.idle_timeout,
        max_seconds=_guard("max_seconds"),
        sandbox=tc.sandbox,
        docker_image=tc.docker_image,
        network_policy=tc.network_policy,
        allowed_hosts=list(getattr(harness, "allowed_hosts", ()) or ()),
        openrouter_providers=cfg.llm.openrouter_providers,
    )


class DeadSessionGuard:
    """Abort a phase when a terminal session ended without a single turn (`stopped_by == "eof"`)."""

    def __init__(self, limit: int = 1) -> None:
        self.limit = limit
        self.streak = 0

    @staticmethod
    def _produced_nothing(outcome: Any) -> bool:
        logs = getattr(outcome, "phase_logs", None)
        if not logs:
            return False
        return all(
            getattr(log, "stopped_by", "") == "eof"
            and not (getattr(log, "turns", None) or [])
            for log in logs
        )

    def record(self, outcome: Any, *, phase: str, step_n: int, sample_id: str) -> None:
        if not self._produced_nothing(outcome):
            self.streak = 0
            return
        self.streak += 1
        if self.streak < self.limit:
            return
        # Phrased for the DEFAULT limit of 1; the streak is named only when more was configured.
        what = (
            "a terminal session started but produced NO turns"
            if self.streak == 1
            else f"{self.streak} consecutive terminal sessions produced NO turns"
        )
        raise RuntimeError(
            f"{what} "
            f"(phase={phase}, through step {step_n}, last sample {sample_id!r}). That is "
            f"a harness fault, not a wrong answer — every one of these would be recorded "
            f"as an empty prediction with refused=False and scored 0. Check that the "
            f"sandbox image exists (scripts/setup/sandbox_image.sh), that the docker "
            f"daemon is up, and that the run's mounts resolve."
        )


class ProviderErrorGuard:
    """Abort a phase after ``limit`` consecutive episodes ended on a provider error."""

    # Pi's retries cannot clear a failed request, so a key out of credit would otherwise finish the cell quietly low.
    def __init__(self, limit: int = 3) -> None:
        self.limit = limit
        self.streak = 0

    def record(self, outcome: Any, *, phase: str, step_n: int, sample_id: str) -> None:
        errors = [
            e for log in (getattr(outcome, "phase_logs", None) or [])
            if (e := getattr(log, "provider_error", ""))
        ]
        if not errors:
            self.streak = 0
            return
        self.streak += 1
        if self.streak < self.limit:
            return
        raise RuntimeError(
            f"{self.streak} consecutive episodes ended on a model provider error that "
            f"Pi's retries did not clear (phase={phase}, through step {step_n}, last "
            f"sample {sample_id!r}, last error: {errors[-1][:300]!r}). The provider is "
            f"failing (rate limit, exhausted key credit or outage), not the agent; each "
            f"such episode would be scored as a failure. Rerun the cell once it recovers."
        )


def _compiler_profile(dom: Any, db_id: str, corpus_dir: Path) -> Any:
    """The suite's compiler prompt profile plus the reference material its generator was given."""
    profile = dom.prompt_profile(db_id)
    reference = dom.compiler_world_reference(corpus_dir)
    if not reference:
        return profile
    return profile.model_copy(update={"compiler_world_reference": reference})


def _safe_compile(
    llm: LLMClient, ci: Any, step_n: int, evidence: str, trajectory_text: str, outcome_text: str,
    run_dir: Path, events: EventLogger, profile: Any, confidence: bool = False,
) -> Any | None:
    """Compile one trajectory, or persist a parse failure and return None instead of aborting."""
    try:
        return compile_trajectory(llm, ci, step_n, evidence, trajectory_text=trajectory_text,
                                  outcome_text=outcome_text, profile=profile, confidence=confidence)
    except StructuredParseError as exc:
        _persist_failure(run_dir, "compiler", f"step-{step_n:04d}-{ci.sample_id}", exc)
        events.emit("compile_failed", phase="train", step=step_n, sample_id=ci.sample_id, error=str(exc))
        return None
