"""One trial's corpus: no-memory train runs, their compilations and insights, and the recommendations."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from wikitrace.config import (
    Compilation, EventLogger, JsonlWriter, append_result, provider_counts, reset,
)
from wikitrace.envs import train_group_for
from wikitrace.harness.llm import LLMClient, OpenRouterClient
from wikitrace.harness.llm.openrouter import provider_routing
from wikitrace.harness.prompts import prompts_source_hash
from wikitrace.memory.compiler import read_episodes
from wikitrace.memory.recommend import Embedder, compute_recommendations, recommendation_path, write_recommendations
from wikitrace.memory.store import compilation_to_markdown
from wikitrace.run.evaluate import (
    DeadSessionGuard, ProviderErrorGuard, _compiler_profile, _default_traverse_runner, _generate,
    _make_sentinel_repair, _safe_compile, _safe_grade, verify_install,
)
from wikitrace.run.spec import RunSpec, load_split, resolve_spec
from wikitrace.run.storage import corpus_dir_for, gather_model_ids


def _transport(spec: RunSpec) -> Any:
    if spec.transport is not None:
        return spec.transport
    cfg = spec.cfg
    return OpenRouterClient(base_url=cfg.llm.base_url, timeout=cfg.llm.request_timeout,
                            api_key=spec.api_key, providers=cfg.llm.openrouter_providers)


def _split_hash(train_ids: list[str], test_ids: list[str]) -> str:
    h = hashlib.sha256()
    h.update("\n".join(train_ids).encode("utf-8"))
    h.update(b"\x00")
    h.update("\n".join(test_ids).encode("utf-8"))
    return h.hexdigest()[:16]


def generate_train_runs(spec: RunSpec, corpus_dir: Path, transport: Any) -> None:
    """Run every train sample with no memory and freeze its trajectory under ``raw/``."""
    cfg, dom = spec.cfg, spec.dom
    for derived in ("raw", "compilation"):
        shutil.rmtree(corpus_dir / derived, ignore_errors=True)
    reset(corpus_dir)
    raw_dir = corpus_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    model_validation = transport.validate_models(gather_model_ids(cfg.models))
    events = EventLogger(corpus_dir / "events.jsonl")
    llm_calls = JsonlWriter(corpus_dir / "llm_calls.jsonl")
    llm = LLMClient(cfg.models, transport, cfg.llm.max_retries, on_call=llm_calls.write)
    split = load_split(spec)
    verify_install(dom, cfg.dataset)
    runner = spec.traverse_runner or _default_traverse_runner(cfg, spec.api_key, spec.harness)
    repair = _make_sentinel_repair(transport, cfg.models.default.model, cfg.models.default.reasoning_effort)
    start_ts = datetime.now(timezone.utc).isoformat()
    events.emit("corpus_started", db_id=spec.db_id, train_size=len(split.train))
    dead_guard, provider_guard = DeadSessionGuard(), ProviderErrorGuard()
    for step_n, sample in enumerate(split.train):
        events.emit("step_started", phase="corpus", step=step_n, sample_id=sample.id)
        outcome = _generate(
            dom, sample, llm=llm, memory_text="", run_dir=corpus_dir, events=events, phase="corpus",
            step_n=step_n, traverse_runner=runner, corpus_dir=None, session_root=corpus_dir / "sessions",
            traverse_cfg=cfg.traverse, repair_fn=repair,
        )
        dead_guard.record(outcome, phase="corpus", step_n=step_n, sample_id=sample.id)
        provider_guard.record(outcome, phase="corpus", step_n=step_n, sample_id=sample.id)
        _safe_grade(dom, sample, outcome.result, llm, corpus_dir, events, "corpus", step_n)
        step_dir = raw_dir / f"step-{step_n:04d}"
        dom.persist_traverse_trajectory(step_dir, sample, outcome)
        dom.persist_train_task_files(step_dir, sample)
        # Beside raw/, not inside it: raw/ is mounted, and this holds the grader's reason.
        append_result(corpus_dir, step_dir.name, outcome.result)
    llm_calls.close()
    verify_install(dom, cfg.dataset)
    train_ids, test_ids = [s.id for s in split.train], [s.id for s in split.test]
    manifest = {
        "db_id": spec.db_id, "train_size": len(train_ids), "test_size": len(test_ids),
        "train_step_ids": train_ids, "test_sample_ids": test_ids,
        "split_hash": _split_hash(train_ids, test_ids),
        "model_ids": gather_model_ids(cfg.models), "model_validation": model_validation,
        "openrouter_routing": provider_routing(cfg.llm.openrouter_providers),
        "provider_calls": provider_counts(corpus_dir / "llm_calls.jsonl"),
        "fp8": spec.fp["fp8"], "seed": cfg.experiment.seed, "start_ts": start_ts, "end_ts": datetime.now(timezone.utc).isoformat(),
    }
    (corpus_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    events.emit("corpus_completed", db_id=spec.db_id)


def compile_train_runs(spec: RunSpec, corpus_dir: Path, transport: Any) -> None:
    """Compile every raw step once into ``compilation/``, keeping each compiler's insights."""
    cfg, dom = spec.cfg, spec.dom
    events = EventLogger(corpus_dir / "events.compile.jsonl")
    llm_calls = JsonlWriter(corpus_dir / "llm_calls.compile.jsonl")
    llm = LLMClient(cfg.models, transport, cfg.llm.max_retries, on_call=llm_calls.write)
    by_id = {s.id: s for s in load_split(spec).train}
    comp_dir = corpus_dir / "compilation"
    shutil.rmtree(comp_dir, ignore_errors=True)
    comp_dir.mkdir(parents=True)
    profile = _compiler_profile(dom, spec.db_id, corpus_dir)
    comps: list[Compilation] = []
    items: dict[str, list[dict[str, Any]]] = {}
    for step_dir in sorted((corpus_dir / "raw").glob("step-*")):
        step_n = int(step_dir.name.split("-")[1])
        result = dom.read_result(step_dir)
        sample = by_id.get(result.sample_id)
        if sample is None:
            events.emit("corpus_sample_missing", phase="compile", step_id=step_dir.name,
                        sample_id=result.sample_id)
        cr = _safe_compile(
            llm, dom.compilation_inputs(result, sample=sample), step_n,
            getattr(sample, "evidence", "") if sample is not None else "",
            dom.render_trajectory_for_compiler(result), dom.render_outcome_for_compiler(result, sample=sample),
            corpus_dir, events, profile, confidence=cfg.memory.compiler_confidence,
        )
        if cr is None:
            continue
        comps.append(cr.compilation)
        items[cr.compilation.step_id] = [i.model_dump() for i in cr.structured.parsed.candidate_insights]
        (comp_dir / f"{cr.compilation.step_id}.md").write_text(
            compilation_to_markdown(cr.compilation, raw_shape=spec.raw_shape), encoding="utf-8")
    comps.sort(key=lambda c: c.step_id)
    (comp_dir / "compilations.jsonl").write_text("".join(
        json.dumps({**json.loads(c.model_dump_json()), "insight_items": items[c.step_id]},
                   ensure_ascii=False) + "\n" for c in comps), encoding="utf-8")
    llm_calls.close()
    events.emit("compile_completed", db_id=spec.db_id, n_compilations=len(comps))


def built_fp8(corpus_dir: Path) -> str:
    """The fingerprint a corpus was built under, from its stamp or else its train manifest."""
    for name in ("corpus.json", "manifest.json"):
        path = corpus_dir / name
        if path.exists():
            return str(json.loads(path.read_text(encoding="utf-8")).get("fp8", ""))
    return ""


def ensure_corpus(spec: RunSpec, transport: Any) -> Path:
    """The suite's corpus for this trial, generating and compiling whatever stage is missing."""
    group = train_group_for(spec.dom.benchmark, spec.db_id)
    corpus_dir = corpus_dir_for(spec.output_root, spec.benchmark, group, spec.trial)
    built = built_fp8(corpus_dir)
    if built and built != spec.fp["fp8"]:
        raise RuntimeError(f"{corpus_dir} was built under fingerprint {built}, the current config gives "
                           f"{spec.fp['fp8']}; delete it to rebuild")
    manifest = corpus_dir / "manifest.json"
    if manifest.exists():
        frozen = json.loads(manifest.read_text(encoding="utf-8")).get("train_step_ids", [])
        if frozen != [s.id for s in load_split(spec).train]:
            raise RuntimeError(f"{corpus_dir} was built from another train split than "
                               f"{spec.benchmark}/{spec.db_id} draws; delete it and rebuild")
    else:
        corpus_dir.mkdir(parents=True, exist_ok=True)
        generate_train_runs(spec, corpus_dir, transport)
    if not (corpus_dir / "compilation" / "compilations.jsonl").exists():
        compile_train_runs(spec, corpus_dir, transport)
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["compile"] = {"git_commit": spec.git["commit"], "git_dirty": spec.git["dirty"],
                           "prompts_hash": prompts_source_hash()}
        manifest.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    stamp = corpus_dir / "corpus.json"
    if not stamp.exists():
        stamp.write_text(json.dumps({
            "benchmark": spec.benchmark, "domain": spec.domain, "train_group": group,
            "fp8": spec.fp["fp8"], "built_from_suite": spec.db_id, "shared_half": "train",
            **({"compiler_confidence": True} if spec.cfg.memory.compiler_confidence else {}),
            "insight_items": True,
        }, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return corpus_dir


def ensure_recommendations(spec: RunSpec, corpus_dir: Path, transport: Any) -> Path | None:
    """The suite's frozen recommendation file in ``corpus_dir``; None where they are made at reset."""
    if spec.harness.task_text_at_reset:
        return None
    path = recommendation_path(corpus_dir, spec.db_id)
    if path.exists():
        return path
    episodes = read_episodes(spec.dom, corpus_dir)
    train = [(ep.step, ep.sample_id, ep.task) for ep in episodes.values()]
    with JsonlWriter(path.with_name(f"{spec.db_id}.llm_calls.jsonl")) as calls:
        content = compute_recommendations(
            train, [(s.id, s.question) for s in load_split(spec).test],
            Embedder(transport=transport, on_call=calls.write, max_retries=spec.cfg.llm.max_retries),
        )
    content.update(benchmark=spec.benchmark, suite=spec.db_id,
                   fp8=json.loads((corpus_dir / "corpus.json").read_text(encoding="utf-8"))["fp8"])
    write_recommendations(path, content)
    return path


def build(
    config_path: str | Path, benchmark: str, suite: str, trial: int, *,
    transport: Optional[Any] = None, api_key: str | None = None, traverse_runner: Optional[Any] = None,
) -> Path:
    """Build the corpus and recommendations ``wikitrace run`` reads; return the corpus dir."""
    spec = resolve_spec(config_path, benchmark, suite, trial, transport=transport, api_key=api_key,
                        traverse_runner=traverse_runner)
    transport = _transport(spec)
    corpus_dir = ensure_corpus(spec, transport)
    ensure_recommendations(spec, corpus_dir, transport)
    return corpus_dir
