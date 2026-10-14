"""One test cell: an method over one suite and trial, from its memory to its summary."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from wikitrace.config import EventLogger, JsonlWriter, dump_resolved
from wikitrace.envs import MemoryBinding, config_view_for, train_group_for
from wikitrace.envs.appworld.workflow import exemplar_template
from wikitrace.harness.llm import LLMClient
from wikitrace.harness.llm.openrouter import provider_routing
from wikitrace.memory.compiler import (
    compilation_index, read_episodes, read_items, relinked_compilations, require_built_corpus,
)
from wikitrace.memory.insights import render_descend, render_inject, render_recommendation
from wikitrace.memory.mount import DESCEND_FILES, DESCEND_SCRIPT, build_wiki, frozen
from wikitrace.memory.recommend import (
    EMBEDDING_MODEL, K, Embedder, load_recommendations, recommend_at_reset, recommendation_path,
)
from wikitrace.run.build import _transport, built_fp8, ensure_recommendations
from wikitrace.run.evaluate import _default_traverse_runner, evaluate_test_samples
from wikitrace.run.resume import plan_resume
from wikitrace.run.spec import METHODS, RunPaths, RunSpec, load_split, resolve_spec
from wikitrace.run.storage import (
    corpus_dir_for, gather_model_ids, record_provider_calls, run_dir_for, update_run_json, write_run_json,
    write_summary,
)

LABELS = {"no-memory": "no-memory", "kinj": "00I-inject", "wikitrace": "RCI-preload-usermem-descend-sourcecheck"}
SOURCE_CHECK_VERSION = {"tau3": 2, "appworld": 3}


def source_check_version(benchmark: str) -> int:
    return SOURCE_CHECK_VERSION.get(benchmark, 1)


def label_for(method: str, benchmark: str) -> str:
    version = source_check_version(benchmark)
    return LABELS[method] + (str(version) if method == "wikitrace" and version > 1 else "")


def memory_kwargs_for(method: str, benchmark: str) -> dict[str, Any]:
    """The solver-prompt arguments an method's episodes run with."""
    if method == "no-memory":
        return {}
    if method == "kinj":
        return {"memory": True, "memory_layers": ("I",)}
    return {"memory": True, "memory_layers": ("R", "C", "I"), "descend_block": render_descend(benchmark),
            "source_check": True, "source_check_version": source_check_version(benchmark),
            "readonly_files": dict.fromkeys(DESCEND_FILES, DESCEND_SCRIPT)}


def _source_check_record(benchmark: str) -> dict[str, Any]:
    """WikiTrace's run.json fields; AppWorld's exemplar sits outside the prompt fingerprint."""
    version = source_check_version(benchmark)
    record: dict[str, Any] = {"source_check": True, **({"source_check_version": version} if version > 1 else {})}
    if benchmark == "appworld":
        record["exemplar_sha256"] = hashlib.sha256(
            exemplar_template(source_check=True, version=version).encode("utf-8")).hexdigest()
    return record


def _memory_for(spec: RunSpec, method: str, corpus: Path, items: list, episodes: dict, test: list,
                embedder: Callable[[], Embedder], transport: Any) -> Callable[[Any], MemoryBinding]:
    """Each test sample's binding: the inlined insights and its recommended train runs."""
    user_block = f"<memory_documents>\n{render_inject(items, source=method == 'wikitrace')}\n</memory_documents>"

    def bind(steps: list[str], **meta: Any) -> MemoryBinding:
        return MemoryBinding(memory_text=render_recommendation(steps, items, episodes),
                             user_prefix=user_block, meta=meta)

    if spec.harness.task_text_at_reset:
        recommend = recommend_at_reset([(ep.step, ep.sample_id, ep.task) for ep in episodes.values()], embedder())

        def at_reset(sample: Any) -> MemoryBinding:
            ranked = recommend(sample.question, sample.id)
            return bind([r["step"] for r in ranked], memory_query=sample.question, memory_recommended=ranked)
        return at_reset
    path = ensure_recommendations(spec, corpus, transport)
    ranking = load_recommendations(path, train={s: ep.sample_id for s, ep in episodes.items()},
                                   test_ids=[s.id for s in test])
    bound = {s.id: bind([step for step, _ in ranking[s.id]]) for s in test}
    return lambda sample: bound[sample.id]


def run_cell(
    config_path: str | Path, benchmark: str, suite: str, trial: int, method: str, *,
    corpus: str | Path | None = None, transport: Optional[Any] = None, api_key: str | None = None,
    traverse_runner: Optional[Any] = None,
) -> Path:
    """Run one method over one suite and trial; return its run directory."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r} (expected one of {METHODS})")
    if corpus is not None and method == "no-memory":
        raise ValueError("--corpus refused: the no-memory method reads no corpus")
    spec = resolve_spec(config_path, benchmark, suite, trial, transport=transport, api_key=api_key,
                        traverse_runner=traverse_runner)
    spec = replace(spec, transport=_transport(spec))
    if spec.traverse_runner is None:
        spec = replace(spec, traverse_runner=_default_traverse_runner(spec.cfg, spec.api_key, spec.harness))
    split = load_split(spec)
    corpus_dir = None
    items: list = []
    fp_mismatch = False
    if method != "no-memory":
        corpus_dir = require_built_corpus(Path(corpus) if corpus is not None else corpus_dir_for(
            spec.output_root, benchmark, train_group_for(spec.dom.benchmark, suite), trial)).resolve()
        built = built_fp8(corpus_dir)
        fp_mismatch = built != spec.fp["fp8"]
        if fp_mismatch:
            message = (f"{corpus_dir} was built under fingerprint {built or '(none)'}, the current config "
                       f"gives {spec.fp['fp8']}")
            if corpus is None:
                raise RuntimeError(f"{message}; rebuild it, or pass it with --corpus to run on it anyway")
            print(f"warning: {message}; running on it as --corpus asks", file=sys.stderr)
        items = read_items(spec.dom, corpus_dir)
    run_dir = run_dir_for(spec.output_root, benchmark, suite, method, trial)
    plan = plan_resume(run_dir, fp8=str(spec.fp["fp8"]), test_ids=[s.id for s in split.test])
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = RunPaths.under(run_dir)
    label = label_for(method, benchmark)
    events = EventLogger(run_dir / "events.jsonl")
    llm_calls = JsonlWriter(run_dir / "llm_calls.jsonl")
    llm = LLMClient(spec.cfg.models, spec.transport, spec.cfg.llm.max_retries, on_call=llm_calls.write)
    dump_resolved(spec.cfg, run_dir / "config.resolved.yaml", config_view_for(spec.dom.benchmark))
    write_run_json(run_dir, {
        "method": method, "label": label, "benchmark": benchmark, "domain": spec.domain,
        "suite": suite, "db_id": suite, "trial": trial,
        "fp8": spec.fp["fp8"], "fingerprint": spec.fp["fingerprint"], "config_hash": spec.fp["config_hash"],
        "prompts_hash": spec.fp["prompts_hash"], "git_commit": spec.git["commit"], "git_dirty": spec.git["dirty"],
        "seed": spec.cfg.experiment.seed, "models": {"default": spec.cfg.models.default.model},
        "model_validation": spec.transport.validate_models(gather_model_ids(spec.cfg.models)),
        "openrouter_routing": provider_routing(spec.cfg.llm.openrouter_providers),
        **({"data_file_hashes": h} if (h := _call(spec.dom.benchmark, "data_file_hashes")) else {}),
        **({"data_version": v} if (v := _call(spec.dom.benchmark, "data_version", spec.cfg.dataset)) else {}),
        **(_corpus_record(corpus_dir, len(items)) if corpus_dir is not None else {}),
        **({"corpus_fingerprint_mismatch": True} if fp_mismatch else {}),
        **({"memory_at": "user"} if method != "no-memory" else {}),
        **({"descend": True, "tools": {"files": list(DESCEND_FILES),
                                       "sha256": hashlib.sha256(DESCEND_SCRIPT.read_bytes()).hexdigest()},
            **_source_check_record(benchmark)} if method == "wikitrace" else {}),
        **({"resumed_at": datetime.now(timezone.utc).isoformat(), "resumed_after": len(plan.done)}
           if plan.done else {}),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    })
    events.emit("run_started", method=method, db_id=suite, trial=trial, items=len(items))
    try:
        memory_for, kwargs, mount = None, memory_kwargs_for(method, benchmark), None
        if corpus_dir is not None:
            episodes = read_episodes(spec.dom, corpus_dir)
            memory_for = _memory_for(
                spec, method, corpus_dir, items, episodes, split.test,
                lambda: Embedder(transport=spec.transport, on_call=llm_calls.write,
                                 max_retries=spec.cfg.llm.max_retries), spec.transport)
            if spec.harness.task_text_at_reset:
                update_run_json(run_dir, recommendations={"at_reset": True, "k": K, "model": EMBEDDING_MODEL})
            else:
                rec = recommendation_path(corpus_dir, suite)
                update_run_json(run_dir, recommendations={
                    "path": rec.as_posix(), "sha256": hashlib.sha256(rec.read_bytes()).hexdigest()})
        if method == "wikitrace":
            mount = build_wiki(
                paths.corpus_mount / "corpus", items=items, compilations=relinked_compilations(corpus_dir, items),
                source_corpus=corpus_dir, dom=spec.dom, raw_shape=spec.raw_shape,
                compilation_index=compilation_index(spec.dom, corpus_dir, spec.raw_shape),
            )
        with frozen(mount):
            metrics, n_failed = evaluate_test_samples(
                spec, paths, split=split, plan=plan, corpus_dir=mount, llm=llm, events=events, method=method,
                memory_for=memory_for, memory_kwargs=kwargs,
            )
        write_summary(run_dir, spec.dom.benchmark.summarize(metrics), metrics, n_failed)
        update_run_json(run_dir, status="completed", finished_at=datetime.now(timezone.utc).isoformat())
        events.emit("run_completed", method=method)
        return run_dir
    except Exception as exc:  # noqa: BLE001
        events.emit("run_failed", method=method, error=f"{type(exc).__name__}: {exc}")
        try:
            update_run_json(run_dir, status="failed", error=f"{type(exc).__name__}: {exc}")
        except Exception:  # noqa: BLE001
            pass
        raise
    finally:
        events.close()
        try:
            llm_calls.close()
        except Exception:  # noqa: BLE001
            pass
        record_provider_calls(run_dir)


def _call(obj: Any, name: str, *args: Any) -> Any:
    fn = getattr(obj, name, None)
    return fn(*args) if callable(fn) else None


def _corpus_record(corpus_dir: Path, n_items: int) -> dict[str, Any]:
    stamp = json.loads((corpus_dir / "corpus.json").read_text(encoding="utf-8"))
    manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    return {"corpus": corpus_dir.as_posix(), "corpus_fp8": stamp.get("fp8", ""),
            "corpus_split_hash": manifest.get("split_hash", ""), "item_count": n_items}
