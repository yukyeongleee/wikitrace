"""The resolved inputs of one build or run, and the paths a run writes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from wikitrace.config import AppConfig, Split, load_config, make_split
from wikitrace.envs import DomainAdapter, HarnessSpec, config_view_for, raw_shape_for, resolve_benchmark
from wikitrace.memory.store import RawLayerShape, with_no_memory
from wikitrace.run.storage import git_info, run_fingerprint

METHODS = ("no-memory", "kinj", "wikitrace")


@dataclass(frozen=True)
class RunSpec:
    """The config, adapters and fingerprint of one suite and trial."""

    cfg: AppConfig
    config_path: str | Path
    benchmark: str
    db_id: str
    trial: int
    dom: DomainAdapter
    harness: HarnessSpec
    raw_shape: RawLayerShape
    fp: Mapping[str, str]
    git: Mapping[str, str | bool]
    transport: Optional[Any] = None
    api_key: str | None = None
    traverse_runner: Optional[Any] = None

    @property
    def domain(self) -> str:
        return self.dom.name

    @property
    def output_root(self) -> Path:
        return Path(self.cfg.experiment.output_root)


@dataclass(frozen=True)
class RunPaths:
    """Where one run writes."""

    run_dir: Path
    corpus_mount: Path
    test_dir: Path
    metrics_dir: Path

    @classmethod
    def under(cls, run_dir: Path) -> "RunPaths":
        return cls(run_dir=run_dir, corpus_mount=run_dir / "test" / "_corpus",
                   test_dir=run_dir / "test", metrics_dir=run_dir / "metrics")


def resolve_spec(
    config_path: str | Path, benchmark: str, suite: str, trial: int, *,
    transport: Optional[Any] = None, api_key: str | None = None, traverse_runner: Optional[Any] = None,
) -> RunSpec:
    """Load the config and bind the benchmark's adapters for one suite and trial."""
    cfg = load_config(config_path)
    dom = resolve_benchmark(benchmark, include_evidence=cfg.dataset.include_evidence)
    harness = dom.benchmark.harness_spec(suite)
    if not harness.terminal:
        raise ValueError(f"{benchmark}/{suite} does not run on the terminal harness")
    git = git_info()
    return RunSpec(
        cfg=cfg, config_path=config_path, benchmark=benchmark, db_id=suite, trial=trial, dom=dom,
        harness=harness, raw_shape=with_no_memory(raw_shape_for(dom, suite)),
        fp=run_fingerprint(cfg, git, config_view_for(dom.benchmark)), git=git,
        transport=transport, api_key=api_key, traverse_runner=traverse_runner,
    )


def load_split(spec: RunSpec) -> Split:
    """The suite's train and test halves."""
    return make_split(spec.dom.benchmark.load_suite(spec.cfg.dataset, spec.db_id), spec.cfg.dataset,
                      spec.db_id)
