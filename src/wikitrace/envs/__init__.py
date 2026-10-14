"""Benchmark environments and their registry; importing this package registers them."""

from __future__ import annotations

from wikitrace.envs.base import (
    BenchmarkAdapter,
    DomainAdapter,
    HarnessSpec,
    MemoryBinding,
    Metrics,
    config_view_for,
    raw_shape_for,
    train_group_for,
    register_benchmark,
    register_domain,
    resolve,
)

# Built-in registrations, by import side effect: the adapters, and the prompt roles
# whose templates each domain owns.
from wikitrace.envs.textgame import prompts as _textgame_prompts  # noqa: F401
from wikitrace.envs.appworld import prompts as _appworld_prompts  # noqa: F401
from wikitrace.envs.tau3 import prompts as _tau3_prompts  # noqa: F401
from wikitrace.envs.textgame.benchmarks import alfworld as _alfworld  # noqa: F401
from wikitrace.envs.textgame.benchmarks import scienceworld as _scienceworld  # noqa: F401
from wikitrace.envs.textgame.benchmarks import plancraft as _plancraft  # noqa: F401
from wikitrace.envs.textgame import adapter as _textgame  # noqa: F401
from wikitrace.envs.appworld.benchmarks import appworld as _appworld_bench  # noqa: F401
from wikitrace.envs.appworld import adapter as _appworld  # noqa: F401
from wikitrace.envs.tau3.benchmarks import tau3 as _tau3_bench  # noqa: F401
from wikitrace.envs.tau3 import adapter as _tau3  # noqa: F401

# The domain each benchmark runs in, and the suites the paper reports.
DOMAIN_OF = {"alfworld": "textgame", "scienceworld": "textgame", "plancraft": "textgame",
             "tau3": "tau3", "appworld": "appworld"}
SUITES = {"alfworld": ("valid-seen", "valid-unseen"), "scienceworld": ("scienceworld",),
          "plancraft": ("plancraft-seen", "plancraft-unseen"), "tau3": ("retail",),
          "appworld": ("test-normal", "test-challenge")}


def resolve_benchmark(benchmark: str, *, include_evidence: bool = True) -> DomainAdapter:
    """The domain adapter bound to ``benchmark``."""
    if benchmark not in DOMAIN_OF:
        raise KeyError(f"unknown benchmark: {benchmark!r} (have {sorted(DOMAIN_OF)})")
    return resolve(DOMAIN_OF[benchmark], benchmark, include_evidence=include_evidence)


__all__ = [
    "DOMAIN_OF",
    "SUITES",
    "resolve_benchmark",
    "BenchmarkAdapter",
    "DomainAdapter",
    "HarnessSpec",
    "MemoryBinding",
    "Metrics",
    "config_view_for",
    "raw_shape_for",
    "train_group_for",
    "register_benchmark",
    "register_domain",
    "resolve",
]
