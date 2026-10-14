"""HarnessSpec: the generator harness is a property of the SUITE, resolved for every
registered benchmark, and a resolved spec is an immutable value."""

from __future__ import annotations

import pytest

from wikitrace.envs import resolve
from wikitrace.envs.base import HarnessSpec

PAPER = [("textgame", "alfworld"), ("textgame", "scienceworld"), ("textgame", "plancraft"),
         ("tau3", "tau3"), ("appworld", "appworld")]


def test_harness_spec_is_frozen():
    # A resolved spec is a value, not a knob: nothing may move the harness mid-run.
    spec = resolve("textgame", "alfworld").benchmark.harness_spec("valid-seen")
    with pytest.raises(Exception):
        spec.terminal = True  # type: ignore[misc]


@pytest.mark.parametrize("domain,benchmark", PAPER)
def test_every_registered_benchmark_resolves_a_spec(domain, benchmark):
    # The runner calls harness_spec BEFORE load_suite validates the id, so resolution
    # must be total: an unknown suite takes the benchmark's default.
    assert isinstance(resolve(domain, benchmark).benchmark.harness_spec("nope"), HarnessSpec)


