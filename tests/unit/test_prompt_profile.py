"""PromptProfile: the wiki-manager prompts are parameterized on the domain and benchmark.
A profile also describes ONE SUITE's harness, which is why it is built from a suite id:
the environment is a per-suite fact, and composing it from the benchmark alone lies."""

from __future__ import annotations

import pytest

from wikitrace.envs import resolve
from wikitrace.envs.base import _BENCHMARKS, _DOMAINS
from wikitrace.harness.prompts import render_system

# The environment family's roles: one compiler template per outcome, and the memory method
# differs by a template variable rather than by a file.
_ENV_ROLES = {
    "compiler_success": ("compiler", "success"),
    "compiler_failure": ("compiler", "failure"),
}

# The suite each benchmark's prompts are rendered for.
_DEFAULT_SUITE = {
    "alfworld": "valid-seen",
    "scienceworld": "scienceworld",
    "plancraft": "plancraft-seen",
    "appworld": "test-normal",
    "tau3": "retail",
}


def _profile(domain: str, benchmark: str, suite_id: str | None = None):
    return resolve(domain, benchmark).prompt_profile(
        suite_id if suite_id is not None else _DEFAULT_SUITE[benchmark]
    )


def _render_all(domain: str, benchmark: str, suite_id: str | None = None) -> dict[str, str]:
    profile = _profile(domain, benchmark, suite_id)
    dump = profile.model_dump()
    return {
        k: render_system(*rv, memory=True, confidence=False, **dump)
        for k, rv in _ENV_ROLES.items()
    }


@pytest.mark.parametrize("benchmark", sorted(_BENCHMARKS))
def test_every_registered_pair_renders_every_role(benchmark):
    # StrictUndefined raises on a missing slot, so rendering IS the assertion: a
    # template referencing a slot no adapter supplies fails loudly here, not in a
    # paid live run.
    domain = _BENCHMARKS[benchmark]().domain
    assert domain in _DOMAINS
    for text in _render_all(domain, benchmark).values():
        assert text.strip()
        assert "{{" not in text


def test_compiler_profile_helper_leaves_a_seedless_corpus_untouched(tmp_path):
    """A seedless corpus leaves the profile itself, not a copy that differs."""
    from wikitrace.run.evaluate import _compiler_profile

    dom = resolve("textgame", "alfworld")
    assert _compiler_profile(dom, "valid-seen", tmp_path) == dom.prompt_profile("valid-seen")
