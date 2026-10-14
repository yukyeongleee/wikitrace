"""Every environment wiki-manager prompt combination, pinned by hash."""

# The templates dispatch on several axes at once (role, outcome, memory, confidence),
# and one variant file per combination is only safe while every one is pinned byte-for-byte.

# The manifest holds one sha256 per combination. Direct execution validates the
# existing manifest without rewriting it; an intentional prompt change requires a
# separate reviewed fixture update.
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import wikitrace.envs  # noqa: F401  (registers the generator prompt roles)
from wikitrace.envs import resolve
from wikitrace.harness.prompts import render_system

MANIFEST = Path(__file__).parents[1] / "fixtures/prompts/prompt_render_hashes.json"

# The compiler's prompts: one file per outcome, with the memory
# clause and the confidence block as template variables.
ENV_WIKI_ROLES = (("compiler", "success"), ("compiler", "failure"))
ENV_WIKI_SUITES = (("textgame", "alfworld", "valid-seen"),
                   ("textgame", "scienceworld", "scienceworld"),
                   ("appworld", "appworld", "test-normal"))
def _renders() -> dict[str, str]:
    """{combination key: sha256 of the rendered system prompt}."""
    out: dict[str, str] = {}
    for domain, benchmark, suite_id in ENV_WIKI_SUITES:
        profile = resolve(domain, benchmark).prompt_profile(suite_id).model_dump()
        for role, variant in ENV_WIKI_ROLES:
            for memory, confidence in itertools.product((False, True), repeat=2):
                key = (f"env|benchmark={benchmark}|role={role}.{variant}"
                       f"|memory={memory}|confidence={confidence}")
                text = render_system(role, variant, memory=memory,
                                     confidence=confidence, **profile)
                out[key] = hashlib.sha256(text.encode()).hexdigest()
    return out


def test_every_traverse_protocol_combination_is_pinned():
    want = json.loads(MANIFEST.read_text())
    got = _renders()
    assert set(got) == set(want), (
        "The combination set changed. Intentional axis changes require "
        "a separately reviewed fixture update."
    )
    moved = sorted(k for k in want if want[k] != got[k])
    assert not moved, (
        f"{len(moved)} of {len(want)} rendered prompts changed, e.g.\n  "
        + "\n  ".join(moved[:5])
        + "\nIf intentional, prepare a separate reviewed fixture update."
    )


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description="Validate pinned prompt hashes without writing fixtures.").parse_args()
    test_every_traverse_protocol_combination_is_pinned()
    print("Pinned prompt hashes verified; fixture unchanged.")
