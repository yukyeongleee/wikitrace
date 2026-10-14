"""Adapter registry unit tests (offline)."""

from __future__ import annotations

import pytest

from wikitrace.envs import resolve


def test_resolve_binds_a_benchmark_to_its_domain():
    dom = resolve("textgame", "alfworld")
    assert dom.name == "textgame"
    assert dom.benchmark.name == "alfworld"
    assert dom.benchmark.domain == "textgame"


def test_resolve_unknown_names_raise():
    with pytest.raises(KeyError):
        resolve("textgame", "nope")
    with pytest.raises(KeyError):
        resolve("nope", "alfworld")


def test_resolve_domain_benchmark_mismatch_raises():
    from wikitrace.envs.base import register_domain

    register_domain("other-domain", lambda bench: None)
    try:
        with pytest.raises(ValueError):
            resolve(domain="other-domain", benchmark="alfworld")
    finally:
        from wikitrace.envs import base

        base._DOMAINS.pop("other-domain", None)
