"""`build` and `run` refuse a default corpus built under another fingerprint."""

from __future__ import annotations

import json

import pytest

from tests.conftest import FakeTransport
from wikitrace.run.build import build
from wikitrace.run.cell import run_cell

pytestmark = pytest.mark.usefixtures("fake_game")
SUITE = ("alfworld", "valid-seen")


@pytest.fixture
def stale(temp_config_writer, tmp_path):
    """A config and, at its trial-0 corpus path, a corpus stamped with another fingerprint."""
    cfg = temp_config_writer({"dataset": {"test_size": 2, "train_size": 2}})
    corpus = tmp_path / "outputs" / "corpus" / "alfworld" / "alfworld" / "rep-00"
    corpus.mkdir(parents=True)
    (corpus / "corpus.json").write_text(json.dumps({"fp8": "deadbeef", "insight_items": True}))
    return cfg, tmp_path / "outputs"


def test_build_refuses_a_corpus_of_another_fingerprint(stale):
    cfg, _ = stale
    with pytest.raises(RuntimeError, match=r"fingerprint deadbeef, the current config gives [0-9a-f]{8}"):
        build(cfg, *SUITE, 0, transport=FakeTransport())


def test_run_refuses_a_default_corpus_of_another_fingerprint(stale):
    cfg, outputs = stale
    with pytest.raises(RuntimeError, match=r"fingerprint deadbeef, the current config gives [0-9a-f]{8}"):
        run_cell(cfg, *SUITE, 0, "kinj", transport=FakeTransport())
    assert not (outputs / "alfworld").exists()
