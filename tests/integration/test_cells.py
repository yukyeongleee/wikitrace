"""`build` then `run` over the offline ALFWorld game: the layout, the records and the refusals."""

from __future__ import annotations

import json
import shutil

import pytest

from tests.regression.test_method_prompts import _Transport
from wikitrace.run.build import build
from wikitrace.run.cell import run_cell

pytestmark = pytest.mark.usefixtures("fake_game")
SUITE = ("alfworld", "valid-seen")


@pytest.fixture
def cfg(temp_config_writer):
    return temp_config_writer({"dataset": {"test_size": 2, "train_size": 2}})


def _rows(run_dir):
    return [json.loads(x) for x in (run_dir / "metrics" / "samples.jsonl").read_text().splitlines()]


@pytest.mark.slow
def test_build_then_every_arm_lands_where_the_readme_says(cfg, tmp_path):
    transport = _Transport()
    corpus = build(cfg, *SUITE, 1, transport=transport)
    outputs = tmp_path / "outputs"
    assert corpus == outputs / "corpus" / "alfworld" / "alfworld" / "rep-01"
    assert (corpus / "recommendations" / "valid-seen.json").exists()
    for method, label in (("no-memory", "no-memory"), ("kinj", "00I-inject"),
                       ("wikitrace", "RCI-preload-usermem-descend-sourcecheck")):
        run_dir = run_cell(cfg, *SUITE, 1, method, transport=transport)
        assert run_dir == outputs / "alfworld" / "valid-seen" / method / "trial-01"
        record = json.loads((run_dir / "run.json").read_text())
        assert (record["status"], record["label"], record["trial"]) == ("completed", label, 1)
        assert ("corpus" in record) == (method != "no-memory")
        assert (run_dir / "metrics" / "summary.json").exists()
        rows = _rows(run_dir)
        assert [(r["method"], r["suite"], r["trial"]) for r in rows] == [(method, "valid-seen", 1)] * 2
        assert {r["outcome"] for r in rows} <= {0, 1} and all(r["value"] == r["outcome"] for r in rows)


@pytest.mark.slow
def test_a_killed_run_resumes_and_a_completed_one_is_refused(cfg):
    transport = _Transport()
    run_dir = run_cell(cfg, *SUITE, 0, "no-memory", transport=transport)
    first = _rows(run_dir)
    # Keep the first sample only, as a run killed after it would have.
    (run_dir / "metrics" / "summary.json").unlink()
    (run_dir / "metrics" / "samples.jsonl").write_text(json.dumps(first[0]) + "\n")
    record = json.loads((run_dir / "run.json").read_text())
    (run_dir / "run.json").write_text(json.dumps({**record, "status": "running"}))
    run_cell(cfg, *SUITE, 0, "no-memory", transport=transport)
    assert [r["sample_id"] for r in _rows(run_dir)] == [r["sample_id"] for r in first]
    assert json.loads((run_dir / "run.json").read_text())["resumed_after"] == 1
    with pytest.raises(RuntimeError, match="completed run"):
        run_cell(cfg, *SUITE, 0, "no-memory", transport=transport)


@pytest.mark.slow
def test_a_corpus_override_is_read_and_recorded_and_refused_for_no_memory(cfg, tmp_path):
    transport = _Transport()
    built = build(cfg, *SUITE, 0, transport=transport)
    other = tmp_path / "elsewhere"
    shutil.copytree(built, other)
    shutil.rmtree(other / "recommendations")
    stamp = json.loads((other / "corpus.json").read_text())
    (other / "corpus.json").write_text(json.dumps({**stamp, "fp8": "deadbeef"}))
    run_dir = run_cell(cfg, *SUITE, 3, "kinj", corpus=other, transport=transport)
    record = json.loads((run_dir / "run.json").read_text())
    assert record["corpus"] == other.as_posix() and record["corpus_fp8"] == "deadbeef"
    assert record["corpus_fingerprint_mismatch"] is True
    assert (other / "recommendations" / "valid-seen.json").exists()
    with pytest.raises(ValueError, match="no-memory"):
        run_cell(cfg, *SUITE, 0, "no-memory", corpus=other, transport=transport)
