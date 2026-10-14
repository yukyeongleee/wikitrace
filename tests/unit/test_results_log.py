"""The rebuild record lives beside the memory layer, not inside it.
``raw/`` is copied wholesale into every wiki, so a record carrying a gold-aware
rationale would ride along; moving it out is what makes that structural."""

from __future__ import annotations

import json

import pytest

from wikitrace.config import (
    RESULTS_LOG,
    append_result,
    read_result_json,
    reset,
)


def _step(corpus, name="step-0000"):
    d = corpus / "raw" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_the_record_is_not_inside_the_layer_that_gets_copied(tmp_path):
    """The whole point: `build_wiki` copies `corpus/raw`, so the record must be outside it."""
    step = _step(tmp_path)
    append_result(tmp_path, step.name, {"correct": True, "grade_reason": "judge said so"})

    assert not list((tmp_path / "raw").rglob("*.json")), (
        "nothing of the rebuild record may land under raw/"
    )
    # And the sensitive field is genuinely off that path.
    assert "grade_reason" not in "".join(
        p.read_text() for p in (tmp_path / "raw").rglob("*") if p.is_file()
    )


def test_round_trip_through_the_step_dir(tmp_path):
    step = _step(tmp_path)
    append_result(tmp_path, step.name, {"correct": True, "answer_letter": "B"})
    assert json.loads(read_result_json(step)) == {"correct": True, "answer_letter": "B"}


def test_a_rerun_appends_and_the_later_record_wins(tmp_path):
    """`rerun_failed` regenerates ONE step. Appending keeps what was repaired visible."""
    step = _step(tmp_path)
    append_result(tmp_path, step.name, {"correct": False, "generation_error": "timeout"})
    append_result(tmp_path, step.name, {"correct": True, "generation_error": ""})

    assert json.loads(read_result_json(step))["correct"] is True
    assert len((tmp_path / RESULTS_LOG).read_text().splitlines()) == 2, (
        "the earlier attempt stays as history — this is a log, not a map"
    )


def test_steps_do_not_read_each_others_records(tmp_path):
    a, b = _step(tmp_path, "step-0000"), _step(tmp_path, "step-0001")
    append_result(tmp_path, a.name, {"answer_letter": "A"})
    append_result(tmp_path, b.name, {"answer_letter": "B"})
    assert json.loads(read_result_json(a))["answer_letter"] == "A"
    assert json.loads(read_result_json(b))["answer_letter"] == "B"


def test_a_corpus_frozen_before_the_move_still_reads(tmp_path):
    """Every corpus built up to 2026-07-29 has it inside raw/. They must still compile."""
    step = _step(tmp_path)
    (step / "result.json").write_text('{"correct": true, "grade_reason": "legacy"}')
    assert json.loads(read_result_json(step))["grade_reason"] == "legacy"


def test_the_log_wins_over_a_legacy_file_for_the_same_step(tmp_path):
    """A repaired step must not read back the record it replaced."""
    step = _step(tmp_path)
    (step / "result.json").write_text('{"correct": false}')
    append_result(tmp_path, step.name, {"correct": True})
    assert json.loads(read_result_json(step))["correct"] is True


def test_reset_clears_the_log_so_a_smaller_split_leaves_no_orphans(tmp_path):
    step = _step(tmp_path)
    append_result(tmp_path, step.name, {"correct": True})
    reset(tmp_path)
    assert not (tmp_path / RESULTS_LOG).exists()
    reset(tmp_path)  # idempotent — a first build has no log to drop


def test_a_missing_record_says_where_it_looked(tmp_path):
    step = _step(tmp_path)
    with pytest.raises(FileNotFoundError, match="step-0000"):
        read_result_json(step)
