"""The citation turn is costed apart from the generator, and its absence is an event."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from wikitrace.config import EventLogger
from wikitrace.run.evaluate import _record_traverse_cost
from wikitrace.memory.citation import CITATION_ROLE
from wikitrace.run.storage import write_summary

_CALLS = [
    {"generation_id": "g1", "prompt_tokens": 100, "completion_tokens": 10, "cost": 0.5},
    {"generation_id": "g2", "prompt_tokens": 50, "completion_tokens": 5, "cached_tokens": 40,
     "cost": 0.1},
]


def _record(tmp_path: Path, turns: list[dict], *, citation_expected: bool):
    recorded: list = []
    llm = SimpleNamespace(on_call=recorded.append)
    events = EventLogger(tmp_path / "events.jsonl")
    log = SimpleNamespace(
        phase="solver", usage={}, usage_calls=list(_CALLS), stopped_by="",
        truncated_calls=0, turns=turns,
    )
    _record_traverse_cost(
        SimpleNamespace(phase_logs=[log]), llm, events, sample_id="s1", phase="test",
        step_n=1, model_requested="m", citation_expected=citation_expected,
    )
    events.close()
    names = [
        json.loads(line)["event"]
        for line in (tmp_path / "events.jsonl").read_text().splitlines() if line
    ]
    return [r.role for r in recorded], names


def test_the_call_the_citation_turn_names_is_costed_under_the_citation_role(tmp_path):
    turns = [{"type": "episode_over"},
             {"type": "citation", "text": '{"K": []}', "parsed": {"K": []}, "response_id": "g2"}]
    roles, names = _record(tmp_path, turns, citation_expected=True)
    assert roles == ["solver", CITATION_ROLE]
    assert names == []


def test_a_citation_turn_answered_after_a_blocked_call_costs_both_calls_as_citation(tmp_path):
    turns = [{"type": "episode_over"},
             {"type": "citation", "text": '{"K": []}', "parsed": {"K": []}, "response_id": "g2",
              "missed_response_ids": ["g1"]}]
    roles, _ = _record(tmp_path, turns, citation_expected=True)
    assert roles == [CITATION_ROLE, CITATION_ROLE]


def test_a_citation_turn_without_an_answering_call_leaves_every_call_under_the_phase(tmp_path):
    turns = [{"type": "episode_over"},
             {"type": "citation", "text": "", "parsed": {"K": []}, "response_id": ""}]
    roles, names = _record(tmp_path, turns, citation_expected=True)
    assert roles == ["solver", "solver"]
    assert names == []


def test_a_citation_call_missing_from_the_usage_never_takes_the_solvers_last_call(tmp_path):
    """A cancelled or failed citation stream has an id but no usage row."""
    turns = [{"type": "episode_over"},
             {"type": "citation", "text": "", "parsed": {"K": []}, "response_id": "g9"}]
    roles, names = _record(tmp_path, turns, citation_expected=True)
    assert roles == ["solver", "solver"]
    assert names == []


def test_an_unparseable_answer_is_one_event_and_no_failed_sample(tmp_path):
    turns = [{"type": "episode_over"},
             {"type": "citation", "text": "I used nothing.", "parsed": None, "response_id": "g2"}]
    roles, names = _record(tmp_path, turns, citation_expected=True)
    assert roles == ["solver", CITATION_ROLE]
    assert names == ["citation_unparsed"]


def test_a_missing_citation_turn_is_skipped_only_when_one_was_expected(tmp_path):
    turns = [{"type": "function_call"}]
    roles, names = _record(tmp_path, turns, citation_expected=True)
    assert roles == ["solver", "solver"]
    assert names == ["citation_skipped"]
    roles, names = _record(tmp_path / "none", turns, citation_expected=False)
    assert roles == ["solver", "solver"]
    assert names == []


def test_the_summary_prices_the_citation_beside_the_generator_totals(tmp_path):
    rows = [
        {"call_id": "a", "role": "solver", "phase": "test", "sample_id": "s",
         "prompt_tokens": 100, "completion_tokens": 10, "cached_tokens": 0, "cost": 0.5},
        {"call_id": "b", "role": CITATION_ROLE, "phase": "test", "sample_id": "s",
         "prompt_tokens": 50, "completion_tokens": 5, "cached_tokens": 40, "cost": 0.1},
    ]
    (tmp_path / "llm_calls.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    write_summary(tmp_path, {}, [], 0)
    api = json.loads((tmp_path / "metrics" / "summary.json").read_text())["api"]
    assert api["citation_cost"] == 0.1
    assert api["totals"]["calls"] == 1
    assert api["totals"]["prompt_tokens"] == 100
    assert api["totals"]["cached_tokens"] == 0
    assert api["totals"]["cost"] == 0.5
    assert api["by_role"][CITATION_ROLE]["cached_tokens"] == 40
