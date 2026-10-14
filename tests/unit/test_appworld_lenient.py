"""The lenient AppWorld answer rule and how it switches the success signal."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import wikitrace.envs  # noqa: F401  (the built-in registrations)
from wikitrace.config import DatasetConfig, load_config
from wikitrace.envs import config_view_for, resolve
from wikitrace.envs.appworld import adapter as adapter_mod
from wikitrace.envs.base import _BENCHMARKS
from wikitrace.envs.appworld.benchmarks import appworld as bench_mod
from wikitrace.envs.appworld.benchmarks.appworld import (
    AppWorldResult,
    gold_answer_is_null,
    lenient_tgc,
)

_ANSWER = {"requirement": "assert answers match.", "label": "no_op_fail", "trace": "..."}
_OTHER = {"requirement": "assert model changes match venmo.Payment", "label": None, "trace": "..."}


def test_lenient_tgc_forgives_only_a_lone_answer_check_on_a_null_gold():
    assert lenient_tgc(False, [_ANSWER], gold_answer_is_null=True)
    assert not lenient_tgc(False, [_ANSWER], gold_answer_is_null=False)
    assert not lenient_tgc(False, [_ANSWER, _OTHER], gold_answer_is_null=True)
    assert not lenient_tgc(False, [_OTHER], gold_answer_is_null=True)
    assert not lenient_tgc(False, [], gold_answer_is_null=True)
    assert lenient_tgc(True, [], gold_answer_is_null=False)


def test_gold_answer_is_read_from_the_ground_truth_file(tmp_path):
    for task, gold in (("a_1", None), ("b_1", "42")):
        d = tmp_path / "data" / "tasks" / task / "ground_truth"
        d.mkdir(parents=True)
        (d / "answer.json").write_text(json.dumps(gold))
    assert gold_answer_is_null(tmp_path, "a_1") and not gold_answer_is_null(tmp_path, "b_1")


def _result(tgc: bool, tgc_lenient: bool | None, scenario: str = "s") -> AppWorldResult:
    return AppWorldResult(
        sample_id=f"{scenario}_1", db_id="test-normal", question="q", scenario_id=scenario,
        tgc=tgc, tgc_lenient=tgc_lenient, tests_passed=1, tests_total=2,
    )


def test_unjudged_result_keeps_the_official_signal_everywhere():
    dom = resolve("appworld", "appworld")
    r = _result(tgc=False, tgc_lenient=None)
    m = dom.benchmark.metrics_from_result(r)
    assert (m.primary, m.success) == (0.0, False) and "tgc_lenient" not in m.extra
    assert "tgc_lenient" not in r.model_dump()   # a flag-off corpus record is unchanged
    assert "- success: False" in adapter_mod._metadata_text(r)
    assert set(dom.benchmark.summarize([m])) == {
        "n", "success_rate", "success_rate_answered", "refusal_rate", "secondary_mean",
        "tgc", "sgc", "n_scenarios",
    }


def test_judged_result_drives_success_while_primary_stays_official():
    dom = resolve("appworld", "appworld")
    r = _result(tgc=False, tgc_lenient=True)
    m = dom.benchmark.metrics_from_result(r)
    assert (m.primary, m.success, m.extra["tgc_lenient"]) == (0.0, True, True)
    assert "- success: True" in adapter_mod._metadata_text(r)
    assert dom.compilation_inputs(r).success is True
    assert "outcome: success" in adapter_mod.build_outcome_text(r)
    rows = [m, dom.benchmark.metrics_from_result(_result(True, True)),
            dom.benchmark.metrics_from_result(_result(False, False, scenario="t"))]
    s = dom.benchmark.summarize(rows)
    assert s["tgc"] == pytest.approx(1 / 3) and s["sgc"] == 0.0
    assert s["tgc_lenient"] == pytest.approx(2 / 3) and s["sgc_lenient"] == 0.5
    assert s["success_rate"] == s["tgc_lenient"]


def test_the_flag_reaches_the_sample_and_the_fingerprint(tmp_path, monkeypatch):
    d = tmp_path / "data" / "tasks" / "a_1"
    d.mkdir(parents=True)
    (d / "specs.json").write_text(json.dumps(
        {"instruction": "i", "supervisor": {"first_name": "J"}, "datetime": "2023"}))
    splits = tmp_path / "splits.json"
    entry = [{"id": "a_1", "scenario_id": "a", "difficulty": 1}]
    splits.write_text(json.dumps({"train": entry, "seen_test": [], "unseen_test": []}))
    monkeypatch.setattr(bench_mod, "SPLITS_PATH", splits)
    for flag in (False, True):
        cfg = DatasetConfig(appworld_root=str(tmp_path), appworld_lenient_answer=flag)
        (s,) = bench_mod.load_suite_samples(cfg, "test-normal")
        assert s.lenient_answer is flag
    assert "appworld_lenient_answer" in config_view_for(resolve("appworld", "appworld").benchmark).dataset_keys


def test_corpus_episodes_follow_the_lenient_judgement(tmp_path):
    from wikitrace.config import append_result
    from wikitrace.memory.compiler import read_episodes

    dom = resolve("appworld", "appworld")
    for n, lenient in enumerate((None, True)):
        step = f"step-{n:04d}"
        (tmp_path / "raw" / step).mkdir(parents=True)
        append_result(tmp_path, step, _result(tgc=False, tgc_lenient=lenient))
    assert [ep.outcome for ep in read_episodes(dom, tmp_path).values()] == ["failure", "success"]


def test_base_config_judges_appworld_leniently_and_only_appworld_sees_the_key():
    cfg = load_config(Path(__file__).resolve().parents[2] / "configs" / "base.yaml")
    assert cfg.dataset.appworld_lenient_answer is True
    assert DatasetConfig().appworld_lenient_answer is False
    seen = {n for n in _BENCHMARKS
            if "appworld_lenient_answer" in config_view_for(_BENCHMARKS[n]()).dataset_keys}
    assert seen == {"appworld"}
