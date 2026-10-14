from __future__ import annotations

from wikitrace.config import DatasetConfig
from wikitrace.envs.tau3.benchmarks import tau3 as bench_mod


def test_default_spec_is_unchanged_and_overrides_are_opt_in() -> None:
    spec = bench_mod.Tau3BenchmarkAdapter.spec_for("10")
    assert set(spec) == {"task_id", "user_model", "user_llm_args", "judge_model", "judge_llm_args"}
    over = bench_mod.Tau3BenchmarkAdapter.spec_for("10", "openrouter/openai/gpt-4.1", "clarified")
    assert over["user_model"] == "openrouter/openai/gpt-4.1" and over["judge_model"] == bench_mod.JUDGE_MODEL
    assert "reasoning_effort" not in over["user_llm_args"]
    assert over["user_guidelines_path"] == str(bench_mod.CLARIFIED_GUIDELINES_PATH)


def test_dataset_keys_default_to_the_official_setting() -> None:
    d = DatasetConfig()
    assert d.tau3_user_model is None and d.tau3_user_guidelines == "official"
