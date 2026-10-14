from __future__ import annotations

import pandas as pd
import pytest

from wikitrace.analysis.gain_regression import compute

# no-memory solves t0 and t1 on every trial; the method loses t0 and gains t2
OUTCOMES = {
    "no-memory": {"t0": 1, "t1": 1, "t2": 0, "t3": 0},
    "m": {"t0": 0, "t1": 1, "t2": 1, "t3": 0},
}


def synthetic_rows() -> pd.DataFrame:
    rows = [("b", "s", method, trial, task, outcome)
            for method, tasks in OUTCOMES.items()
            for trial in (range(6) if method == "no-memory" else range(3))
            for task, outcome in tasks.items()]
    return pd.DataFrame(rows, columns=["benchmark", "suite", "method", "trial", "sample_id", "outcome"])


@pytest.fixture(scope="module")
def tables():
    return compute(synthetic_rows())


def test_gain_and_regression_follow_the_definition(tables):
    t = tables["suite"].set_index(["metric", "method"])
    assert t.loc[("regression", "m"), "estimate"] == pytest.approx(50.0)
    assert t.loc[("gain", "m"), "estimate"] == pytest.approx(50.0)
    assert t.loc[("regression", "floor"), "estimate"] == pytest.approx(0.0)
    assert t.loc[("gain", "floor"), "estimate"] == pytest.approx(0.0)


def test_row_order_does_not_matter(tables):
    shuffled = compute(synthetic_rows().sample(frac=1, random_state=0))
    for name, t in tables.items():
        keys = [c for c in ("metric", "benchmark", "suite", "method") if c in t]
        pd.testing.assert_frame_equal(t.sort_values(keys).reset_index(drop=True),
                                      shuffled[name].sort_values(keys).reset_index(drop=True))


def test_a_dropped_row_is_refused():
    rows = synthetic_rows()
    drop = rows[(rows.method == "m") & (rows.trial == 0) & (rows.sample_id == "t0")].index[0]
    with pytest.raises(ValueError, match=r"^\(s, m, t0\) has trials \[1, 2\]"):
        compute(rows.drop(index=drop))


def test_a_missing_baseline_trial_is_refused():
    rows = synthetic_rows()
    rows = rows.drop(index=rows[(rows.method == "no-memory") & rows.trial.isin([1, 2])].index)
    with pytest.raises(ValueError, match=r"^\(s, no-memory, .+\) has trials \[0, 3, 4, 5\], expected \[0, 1"):
        compute(rows)
