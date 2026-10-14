"""Gain and Regression per suite, per benchmark and averaged over suites, with bootstrap 95% intervals."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260929
RESAMPLES = 10_000
NO_MEMORY = "no-memory"
FLOOR = "floor"
BASELINE_TRIALS = (0, 1, 2)
METRICS = ("gain", "regression")
PAPER_SUITES = ("valid-seen", "valid-unseen", "scienceworld", "plancraft-seen", "plancraft-unseen", "retail",
                "test-normal", "test-challenge")


def suite_order(name: str) -> tuple[int, str]:
    """Canonical suite order, so the bootstrap draws do not depend on CSV row order."""
    return (PAPER_SUITES.index(name), "") if name in PAPER_SUITES else (len(PAPER_SUITES), name)


def task_means(df: pd.DataFrame, ids: list[str], trials) -> np.ndarray:
    """Per-task mean outcome over the given trials, NaN where a task has none."""
    sub = df[df.trial.isin(trials)]
    return sub.groupby("sample_id").outcome.mean().reindex(ids).to_numpy(float)


class Suite:
    def __init__(self, name: str, df: pd.DataFrame):
        self.name = name
        self.benchmark = df.benchmark.iloc[0]
        self.ids = sorted(df.sample_id.unique())
        nm = df[df.method == NO_MEMORY]
        self.b = (task_means(nm, self.ids, BASELINE_TRIALS) * 3 >= 2).astype(int)
        self.p = {m: task_means(g, self.ids, BASELINE_TRIALS)
                  for m, g in df.groupby("method", sort=False) if m != NO_MEMORY}
        floor_trials = sorted(t for t in nm.trial.unique() if t >= 3)
        if floor_trials:
            self.p[FLOOR] = task_means(nm, self.ids, floor_trials)

    def parts(self, method: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
        """Per-task numerator and eligibility of a metric."""
        p = self.p[method]
        elig = (self.b == 0) if metric == "gain" else (self.b == 1)
        num = p if metric == "gain" else 1 - p
        present = ~np.isnan(p)
        return np.where(elig & present, num, 0.0), (elig & present).astype(float)


def pooled(suites: list[Suite], weights: list[np.ndarray | None], method: str, metric: str):
    num = den = 0
    for s, w in zip(suites, weights):
        a, d = s.parts(method, metric)
        num = num + (a.sum() if w is None else w @ a)
        den = den + (d.sum() if w is None else w @ d)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 * np.asarray(num, float) / np.asarray(den, float)


def ci(draws: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def read_rows(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"sample_id": str, "method": str, "suite": str, "benchmark": str})
    df["trial"] = df.trial.astype(int)
    df["outcome"] = df.outcome.astype(float)
    return df


def validate(df: pd.DataFrame) -> None:
    """Raise unless every task of a suite has the same trials per method, no-memory's including 0-2."""
    keys = ["suite", "method", "sample_id", "trial"]
    if len(dup := df[df.duplicated(keys)]):
        raise ValueError(f"duplicate row ({', '.join(map(str, dup.iloc[0][keys]))})")
    for suite in sorted(df.suite.unique(), key=suite_order):
        sdf = df[df.suite == suite]
        for method in sorted(sdf.method.unique()):
            have = sdf[sdf.method == method].groupby("sample_id").trial.agg(frozenset).to_dict()
            want = Counter(have.values()).most_common(1)[0][0]
            if method == NO_MEMORY:
                want |= set(BASELINE_TRIALS)
            for sample_id in sorted(sdf.sample_id.unique()):
                if (got := have.get(sample_id, frozenset())) != want:
                    raise ValueError(f"({suite}, {method}, {sample_id}) has trials {sorted(got)}, "
                                     f"expected {sorted(want)}")


def compute(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Tables `suite`, `benchmark` and `average`, one row per (metric, unit, method)."""
    validate(df)
    suites = sorted((Suite(name, g) for name, g in df.groupby("suite")), key=lambda s: suite_order(s.name))
    rng = np.random.default_rng(SEED)
    weights = {s.name: rng.multinomial(len(s.ids), np.full(len(s.ids), 1 / len(s.ids)),
                                       size=RESAMPLES).astype(float) for s in suites}
    methods = list(dict.fromkeys(m for s in suites for m in s.p))
    suite_rows, bench_rows, avg_rows = [], [], []
    for metric in METRICS:
        for method in methods:
            points, draws = [], []
            for s in (s for s in suites if method in s.p):
                pt, dr = float(pooled([s], [None], method, metric)), pooled([s], [weights[s.name]], method, metric)
                points.append(pt)
                draws.append(dr)
                suite_rows.append(dict(metric=metric, benchmark=s.benchmark, suite=s.name, method=method,
                                       eligible_tasks=int(s.parts(method, metric)[1].sum()), estimate=pt,
                                       ci=ci(dr)))
            for bench in dict.fromkeys(s.benchmark for s in suites):
                group = [s for s in suites if s.benchmark == bench and method in s.p]
                if group:
                    bench_rows.append(dict(
                        metric=metric, benchmark=bench, method=method,
                        eligible_tasks=int(sum(s.parts(method, metric)[1].sum() for s in group)),
                        estimate=float(pooled(group, [None] * len(group), method, metric)),
                        ci=ci(pooled(group, [weights[s.name] for s in group], method, metric))))
            avg_rows.append(dict(metric=metric, method=method, suites=len(points), estimate=float(np.mean(points)),
                                 ci=ci(np.mean(draws, axis=0))))
    out = {}
    for name, rows in (("suite", suite_rows), ("benchmark", bench_rows), ("average", avg_rows)):
        t = pd.DataFrame(rows)
        t[["ci_lo", "ci_hi"]] = pd.DataFrame(t.pop("ci").tolist(), index=t.index)
        out[name] = t
    return out


def write(tables: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(out_dir / f"gain_regression_{name}.csv", index=False, float_format="%.4f")
