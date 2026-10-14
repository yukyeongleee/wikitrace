# Gain and Regression

`wikitrace analyze` measures how a memory method changes outcomes relative to an agent without memory.

## Definitions

Each task has a baseline outcome B_i, the majority outcome of three no-memory trials (trials 0, 1 and 2).

- **Gain** is the share of tasks with B_i = 0 that a method solves.
- **Regression** is the share of tasks with B_i = 1 that a method fails.

Both are computed per trial and averaged over trials. They are reported per suite, per benchmark (its suites pooled) and as the average over suites. Each estimate has a 95% bootstrap interval over tasks (seed 20260929, 10,000 resamples).

No-memory trials 3 and higher, when present, are scored against B_i in the same way and give the **No Memory floor**: the Gain and Regression that rerunning the same agent produces by chance. The floor averages every such trial, so it matches the paper's floor (trials 3, 4 and 5) only when exactly those trials are present.

## Input

One CSV row per task and trial:

```
benchmark,suite,method,trial,sample_id,outcome,value
```

- `outcome` is 1 for a success and 0 otherwise. `value` is the benchmark's raw score and is not read by the statistics.
- Every task needs `no-memory` rows for trials 0, 1 and 2.
- Within a suite, every task must have the same trials under each method. `stats` refuses an input with a missing or duplicated row.
- Every `method` value other than `no-memory` is treated as a memory method.

## Commands

```bash
uv sync --extra analysis
wikitrace analyze export --runs outputs/ --out analysis/per_sample.csv
wikitrace analyze stats --per-sample analysis/per_sample.csv --out analysis/
wikitrace analyze plot --stats analysis/ --out analysis/gain_regression.pdf
```

- `export` writes one row per test sample of every completed run under `--runs`.
- `stats` writes `gain_regression_suite.csv`, `gain_regression_benchmark.csv` and `gain_regression_average.csv`.
- `plot` writes the suite-average figure and a per-suite figure, each as PDF and PNG. `--methods` sets the order of the method rows.
- `wikitrace analyze all --runs outputs/ --out analysis/` runs the three steps in sequence.

## Evaluating another memory method

Run the method on the same suites and trials as the no-memory baseline and add its rows to the CSV under a `method` label of your choice. A method that runs on this harness can be exported from its run directories instead. Then run `stats` and `plot`.
