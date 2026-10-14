"""The `wikitrace` command line: `build` a corpus, `run` one method, `analyze` the runs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from wikitrace.analysis.cli import app as analysis_app
from wikitrace.envs import SUITES
from wikitrace.run.spec import METHODS

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="WikiTrace: an evidence-linked memory for terminal agents.")

app.add_typer(analysis_app, name="analyze")

_BENCHMARKS = " | ".join(SUITES)
_SUITES = "; ".join(f"{b}: {', '.join(s)}" for b, s in SUITES.items())


def _check(benchmark: str, suite: str) -> None:
    if benchmark not in SUITES:
        raise typer.BadParameter(f"unknown benchmark {benchmark!r} (expected {_BENCHMARKS})")
    if suite not in SUITES[benchmark]:
        raise typer.BadParameter(f"unknown suite {suite!r} for {benchmark} (expected {', '.join(SUITES[benchmark])})")


@app.command("build")
def build_cmd(
    config: Path = typer.Option(..., "--config", help="A benchmark config, e.g. configs/alfworld.yaml."),
    benchmark: str = typer.Option(..., "--benchmark", help=_BENCHMARKS),
    suite: str = typer.Option(..., "--suite", help=_SUITES),
    trial: int = typer.Option(..., "--trial", help="The trial whose corpus to build (rep-<trial>)."),
) -> None:
    """Run the no-memory train episodes, compile each into insights, write the recommendations."""
    from wikitrace.run.build import build

    _check(benchmark, suite)
    typer.echo(str(build(config, benchmark, suite, trial)))


@app.command("run")
def run_cmd(
    config: Path = typer.Option(..., "--config", help="A benchmark config, e.g. configs/alfworld.yaml."),
    benchmark: str = typer.Option(..., "--benchmark", help=_BENCHMARKS),
    suite: str = typer.Option(..., "--suite", help=_SUITES),
    trial: int = typer.Option(..., "--trial", help="The trial to run; it reads the corpus of the same trial."),
    method: str = typer.Option(..., "--method", help=" | ".join(METHODS)),
    corpus: Optional[Path] = typer.Option(
        None, "--corpus", help="A built corpus directory to read instead of this trial's; kinj and wikitrace only."),
) -> None:
    """Test one method on the suite's test split and summarise it."""
    from wikitrace.run.cell import run_cell

    _check(benchmark, suite)
    if method not in METHODS:
        raise typer.BadParameter(f"unknown method {method!r} (expected {' | '.join(METHODS)})")
    if corpus is not None and method == "no-memory":
        raise typer.BadParameter("--corpus applies to the kinj and wikitrace methods only")
    typer.echo(str(run_cell(config, benchmark, suite, trial, method, corpus=corpus)))




if __name__ == "__main__":
    app()
