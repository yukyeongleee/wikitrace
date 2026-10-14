"""Command line for exporting per-sample rows, computing Gain and Regression, and plotting them."""
from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="Gain and Regression against the no-memory baseline.", no_args_is_help=True)


@app.command()
def export(runs: list[Path] = typer.Option(..., help="Directories searched for run.json."),
           out: Path = typer.Option(Path("analysis/per_sample.csv"))) -> None:
    """Write per-sample rows of every completed run."""
    from .per_sample import export as run_export
    typer.echo(f"wrote {run_export(runs, out)} rows to {out}")


@app.command()
def stats(per_sample: Path = typer.Option(Path("analysis/per_sample.csv")),
          out: Path = typer.Option(Path("analysis"))) -> None:
    """Write Gain and Regression tables with bootstrap intervals."""
    from .gain_regression import compute, read_rows, write
    write(compute(read_rows(per_sample)), out)
    typer.echo(f"wrote {out}/gain_regression_{{suite,benchmark,average}}.csv")


@app.command()
def plot(stats: Path = typer.Option(Path("analysis")),
         out: Path = typer.Option(Path("analysis/gain_regression.pdf")),
         methods: list[str] = typer.Option(None, help="Method rows in order; default is CSV order.")) -> None:
    """Draw the suite-average and per-suite figures."""
    from .plot import plot as draw
    for path in draw(stats, out, methods or None):
        typer.echo(f"wrote {path} and its .png")


@app.command("all")
def run_all(runs: list[Path] = typer.Option(...), out: Path = typer.Option(Path("analysis"))) -> None:
    """Export, compute and plot in one step."""
    export(runs, out / "per_sample.csv")
    stats(out / "per_sample.csv", out)
    plot(out, out / "gain_regression.pdf", None)
