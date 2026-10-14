"""The CLI hands its flags to `build` and `run` unchanged and refuses what no method reads."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from typer.testing import CliRunner

import wikitrace.cli as cli
from wikitrace.run import build as build_mod
from wikitrace.run import cell as cell_mod


def test_build_forwards_its_flags(monkeypatch):
    build = Mock(return_value="outputs/corpus/alfworld/alfworld/rep-01")
    monkeypatch.setattr(build_mod, "build", build)
    result = CliRunner().invoke(cli.app, ["build", "--config", "c.yaml", "--benchmark", "alfworld",
                                          "--suite", "valid-seen", "--trial", "1"])
    assert result.exit_code == 0, result.output
    build.assert_called_once_with(Path("c.yaml"), "alfworld", "valid-seen", 1)


def test_run_forwards_its_flags(monkeypatch):
    run = Mock(return_value="outputs/tau3/retail/wikitrace/trial-02")
    monkeypatch.setattr(cell_mod, "run_cell", run)
    args = ["run", "--config", "c.yaml", "--benchmark", "tau3", "--suite", "retail", "--trial", "2",
            "--method", "wikitrace"]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    run.assert_called_once_with(Path("c.yaml"), "tau3", "retail", 2, "wikitrace", corpus=None)
    CliRunner().invoke(cli.app, [*args, "--corpus", "elsewhere"])
    assert run.call_args.kwargs == {"corpus": Path("elsewhere")}


def test_unknown_values_and_a_no_memory_corpus_are_refused(monkeypatch):
    run = Mock()
    monkeypatch.setattr(cell_mod, "run_cell", run)
    base = ["run", "--config", "c.yaml", "--trial", "0"]
    for extra in (["--benchmark", "bird", "--suite", "dev", "--method", "kinj"],
                  ["--benchmark", "alfworld", "--suite", "retail", "--method", "kinj"],
                  ["--benchmark", "alfworld", "--suite", "valid-seen", "--method", "rck"],
                  ["--benchmark", "alfworld", "--suite", "valid-seen", "--method", "no-memory", "--corpus", "x"]):
        assert CliRunner().invoke(cli.app, [*base, *extra]).exit_code != 0
    run.assert_not_called()


def test_every_subcommand_has_help():
    for args in ([], ["build"], ["run"]):
        assert CliRunner().invoke(cli.app, [*args, "--help"]).exit_code == 0
