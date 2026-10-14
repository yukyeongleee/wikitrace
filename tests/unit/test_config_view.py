"""The recorded and hashed config is the benchmark's view: another benchmark's keys
cannot move a cohort, and every key in the view is one the benchmark reads."""

from __future__ import annotations

import pytest
import yaml

from wikitrace.config import config_hash, dump_resolved, load_config, resolved_dict
from wikitrace.envs import config_view_for
from wikitrace.envs.base import _BENCHMARKS, WIKI_ROLES

SUITE_CFG = {
    "alfworld": "configs/alfworld.yaml",
    "scienceworld": "configs/scienceworld.yaml",
    "plancraft": "configs/plancraft.yaml",
    "appworld": "configs/appworld.yaml",
    "tau3": "configs/tau3.yaml",
}


def test_every_registered_benchmark_has_a_suite_config_here():
    assert set(SUITE_CFG) == set(_BENCHMARKS)


@pytest.mark.parametrize("benchmark", sorted(_BENCHMARKS))
@pytest.mark.parametrize("config_path", ["configs/base.yaml", None])
def test_view_holds_only_the_benchmarks_own_keys(benchmark, config_path):
    cfg = load_config(config_path or SUITE_CFG[benchmark])
    view = config_view_for(_BENCHMARKS[benchmark]())
    full = resolved_dict(cfg)
    reduced = resolved_dict(cfg, view)

    roots = {k for k in reduced["dataset"] if k.endswith("_root")}
    assert roots == {f"{benchmark}_root"}
    assert set(WIKI_ROLES) <= set(view.roles)
    # The view is a subset of the full dump, and every declared key exists.
    assert set(view.dataset_keys) <= set(full["dataset"])
    kept = [k for k in view.dataset_keys if k not in view.omit_at_default]
    assert set(kept) <= set(reduced["dataset"]) <= set(view.dataset_keys)
    assert reduced["dataset"] == {k: full["dataset"][k] for k in reduced["dataset"]}
    # A key omitted at its default appears once it is set, so it still splits cohorts.
    changed = cfg.model_copy(update={"dataset": cfg.dataset.model_copy(update={k: "x" for k in view.omit_at_default})})
    assert set(resolved_dict(changed, view)["dataset"]) == set(view.dataset_keys)
    assert set(reduced["models"]["roles"]) <= set(full["models"]["roles"])
    for section in reduced:
        if section not in ("dataset", "models"):
            assert reduced[section] == full[section]


def test_another_benchmarks_key_does_not_move_the_cohort():
    cfg = load_config("configs/base.yaml")
    appworld = config_view_for(_BENCHMARKS["appworld"]())
    alfworld = config_view_for(_BENCHMARKS["alfworld"]())
    before = config_hash(cfg, appworld), config_hash(cfg, alfworld)

    cfg.dataset.scienceworld_root = "/elsewhere/scienceworld"
    cfg.models.roles["judge"] = {"model": "other/judge"}
    assert config_hash(cfg, appworld) == before[0]
    assert config_hash(cfg, alfworld) == before[1]

    cfg.dataset.appworld_root = "/elsewhere/appworld"
    assert config_hash(cfg, appworld) != before[0]
    assert config_hash(cfg, alfworld) == before[1]


@pytest.mark.parametrize("benchmark", sorted(_BENCHMARKS))
def test_reduced_file_loads_back_as_a_config(benchmark, tmp_path):
    """`config.resolved.yaml` must still validate for the tools that read it back."""
    cfg = load_config(SUITE_CFG[benchmark])
    view = config_view_for(_BENCHMARKS[benchmark]())
    dest = tmp_path / "config.resolved.yaml"
    dump_resolved(cfg, dest, view)
    written = yaml.safe_load(dest.read_text(encoding="utf-8"))
    assert written == resolved_dict(cfg, view)
    back = load_config(dest)
    assert resolved_dict(back, view) == written
    assert config_hash(back, view) == config_hash(cfg, view)
