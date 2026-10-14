"""The committed Plancraft data files: the frozen suite manifest and the solver's prompt data."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from wikitrace.envs.textgame.exemplars import load_exemplars
from wikitrace.envs.textgame.manifest import read_manifest


def _bins(entries: list[dict]) -> dict[str, int]:
    return dict(Counter(e["complexity_split"] for e in entries))


def test_manifest_counts_and_bins():
    m = read_manifest("plancraft")
    assert (len(m["train"]), len(m["test_seen"]), len(m["test_unseen"])) == (50, 100, 100)
    assert _bins(m["train"]) == {"easy": 7, "medium": 15, "hard": 28}
    assert _bins(m["test_seen"]) == {"easy": 9, "medium": 42, "hard": 49}
    assert _bins(m["test_unseen"]) == {"medium": 23, "hard": 77}


def test_manifest_targets_and_budget():
    m = read_manifest("plancraft")
    train_targets = {e["target"] for e in m["train"]}
    assert len(train_targets) == 50
    assert {e["target"] for e in m["test_seen"]} <= train_targets
    assert not train_targets & {e["target"] for e in m["test_unseen"]}
    ids = [e["id"] for k in ("train", "test_seen", "test_unseen") for e in m[k]]
    assert len(ids) == len(set(ids))
    assert all(e["oracle_steps"] <= 30 for k in ("train", "test_seen", "test_unseen") for e in m[k])


def test_manifest_unseen_targets_have_no_solvable_train_example():
    venv = Path(__file__).resolve().parents[2] / "data" / "plancraft" / ".venv"
    found = sorted(venv.glob("lib/python*/site-packages/plancraft/data/train.json"))
    if not found:
        pytest.skip(f"no plancraft package under {venv}: run scripts/setup/plancraft.sh")
    train = json.loads(found[0].read_text())
    solvable = {r["target"] for r in train if r["impossible"] in (False, "False")}
    assert not solvable & {e["target"] for e in read_manifest("plancraft")["test_unseen"]}


def test_manifest_head_slices_mirror_the_bin_mix():
    m = read_manifest("plancraft")
    assert _bins(m["test_seen"][:20]) == {"easy": 2, "medium": 8, "hard": 10}
    assert _bins(m["train"][:8]) == {"easy": 1, "medium": 2, "hard": 5}
    assert _bins(m["test_unseen"][:4]) == {"medium": 1, "hard": 3}


def test_prompt_data_holds_the_four_actions_and_no_think():
    d = load_exemplars("plancraft")
    assert set(d) == {"system", "fewshot"}
    for name in ("move", "smelt", "search", "impossible"):
        assert f"\t- {name}: " in d["system"]
    assert "think" not in d["system"] and "think" not in d["fewshot"]
    commands = [ln for ln in d["fewshot"].splitlines() if ln.startswith("> ")]
    assert commands == [
        "> search: andesite", "> move: from [I18] to [B1] with quantity 1",
        "> move: from [I30] to [B2] with quantity 1", "> move: from [0] to [I6] with quantity 1",
        "> search: iron_ingot", "> smelt: from [I36] to [I35] with quantity 1",
    ]
