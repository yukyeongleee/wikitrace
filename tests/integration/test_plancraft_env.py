"""The Plancraft environment under its own venv: oracle replays, a rejected string, a seeded search."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO / "data" / "plancraft" / ".venv" / "bin" / "python"
TEXTGAME = REPO / "src" / "wikitrace" / "envs" / "textgame"

pytestmark = pytest.mark.skipif(not PYTHON.exists(), reason=f"{PYTHON} is absent: run scripts/setup/plancraft.sh")

# Runs under the benchmark venv: replays the authors' OracleModel plan (`get_subplans`, as
# `OracleModel.step` flattens it) through `make_env` and prints one JSON line of findings.
_SCRIPT = r"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("pc", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from plancraft.environment.planner import get_subplans
out = {"episodes": []}
for split, example_id in json.loads(sys.argv[2]):
    env = m.make_env({"split": split, "example_id": example_id})
    obs, info = env.reset()
    plans, _ = get_subplans({"inventory": env.env.step()["inventory"], "target": env.example.target})
    steps, done = 0, False
    for action in (str(a) for sub in plans for a in sub):
        obs, done, info = env.step(action)
        steps += 1
        if done:
            break
    out["episodes"].append({"id": example_id, "done": done, "success": info["success"], "score": info["score"], "steps": steps,
                            "evaluation": env.evaluation()})
env = m.make_env({"split": "test", "example_id": json.loads(sys.argv[2])[0][1]})
first, _ = env.reset()
out["reset"] = first
out["invalid"] = env.step("craft it now")
out["search"] = [env.step("search: andesite")[0], env.step("search: andesite")[0]]
out["after"] = env.step("move: from [I1] to [I1] with quantity 1")
env.on_end_command("impossible: no recipe fits")
out["evaluation"] = env.evaluation()
print(json.dumps(out))
"""


def test_oracle_replays_succeed_and_replies_are_the_authors():
    manifest = json.loads((TEXTGAME / "data" / "plancraft.manifest.json").read_text())
    picks = [manifest["test_seen"][0], manifest["test_unseen"][1]]
    ids = [["test", e["id"]] for e in picks]
    proc = subprocess.run([str(PYTHON), "-c", _SCRIPT, str(TEXTGAME / "envs" / "plancraft.py"), json.dumps(ids)],
                          capture_output=True, text=True, timeout=300, check=True)
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    for episode, entry in zip(out["episodes"], picks):
        assert episode["done"] and episode["success"] and episode["score"] == 1.0
        assert episode["steps"] == entry["oracle_steps"]
    assert out["reset"].startswith(f"Craft an item of type: {picks[0]['target']}\ninventory:")
    reply, done, info = out["invalid"]
    assert reply == "Only select actions from the following: move, smelt, search, impossible"
    assert not done and info["invalid_command"] and not info["success"] and not info["failed"]
    first, second = out["search"]
    assert first == second and first.startswith("Recipes to craft andesite:")
    fewshot = json.loads((TEXTGAME / "data" / "plancraft_react_prompts.json").read_text())["fewshot"]
    assert f"> search: andesite\n{first}" in fewshot
    reply, done, info = out["after"]
    assert reply.startswith("Format Error: ") and info["invalid_command"] and not done
    for episode, entry in zip(out["episodes"], picks):
        ev = episode["evaluation"]
        assert (ev["target"], ev["complexity_split"], ev["give_up_reason"]) == (
            entry["target"], entry["complexity_split"], None)
        assert ev["searches"] == {"count": 0, "queries": []}
        assert any(i["item"] == entry["target"] and i["slot"] != "[0]" for i in ev["final_inventory"])
    # The stained glass is crafted through the output slot; the glazed terracotta is smelted straight into the inventory.
    assert [e["evaluation"]["target_reached_output_slot"] for e in out["episodes"]] == [True, False]
    ev = out["evaluation"]
    assert ev["searches"] == {"count": 2, "queries": ["andesite", "andesite"]}
    assert ev["give_up_reason"] == "no recipe fits" and not ev["target_reached_output_slot"]
    lines = out["reset"].splitlines()[2:]
    assert [f" - {i['item']} {i['slot']} quantity {i['quantity']}" for i in ev["final_inventory"]] == lines


# Runs under this repo's interpreter with the hash seed under test: one episode server started
# as the textgame workflow starts it, asked one search.
_SESSION = r"""
import json, sys
from pathlib import Path
from wikitrace.envs import resolve
from wikitrace.harness.environment import EnvironmentSession
b = resolve("textgame", "plancraft").benchmark
with EnvironmentSession(factory=b.env_factory, spec=json.loads(sys.argv[2]), out_dir=Path(sys.argv[3]),
                        step_limit=b.step_limit, end_command=b.end_command, python=sys.argv[1],
                        env=dict(b.server_env)) as env:
    env.reset()
    reply = env.step("search: gold_nugget")["observation"]
    env.step("impossible: stuck")
print(json.dumps([reply, json.loads((Path(sys.argv[3]) / "evaluation.json").read_text())]))
"""


def test_search_reply_ignores_the_parent_hash_seed(tmp_path):
    manifest = json.loads((TEXTGAME / "data" / "plancraft.manifest.json").read_text())
    spec = json.dumps({"split": "test", "example_id": manifest["test_seen"][0]["id"]})
    replies = []
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        proc = subprocess.run([sys.executable, "-c", _SESSION, str(PYTHON), spec, str(tmp_path / seed)],
                              capture_output=True, text=True, timeout=300, check=True, env=env)
        reply, evaluation = json.loads(proc.stdout.strip().splitlines()[-1])
        replies.append(reply)
    assert replies[0] == replies[1] and "smelt" in replies[0]
    assert (evaluation["ended_by"], evaluation["give_up_reason"]) == ("agent_stopped", "stuck")
    assert evaluation["searches"] == {"count": 1, "queries": ["gold_nugget"]}
