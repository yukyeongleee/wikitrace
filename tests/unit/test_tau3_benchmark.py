"""The tau3 benchmark class, its manifest, the take_action parser and the JSON step string."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from wikitrace.config import DatasetConfig
from wikitrace.envs import resolve, train_group_for
from wikitrace.envs.tau3.benchmarks import tau3 as bench_mod
from wikitrace.harness.environment import EnvTool, step_input
from wikitrace.harness.environment.server import Episode

REPO = Path(__file__).resolve().parents[2]
# The recorded split is a local, untracked file of the main checkout (a worktree has none).
_MAIN = REPO.parents[2] if REPO.parent.name == "worktrees" else REPO
SPLIT_JSON = _MAIN / ".reference" / "benchmarks" / "tau3" / "tau3-split" / "split.json"
ENV_MODULE = REPO / "src" / "wikitrace" / "envs" / "tau3" / "envs" / "tau3.py"


def _env_module():
    """The environment module by file path, as the server loads it: no tau2 import at module level."""
    spec = importlib.util.spec_from_file_location("tau3_env_under_test", ENV_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registered_with_the_record_settings() -> None:
    d = resolve("tau3", "tau3")
    b = d.benchmark
    assert type(d).__module__ == "wikitrace.envs.tau3.adapter" and d.name == "tau3"
    assert (b.env_tool, b.env_extension, b.end_command, b.step_limit, b.max_turns) == (
        "take_action", "take_action_tau3", "", 0, 150)
    assert b.multiline_observations and b.task_kind == "retail customer service with tools"
    assert b.harness_spec("retail").kind == "environment" and not b.harness_spec("retail").input_on_disk
    assert b.harness_spec("retail").allowed_hosts == ()
    assert "`take_action` call is a retail tool call" in b.raw_metric_vocabulary.step_contents
    assert "assistant `message` row is a message to the user" in b.raw_metric_vocabulary.step_contents
    assert "respond" not in b.raw_metric_vocabulary.step_contents and b.message_channel
    assert "`act`" not in b.raw_metric_vocabulary.step_contents
    assert "assistant `message` row is a message to the user" in d.raw_layer_shape.metadata_artifacts
    assert train_group_for(b, "retail") == "retail"
    assert b.env_factory.endswith("envs/tau3/envs/tau3.py:make_env")
    assert EnvTool(b.env_tool, Path("/tmp"), b.env_extension).extension.name == "take_action_tau3.ts"
    assert b.exemplars_for(None) == "" and b.initial_observation("hi", {}) == "" and b.task_block("hi", {}) == "hi"


@pytest.mark.skipif(not (REPO / "data" / "tau3" / ".venv" / "bin" / "python").exists(),
                    reason="data/tau3 is not installed")
def test_suite_loads_from_the_manifest_with_the_task_text_left_to_reset() -> None:
    b = resolve("tau3", "tau3").benchmark
    samples = b.load_suite(DatasetConfig(), "retail")
    assert len(samples) == 114
    assert [s.split_role for s in samples] == ["train"] * 38 + ["test"] * 76
    assert {s.task_type for s in samples} == {"retail"} and {s.db_id for s in samples} == {"retail"}
    assert all(s.task_text == "" for s in samples)
    s = samples[0]
    assert s.id == f"tau3_0000_{s.variant}" and s.spec["task_id"] == s.variant == s.meta["task_id"]
    assert s.spec["user_model"] == s.spec["judge_model"] == bench_mod.SIMULATOR_MODEL
    assert s.spec["user_llm_args"] == {"reasoning_effort": "medium", "max_tokens": 30000, "timeout": 35, "num_retries": 1}
    assert set(s.spec) == {"task_id", "user_model", "user_llm_args", "judge_model", "judge_llm_args"}
    assert b.python.endswith("data/tau3/.venv/bin/python")
    with pytest.raises(ValueError):
        b.load_suite(DatasetConfig(), "airline")
    # The smoke slice is the head of each role.
    smoke = b.load_suite(DatasetConfig(limit_explicit_split=True, train_size=4, test_size=2), "retail")
    assert len(smoke) == 114  # `make_split` takes the head slice; the loader keeps the manifest order


def test_manifest_is_the_split_with_the_smoke_slice_first() -> None:
    m = bench_mod.read_manifest()
    assert m["source"]["commit"] == "fc0055dc4e0a316c3f83133267fbd6faaa770992"
    assert (m["pool"], m["users"], m["seed"]) == (114, 53, 0)
    train, test = m["train"], m["test"]
    assert (len(train), len(test)) == (38, 76)
    assert len({r["task_id"] for r in train} | {r["task_id"] for r in test}) == 114
    users_train = {r["user_id"] for r in train}
    users_test = {r["user_id"] for r in test}
    by_user: dict[str, int] = {}
    for r in train + test:
        by_user[r["user_id"]] = by_user.get(r["user_id"], 0) + 1
    multi = {u for u, n in by_user.items() if n > 1}
    assert multi and multi <= users_train and multi <= users_test
    head = [r["task_id"] for r in train[:4] + test[:2]]
    assert m["smoke"] == {"seed": 0, "train": 4, "test": 2, "nl_assertions": ["16", "21", "103", "104"]}
    # The judge runs in the smoke: the builder redraws until one of the six has NL assertions.
    assert head == ["16", "21", "53", "103", "51", "104"] and set(m["smoke"]["nl_assertions"]) <= set(head)
    smoke_users = {r["user_id"] for r in train[:4]}
    assert len(train[:4]) == 4 and {r["user_id"] for r in test[:2]} <= smoke_users
    assert [r["task_id"] for r in train[4:]] == sorted((r["task_id"] for r in train[4:]), key=int)
    assert [r["task_id"] for r in test[2:]] == sorted((r["task_id"] for r in test[2:]), key=int)


@pytest.mark.skipif(not (REPO / "data" / "tau3" / "repo" / "uv.lock").exists(), reason="data/tau3 is not installed")
def test_the_committed_manifest_is_what_the_builder_writes_from_the_pinned_clone() -> None:
    import subprocess
    import sys

    builder = REPO / "scripts" / "setup" / "build_tau3_manifest.py"
    done = subprocess.run([sys.executable, str(builder), "--check"], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.skipif(not SPLIT_JSON.exists(), reason=f"{SPLIT_JSON} is absent")
def test_manifest_lists_the_same_ids_and_users_as_the_recorded_split() -> None:
    m = bench_mod.read_manifest()
    s = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    for role in ("train", "test"):
        assert {r["task_id"]: r["user_id"] for r in m[role]} == {r["task_id"]: r["user_id"] for r in s[role]}
    assert m["source"] == s["source"]


def test_step_input_is_the_extensions_json_string_for_the_tau3_extension() -> None:
    args = {"action": "get_order_details", "argument": {"order_id": "#W1", "n": 2}}
    assert step_input("take_action", args, "take_action_tau3") == '{"action":"get_order_details","argument":{"order_id":"#W1","n":2}}'
    assert step_input("take_action", {"action": "respond", "argument": "héllo"}, "take_action_tau3") == '{"action":"respond","argument":"héllo"}'
    # The Plancraft spelling is untouched.
    assert step_input("take_action", {"action": "search", "argument": "iron"}) == "search: iron"
    assert EnvTool("take_action", Path("/tmp")).extension.name == "take_action.ts"


def test_parser_returns_the_action_and_argument_or_the_one_parse_error() -> None:
    m = _env_module()
    assert m.parse_command('{"action":"respond","argument":"hi"}') == ("respond", "hi")
    assert m.parse_command('{"action":"get_user_details","argument":{"user_id":"u"}}') == ("get_user_details", {"user_id": "u"})
    for bad in ("not json", "[1]", '{"action": 1, "argument": "x"}', '{"action": "respond"}'):
        assert m.parse_command(bad) == m.PARSE_ERROR
    assert m.PARSE_ERROR.startswith("Error: ") and "\n" not in m.PARSE_ERROR


def test_argument_check_rejects_our_type_cases_and_leaves_the_rest_to_the_authors() -> None:
    m = _env_module()
    params = {"properties": {"order_id": {"type": "string", "title": "Order Id"},
                             "item_ids": {"type": "array", "items": {"type": "string"}},
                             "note": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
              "required": ["order_id", "item_ids"]}
    # `respond` is an unknown tool like any other, whatever its argument: the authors answer it.
    for argument in ("text", {"message": "x"}, "", 5):
        assert m.check_argument("respond", argument, None) == ""
    assert not hasattr(m, "RESPOND") and not hasattr(m, "EMPTY_RESPOND")
    assert m.check_argument("f", "x", params) == "Error: `f` takes its parameters as a JSON object, got string."
    assert m.check_argument("f", {"order_id": 5}, params) == "Error: `f` parameter `order_id` must be string, got integer."
    assert m.check_argument("f", {"item_ids": ["a", 1]}, params) == "Error: `f` parameter `item_ids` must be array of string, got array."
    assert m.check_argument("f", {"note": 1.5}, params) == "Error: `f` parameter `note` must be string or null, got number."
    assert m.check_argument("f", {"order_id": True}, params).startswith("Error: `f` parameter `order_id` must be string, got boolean")
    # Accepted here, answered by the authors' code: a missing key, an unknown key, an unknown tool.
    assert m.check_argument("f", {}, params) == ""
    assert m.check_argument("f", {"order_id": "#W1", "extra": 1}, params) == ""
    assert m.check_argument("nope", {"x": 1}, None) == ""
    assert m.check_argument("f", {"order_id": "#W1", "item_ids": [], "note": None}, params) == ""


def test_the_simulator_and_judge_requests_time_out_inside_the_runners_idle_kill() -> None:
    from wikitrace.config import load_config
    from wikitrace.harness.terminal.runner import _CMD_BACKSTOP_SLACK

    tc = load_config(REPO / "configs" / "base.yaml").traverse
    kill = min(tc.idle_timeout, tc.cmd_timeout + _CMD_BACKSTOP_SLACK)
    for args in (bench_mod.SIMULATOR_LLM_ARGS, bench_mod.JUDGE_LLM_ARGS):
        # One httpx attempt, then `num_retries` more from litellm's wrapper with no wait between
        # them (test_tau3_env.py measures the count), so every attempt ends before the kill.
        attempts = 1 + args["num_retries"]
        assert attempts * args["timeout"] < kill
        assert args["timeout"] == 35 and args["num_retries"] == 1
    assert bench_mod.JUDGE_LLM_ARGS == bench_mod.SIMULATOR_LLM_ARGS
    spec = bench_mod.Tau3BenchmarkAdapter.spec_for("10")
    assert spec["user_llm_args"] == spec["judge_llm_args"] == bench_mod.SIMULATOR_LLM_ARGS


class _TwoSources:
    """An environment whose replies come from a user or a tool, named in ``info``."""

    def reset(self):
        return "hello", {"source": "user", "task": "hello"}

    def step(self, command):
        src = "user" if command.startswith("respond") else "env"
        return f"reply to {command}", False, {"source": src}

    def close(self):
        pass


def test_the_server_record_carries_the_observation_source(tmp_path) -> None:
    ep = Episode(_TwoSources, tmp_path)
    ep.reset()
    ep.step("respond hi")
    ep.step("get_user_details")
    ep.close("agent_stopped")
    env_lines = [r for r in ep.trajectory if r["role"] == "environment"]
    assert [r.get("source") for r in env_lines] == ["user", "user", "env"]
    assert [r["step"] for r in env_lines] == [0, 1, 2]
    # A record of the fake environment, whose info names no source, is unchanged.
    from wikitrace.harness.environment import fake

    (tmp_path / "b").mkdir()
    ep = Episode(lambda: fake.make_env({"task": "t"}), tmp_path / "b")
    ep.reset()
    ep.step("look")
    ep.close("agent_stopped")
    assert all("source" not in r for r in ep.trajectory)


def test_verify_install_checks_the_clone_commit_the_lock_sync_and_the_venv(tmp_path) -> None:
    import hashlib

    assert bench_mod.TAU2_COMMIT == bench_mod.read_manifest()["source"]["commit"]
    assert f'TAU2_COMMIT:-{bench_mod.TAU2_COMMIT}' in (REPO / "scripts/setup/tau3.sh").read_text()
    cfg = DatasetConfig(tau3_root=str(tmp_path))
    check = resolve("tau3", "tau3").benchmark.verify_install
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    head, lock, mark = tmp_path / "repo/.git/HEAD", tmp_path / "repo/uv.lock", tmp_path / ".venv/.uv-lock.sha256"
    head.write_text("0" * 40 + "\n")
    with pytest.raises(RuntimeError, match="not the pinned"):
        check(cfg)
    head.write_text(bench_mod.TAU2_COMMIT + "\n")
    lock.write_text("lock v1")
    mark.write_text(hashlib.sha256(b"lock v0").hexdigest() + "\n")
    with pytest.raises(RuntimeError, match="was not synced from"):
        check(cfg)
    mark.write_text(hashlib.sha256(b"lock v1").hexdigest() + "\n")
    with pytest.raises(FileNotFoundError, match="python is missing"):
        check(cfg)
    (tmp_path / ".venv" / "bin" / "python").write_text("")
    check(cfg)
