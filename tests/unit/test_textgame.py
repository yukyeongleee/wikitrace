"""The textgame domain: manifests, the episode loop over the fake environment, the raw step."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

import wikitrace.envs  # noqa: F401  (registers the built-in domains)
import wikitrace.envs.textgame.prompts  # noqa: F401
from wikitrace.config import append_result
from wikitrace.config import DatasetConfig
from wikitrace.config import make_split
from wikitrace.envs.base import _BENCHMARKS, resolve
from wikitrace.envs.textgame import adapter as tg_adapter
from wikitrace.envs.textgame import manifest as tg_manifest
from wikitrace.envs.textgame.benchmarks.alfworld import AlfworldBenchmarkAdapter
from wikitrace.envs.textgame.benchmarks.scienceworld import ScienceworldBenchmarkAdapter
from wikitrace.envs.textgame.exemplars import data_file_hashes, load_exemplars, strip_think
from wikitrace.envs.textgame.models import Action, TextGameResult, TextGameSample
from wikitrace.envs.textgame.workflow import run_textgame_episode
from wikitrace.harness.environment import fake
from wikitrace.harness.environment.protocol import request
from wikitrace.harness.terminal import FakeTraverseRunner, TerminalResult
from wikitrace.config import ToolStep


class _Dataset(DatasetConfig):
    """The dataset config with the two roots the textgame loaders read."""

    alfworld_root: str = "data/alfworld"
    scienceworld_root: str = "data/scienceworld"


def _dataset(**kw) -> DatasetConfig:
    return _Dataset(bird_root="x", databases=[], **kw)


def _sample(task_type: str = "pick_and_place_simple", **kw) -> TextGameSample:
    return TextGameSample(
        id="alfworld_0000_x", db_id="valid-seen", task_type=task_type, variant="v",
        task_text="Your task is to: put a kettle in shelf.", meta={"task_index": 0}, **kw,
    )


# --- registration and harness --------------------------------------------------


@pytest.mark.parametrize("benchmark,suite", [
    ("alfworld", "valid-seen"), ("alfworld", "valid-unseen"), ("scienceworld", "scienceworld"),
])
def test_every_suite_runs_on_the_environment_harness(benchmark, suite):
    dom = resolve("textgame", benchmark)
    spec = dom.benchmark.harness_spec(suite)
    assert spec.kind == "environment" and spec.environment and spec.terminal
    assert not spec.input_on_disk
    assert dom.benchmark.env_factory.endswith(f"envs/{benchmark}.py:make_env")


def test_both_alfworld_suites_share_one_train_group():
    assert AlfworldBenchmarkAdapter.train_group("valid-seen") == "alfworld"
    assert AlfworldBenchmarkAdapter.train_group("valid-unseen") == "alfworld"


def test_env_adapters_import_nothing_from_the_package():
    """The server loads them under the benchmark venv, which has none of wikitrace."""
    envs = Path(tg_adapter.__file__).parent / "envs"
    for path in envs.glob("*.py"):
        assert "wikitrace" not in path.read_text(encoding="utf-8"), path


# --- exemplars -----------------------------------------------------------------


def test_strip_think_drops_the_think_turns_and_their_ok_replies():
    text = "obs\n> think: plan\nOK.\n> go to desk 1\nOn the desk 1, you see a pen 1.\n> think: done\nOK. \n> use lamp 1\nYou turn on the lamp 1."
    assert strip_think(text) == "obs\n> go to desk 1\nOn the desk 1, you see a pen 1.\n> use lamp 1\nYou turn on the lamp 1."


@pytest.mark.parametrize("benchmark", ["alfworld", "scienceworld"])
def test_exemplar_files_carry_no_think_after_stripping(benchmark):
    for text in load_exemplars(benchmark).values():
        assert "think:" not in strip_think(text)
    hashes = data_file_hashes(benchmark)
    assert set(hashes) == {f"{benchmark}_react_prompts.json", f"{benchmark}.manifest.json"}
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes.values())


def test_alfworld_exemplars_follow_the_notebook_pairing():
    d = load_exemplars("alfworld")
    bench = AlfworldBenchmarkAdapter()
    text = bench.exemplars_for(_sample("pick_clean_then_place_in_recep"))
    assert text == strip_think(d["react_clean_1"] + d["react_clean_0"])
    assert "in/on" not in text and "move " in text  # the current-grammar rewrite


# --- manifests -----------------------------------------------------------------


def _fake_alfworld_root(tmp_path: Path) -> Path:
    root = tmp_path / "alfworld"
    for split, name in (("train", "pick_and_place_simple-Kettle-None-Shelf-1"),
                        ("train", "look_at_obj_in_light-Pen-None-DeskLamp-2"),
                        ("valid_seen", "pick_and_place_simple-Cup-None-Shelf-1")):
        game = root / "json_2.1.1" / split / name / "trial_0"
        game.mkdir(parents=True)
        (game / "game.tw-pddl").write_text("{}")
    return root


def _write_manifest(tmp_path: Path, monkeypatch, name: str, manifest: dict) -> None:
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    (data / f"{name}.manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(tg_manifest, "DATA_DIR", data)


def test_alfworld_manifest_loads_as_an_explicit_role_split(tmp_path, monkeypatch):
    root = _fake_alfworld_root(tmp_path)
    _write_manifest(tmp_path, monkeypatch, "alfworld", {
        "train": [
            {"game": "json_2.1.1/train/pick_and_place_simple-Kettle-None-Shelf-1/trial_0", "split": "train",
             "task_type": "pick_and_place_simple", "scene_id": 1, "task_text": "Your task is to: put a kettle in shelf."},
            {"game": "json_2.1.1/train/look_at_obj_in_light-Pen-None-DeskLamp-2/trial_0", "split": "train",
             "task_type": "look_at_obj_in_light", "scene_id": 2, "task_text": "Your task is to: look at pen under the desklamp."},
        ],
        "valid_seen": [
            {"game": "json_2.1.1/valid_seen/pick_and_place_simple-Cup-None-Shelf-1/trial_0", "split": "valid_seen",
             "task_type": "pick_and_place_simple", "scene_id": 1, "task_text": "Your task is to: put a cup in shelf."},
        ],
        "valid_unseen": [],
    })
    bench = AlfworldBenchmarkAdapter()
    samples = bench.load_suite(_dataset(alfworld_root=str(root)), "valid-seen")
    assert [s.split_role for s in samples] == ["train", "train", "test"]
    assert samples[0].spec["game_file"].endswith("Kettle-None-Shelf-1/trial_0/game.tw-pddl")
    assert samples[0].spec["alfworld_data"] == str(root)
    assert samples[0].stratum == "pick_and_place_simple" and samples[0].question == samples[0].task_text
    assert samples[2].meta == {"scene_id": 1, "split": "valid_seen"}
    split = make_split(samples, _dataset(alfworld_root=str(root)), "valid-seen")
    assert [s.id for s in split.train] == [samples[0].id, samples[1].id]
    assert [s.id for s in split.test] == [samples[2].id]
    # The smoke's head slice keeps the frozen order.
    head = make_split(samples, _dataset(alfworld_root=str(root), limit_explicit_split=True,
                                        train_size=1, test_size=1), "valid-seen")
    assert [s.id for s in head.train] == [samples[0].id]
    with pytest.raises(ValueError):
        bench.load_suite(_dataset(alfworld_root=str(root)), "alfworld-smoke")


def test_alfworld_manifest_naming_a_missing_game_fails_loudly(tmp_path, monkeypatch):
    root = _fake_alfworld_root(tmp_path)
    _write_manifest(tmp_path, monkeypatch, "alfworld", {
        "train": [{"game": "json_2.1.1/train/gone/trial_0", "split": "train", "task_type": "x",
                   "scene_id": 1, "task_text": "t"}],
        "valid_seen": [], "valid_unseen": [],
    })
    with pytest.raises(FileNotFoundError, match="diverged"):
        AlfworldBenchmarkAdapter().load_suite(_dataset(alfworld_root=str(root)), "valid-seen")


def test_scienceworld_manifest_loads_with_the_reference_loop_settings(tmp_path, monkeypatch):
    root = tmp_path / "scienceworld"
    (root / "jdk" / "bin").mkdir(parents=True)
    (root / "jdk" / "bin" / "java").write_text("")
    _write_manifest(tmp_path, monkeypatch, "scienceworld", {
        "train": [{"task": "boil", "task_index": 0, "variation": 3, "split": "train",
                   "description": "Your task is to boil water.", "gold_length": 39}],
        "test": [{"task": "boil", "task_index": 0, "variation": 22, "split": "test",
                  "description": "Your task is to boil milk.", "gold_length": 40}],
    })
    bench = ScienceworldBenchmarkAdapter()
    samples = bench.load_suite(_dataset(scienceworld_root=str(root)), "scienceworld")
    assert [s.split_role for s in samples] == ["train", "test"]
    assert samples[0].spec == {"task": "boil", "variation": 3, "simplification": "easy",
                               "env_step_limit": 100, "java_home": str(root / "jdk")}
    assert samples[0].meta["task_index"] == 0 and samples[0].variant == "3"
    # The budget is the simulator's clock; the server enforces no action limit.
    assert bench.step_limit == 0 and bench.harness_spec("scienceworld").max_turns == 400


def test_committed_manifests_match_the_brief():
    """The frozen files: sizes, type balance, distinct train descriptions and disjointness."""
    alf = tg_manifest.DATA_DIR / "alfworld.manifest.json"
    assert alf.exists(), "the frozen manifest is a committed artifact"
    m = json.loads(alf.read_text())
    assert len(m["train"]) == 50 and len(m["valid_seen"]) == 101 and len(m["valid_unseen"]) == 134
    counts = {}
    for e in m["train"]:
        counts[e["task_type"]] = counts.get(e["task_type"], 0) + 1
    assert sorted(counts.values()) == [8, 8, 8, 8, 9, 9]
    assert len({e["task_type"] for e in m["train"][:6]}) == 6  # interleaved
    assert len({e["task_text"] for e in m["train"]}) == 50
    sw = tg_manifest.DATA_DIR / "scienceworld.manifest.json"
    assert sw.exists(), "the frozen manifest is a committed artifact"
    m = json.loads(sw.read_text())
    assert len(m["train"]) == 50 and len(m["test"]) == 100
    train_desc = {(e["task"], e["description"]) for e in m["train"]}
    assert not train_desc & {(e["task"], e["description"]) for e in m["test"]}
    assert len({e["task"] for e in m["train"][:10]}) == 10


# --- scoring and the index vocabulary ---------------------------------------------


def _result(**kw) -> TextGameResult:
    base = dict(sample_id="s", db_id="d", question="t", task_type="boil", variant="3")
    return TextGameResult(**{**base, **kw})


def test_alfworld_metrics_are_binary_with_no_secondary():
    bench = AlfworldBenchmarkAdapter()
    m = bench.metrics_from_result(_result(success=True, score=1.0, task_type="look_at_obj_in_light"))
    assert m.primary == 1.0 and m.secondary is None and m.success and not m.refused
    summary = bench.summarize([m, bench.metrics_from_result(_result(success=False))])
    assert summary["success_rate"] == 0.5 and summary["secondary_mean"] is None
    assert summary["success_rate_by_task_type"] == {"boil": 0.0, "look_at_obj_in_light": 1.0}


def test_scienceworld_score_is_the_secondary_metric_and_failed_is_reported():
    bench = ScienceworldBenchmarkAdapter()
    m = bench.metrics_from_result(_result(success=False, score=0.4, failed=True))
    assert m.primary == 0.0 and m.secondary == 0.4 and m.extra["failed"] is True
    summary = bench.summarize([m, bench.metrics_from_result(_result(success=True, score=1.0))])
    assert summary["mean_score"] == pytest.approx(0.7)
    assert summary["failed_rate"] == 0.5 and summary["success_rate"] == 0.5


@pytest.mark.parametrize("benchmark", ["alfworld", "scienceworld"])
def test_index_entries_carry_the_outcome_and_no_score_column(benchmark):
    from wikitrace.config import Compilation, Outcome
    from wikitrace.memory.store import RawStepFacts, render_compilation_index, render_raw_index_from_steps

    shape = resolve("textgame", benchmark).raw_layer_shape
    assert shape.compilation_metric_keys == () and shape.index_score == ""
    comp = Compilation(step_id="step-0000", outcome=Outcome.FAILURE, metrics={"success": 0.0},
                       title="[failure] Kettle never found", question="Your task is to: x", body="")
    idx = render_compilation_index([comp], entry_source="title", raw_shape=shape)
    line = next(ln for ln in idx.splitlines() if ln.startswith("1. "))
    assert line == "1. [step-0000](/work/corpus/compilation/step-0000.md): failure | Kettle never found"
    assert "Each entry: N. [step](path): <outcome> | <one-line summary>" in idx
    assert "roll-up of the score" not in idx and "()" not in idx and "success: 0 | 1" not in idx
    raw = render_raw_index_from_steps(
        [("step-0000", RawStepFacts(outcome="failure", tail="Your task is to: x"))], raw_shape=shape)
    assert "1. [step-0000](/work/corpus/raw/step-0000/): failure | Your task is to: x" in raw


# --- the episode loop over the fake environment ----------------------------------


class _FakeBench(AlfworldBenchmarkAdapter):
    """ALFWorld's prompts and vocabulary over the harness's four-command fake game."""

    env_factory = f"{fake.__file__}:make_env"
    python = sys.executable


def _play(commands: list[str]):
    """A fake-runner handler that drives the environment through the tool's socket."""
    def handler(*, system_prompt, user_msg, env_tool, **_):
        assert env_tool is not None and env_tool.name == "act"
        sock = str(env_tool.socket_dir / "act.sock")
        turns, ctx = [], [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}]
        for i, cmd in enumerate(commands):
            reply = request(sock, {"op": "step", "input": cmd})
            turns.append({"type": "function_call", "name": "act", "arguments": {"command": cmd}, "call_id": f"c{i}"})
            ctx.append({"role": "assistant", "content": [{"type": "thinking", "thinking": f"try {cmd}"},
                                                         {"type": "toolCall", "id": f"c{i}"}]})
            ctx.append({"role": "toolResult", "toolCallId": f"c{i}",
                        "content": [{"type": "text", "text": reply["observation"]}]})
        turns.append({"type": "function_call", "name": "read", "arguments": {"path": "corpus/knowledge/document-0001.md"}, "call_id": "r"})
        ctx.append({"role": "assistant", "content": [{"type": "toolCall", "id": "r"}]})
        ctx.append({"role": "toolResult", "toolCallId": "r", "content": [{"type": "text", "text": "# doc"}]})
        return TerminalResult(final_text="done", turns=turns, final_context=ctx,
                              read_set=["corpus/knowledge/document-0001.md"],
                              usage={"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.001})
    return handler


def test_episode_outcome_is_the_environments_final_state(tmp_path):
    bench = _FakeBench()
    sample = _sample()
    sample.spec = {"task": "put the key in the box"}
    runner = FakeTraverseRunner(_play(["dance", "examine table", "take key", "put key in box"]))
    outcome = run_textgame_episode(
        sample, runner, None, tmp_path / "sessions", bench=bench, memory_layers=("R", "C", "K"),
    )
    r = outcome.result
    assert r.success and r.score == 1.0 and r.steps == 4 and r.invalid_commands == 1
    assert r.ended_by == "done" and not r.step_limit_hit and r.step_limit == 49
    assert [a.command for a in r.actions] == ["dance", "examine table", "take key", "put key in box"]
    assert r.actions[0].invalid and r.actions[0].observation == "Nothing happens."
    assert [s.command for s in r.tool_steps][:2] == ["> dance", "> examine table"]
    assert r.tool_steps[-1].command == "read corpus/knowledge/document-0001.md"
    assert r.citations[0].read["K"] == ["document-0001"]
    assert outcome.phase_logs[0].phase == "solver" and outcome.phase_logs[0].usage["cost"] == 0.001
    # The session dir holds the server's log and its one evaluation file, nothing else of it.
    session = tmp_path / "sessions" / sample.id / "solver"
    assert sorted(p.name for p in session.iterdir()) == ["env_server.log", "evaluation.json"]
    assert json.loads((session / "evaluation.json").read_text()) == {
        "success": True, "ended_by": "done", "steps": 4, "invalid_commands": 1,
    }


class _InventoryBench(_FakeBench):
    """The fake game with a Plancraft-shaped inventory as its opening observation."""

    @staticmethod
    def initial_observation(observation, info):
        return "inventory:\n - quartz [I1] quantity 43\n - lodestone [I2] quantity 47"


@pytest.mark.parametrize("multiline", [True, False])
def test_multiline_initial_observation_nests_one_tab_under_its_field(tmp_path, multiline):
    bench = type("B", (_InventoryBench,), {"multiline_observations": multiline})()
    dom = resolve("textgame", "alfworld")
    dom.benchmark = bench
    sample = _sample()
    sample.spec = {"task": "put the key in the box"}
    outcome = run_textgame_episode(sample, FakeTraverseRunner(_play(["take key"])), None,
                                   tmp_path / "sessions", bench=bench)
    dom.persist_traverse_trajectory(tmp_path / "step", sample, outcome)
    meta = (tmp_path / "step" / "metadata.md").read_text()
    fields = dict(dom.compilation_inputs(outcome.result).task_fields)
    if multiline:
        assert "- initial_observation: inventory:\n\t- quartz [I1] quantity 43\n" \
               "\t- lodestone [I2] quantity 47\n- success: " in meta
        assert fields["initial_observation"] == (
            "inventory:\n\t- quartz [I1] quantity 43\n\t- lodestone [I2] quantity 47")
    else:
        flat = "inventory: - quartz [I1] quantity 43 - lodestone [I2] quantity 47"
        assert f"- initial_observation: {flat}\n- success: " in meta
        assert fields["initial_observation"] == flat


def test_persisted_raw_step_holds_the_outcome_and_the_turn_log_only(tmp_path):
    bench = _FakeBench()
    dom = resolve("textgame", "alfworld")
    dom.benchmark = bench
    sample = _sample()
    sample.spec = {"task": "put the key in the box"}
    runner = FakeTraverseRunner(_play(["examine table", "take key"]))
    outcome = run_textgame_episode(sample, runner, None, tmp_path / "sessions", bench=bench)
    corpus = tmp_path / "corpus"
    step = corpus / "raw" / "step-0000"
    dom.persist_traverse_trajectory(step, sample, outcome)
    append_result(corpus, "step-0000", outcome.result)

    assert sorted(p.name for p in step.iterdir()) == ["metadata.md", "solver.full.jsonl"]
    meta = (step / "metadata.md").read_text()
    assert meta == (
        "# alfworld_0000_x\n\n"
        "- task_type: pick_and_place_simple\n"
        "- instruction: Your task is to: put a kettle in shelf.\n"
        "- initial_observation: You are in a small room. You see a table and a box.\n"
        "- success: False\n"
        "- steps: 2 act calls (ended by the agent)\n"
    )
    turns = [json.loads(ln) for ln in (step / "solver.full.jsonl").read_text().splitlines()]
    assert turns[0]["role"] == "system" and turns[-1]["type"] == "answer" and turns[-1]["parsed"] is None
    assert any(t.get("type") == "function_call" and t.get("result") == "On the table you see a key." for t in turns)

    facts = dom.read_raw_step(step)
    assert facts.outcome == "failure" and facts.metrics == {"success": 0.0}
    inputs = dom.compilation_inputs(dom.read_result(step), sample=sample)
    assert inputs.metrics == {"success": 0.0, "steps": 2.0, "step_limit_hit": 0.0}
    assert inputs.prediction == "success=0; steps=2 act calls (ended by the agent)"
    row = dom.sample_row(outcome.result, "raw/step-0000/metadata.md")
    assert row["steps"] == 2 and row["task_index"] == 0 and row["score"] == 0.5  # the run keeps it


def test_initial_observation_is_the_reset_text_on_one_line_without_the_task():
    # What the ALFWorld env adapter emits at reset: the room paragraph, then the task line.
    alf = AlfworldBenchmarkAdapter.initial_observation(
        "You are in the middle of a room. Looking quickly around you, you see a bed 1,\n\ta desk 1."
        "\n\nYour task is to: put a kettle in shelf.",
        {"task": "Your task is to: put a kettle in shelf."},
    )
    assert " ".join(alf.split()) == (
        "You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1."
    )
    sci = ScienceworldBenchmarkAdapter.initial_observation(
        "This room is called the hallway.\n In it, you see: \n\tthe agent\n\ta picture\n",
        {"task": "Your task is to boil water."},
    )
    assert " ".join(sci.split()) == "This room is called the hallway. In it, you see: the agent a picture"
    r = _result(initial_observation=" ".join(sci.split()))
    assert (
        "- instruction: t\n- initial_observation: This room is called the hallway. In it, you see: "
        "the agent a picture\n- success: False\n"
    ) in tg_adapter._metadata_text(r, "0")


def test_metadata_names_what_ended_an_episode_the_task_did_not():
    r = _result(success=False, score=0.4, failed=True)
    r.steps, r.step_limit_hit, r.ended_by = 51, True, "game_time_exhausted"
    dom = resolve("textgame", "scienceworld")
    steps = dom.benchmark.steps_text(r)
    assert steps == "51 act calls (ended by the environment: its step clock ran out)"
    meta = tg_adapter._metadata_text(r, steps)
    assert meta.endswith(f"- success: False\n- steps: {steps}\n")
    assert "score" not in meta and "failed" not in meta
    assert dom.render_outcome_for_compiler(r) == (
        f"## Outcome\noutcome: failure\nsteps: {steps}"
    )
    assert dom.compilation_inputs(r).prediction == f"success=0; steps={steps}"
    # The unit is the `act` call and no budget number appears, whichever benchmark; the
    # phrase is keyed on `ended_by`, the server's own limit against the simulator's clock.
    alf = resolve("textgame", "alfworld").benchmark
    r.steps, r.ended_by = 49, "budget_exhausted"
    assert alf.steps_text(r) == "49 act calls (ended by the harness: act budget ran out)"
    r.steps, r.step_limit_hit, r.ended_by = 17, False, "harness_stopped"
    assert alf.steps_text(r) == "17 act calls (ended by the harness: session cut off)"
    r.ended_by = "simulator_error"
    assert alf.steps_text(r) == "17 act calls (ended by a simulator crash, not the agent)"
    r.ended_by = "agent_stopped"
    assert alf.steps_text(r) == "17 act calls (ended by the agent)"
    r.ended_by = "task_failed"
    assert alf.steps_text(r) == "17 act calls (ended by the environment: an action failed the task)"
    r.steps, r.ended_by = 12, "done"
    assert alf.steps_text(r) == "12 act calls"


def _play_as_pi(commands: list[str], *, stopped_by: str = ""):
    """Like `_play`, but the context mirrors Pi: the whole step reply rides as `details`."""
    def handler(*, system_prompt, user_msg, env_tool, **_):
        sock = str(env_tool.socket_dir / "act.sock")
        turns, ctx = [], [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}]
        for i, cmd in enumerate(commands):
            if cmd.startswith("!"):  # a second act in one turn: refused by the hook, never sent
                reply = {"error": "one act per turn: read this observation before sending the next command"}
            else:
                reply = request(sock, {"op": "step", "input": cmd})
            turns.append({"type": "function_call", "name": "act", "arguments": {"command": cmd.lstrip("!")},
                          "call_id": f"c{i}"})
            ctx.append({"role": "assistant", "content": [{"type": "toolCall", "id": f"c{i}"}]})
            ctx.append({"role": "toolResult", "toolCallId": f"c{i}", "isError": "error" in reply,
                        "content": [{"type": "text", "text": reply.get("observation") or reply.get("error", "")}],
                        "details": reply})
        return TerminalResult(final_text="", turns=turns, final_context=ctx, stopped_by=stopped_by)
    return handler


def test_persisted_solver_log_carries_no_score_or_limit(tmp_path):
    """What the tool hands Pi is the observation alone, so the raw step cannot leak supervision."""
    bench = _FakeBench()
    dom = resolve("textgame", "alfworld")
    sample = _sample()
    sample.spec = {"task": "put the key in the box"}
    runner = FakeTraverseRunner(_play_as_pi(["examine table", "take key", "put key in box", "look"]))
    outcome = run_textgame_episode(sample, runner, None, tmp_path / "sessions", bench=bench)
    assert outcome.result.success and outcome.result.ended_by == "done"
    step = tmp_path / "corpus" / "raw" / "step-0000"
    dom.persist_traverse_trajectory(step, sample, outcome)
    log = (step / "solver.full.jsonl").read_text()
    for word in ("score", "step_limit", "tests_passed", "env_steps", "success"):
        assert f'"{word}"' not in log, word


def test_a_simulator_that_raises_is_a_generation_failure(tmp_path):
    (tmp_path / "faulty.py").write_text(
        "class Env:\n"
        "    def reset(self):\n        return 'start', {'task': 't'}\n"
        "    def step(self, command):\n        raise RuntimeError('JVM gone')\n"
        "    def close(self):\n        pass\n"
        "def make_env(spec):\n    return Env()\n"
    )
    bench = _FakeBench()
    bench.env_factory = f"{tmp_path / 'faulty.py'}:make_env"
    sample = _sample()
    runner = FakeTraverseRunner(_play_as_pi(["look", "look"]))
    outcome = run_textgame_episode(sample, runner, None, tmp_path / "sessions", bench=bench)
    r = outcome.result
    assert outcome.generation_failed and not r.success
    assert r.ended_by == "simulator_error" and r.generation_error.startswith("RuntimeError: JVM gone")
    assert r.steps == 1 and r.tool_steps[0].is_error and r.tool_steps[1].result == "Episode over"


def test_a_session_the_runner_cut_short_is_harness_stopped(tmp_path):
    bench = _FakeBench()
    sample = _sample()
    sample.spec = {"task": "put the key in the box"}
    runner = FakeTraverseRunner(_play_as_pi(["look"], stopped_by="turn_cap"))
    outcome = run_textgame_episode(sample, runner, None, tmp_path / "sessions", bench=bench)
    assert outcome.result.ended_by == "harness_stopped" and not outcome.generation_failed


def test_a_session_that_ended_on_a_provider_error_is_a_generation_failure(tmp_path):
    bench = _FakeBench()
    sample = _sample()
    sample.spec = {"task": "put the key in the box"}
    play = _play_as_pi(["look"])

    def handler(**kw):
        res = play(**kw)
        res.provider_error = "429 Provider returned error"
        return res

    outcome = run_textgame_episode(sample, FakeTraverseRunner(handler), None, tmp_path / "sessions", bench=bench)
    r = outcome.result
    assert outcome.generation_failed and r.generation_error == "provider_error: 429 Provider returned error"
    assert r.ended_by == "harness_stopped" and outcome.phase_logs[0].provider_error
    assert "A harness outcome, not the agent's." in tg_adapter.build_outcome_text(r, "1")


def test_an_act_the_hook_refused_is_not_a_step_in_any_render(tmp_path):
    bench = _FakeBench()
    dom = resolve("textgame", "alfworld")
    sample = _sample()
    sample.spec = {"task": "put the key in the box"}
    runner = FakeTraverseRunner(_play_as_pi(["examine table", "!take key", "take key"]))
    outcome = run_textgame_episode(sample, runner, None, tmp_path / "sessions", bench=bench)
    r = outcome.result
    assert r.steps == 2 and len(r.tool_steps) == 3 and r.tool_steps[1].is_error
    text = dom.render_trajectory_for_compiler(r)
    assert "one act per turn" not in text and "[3]" not in text
    assert "[1]\n  act    │ examine table\n  obs    │ On the table you see a key.\n[2]\n  act    │ take key\n" in text
    step = tmp_path / "corpus" / "raw" / "step-0000"
    dom.persist_traverse_trajectory(step, sample, outcome)


def test_a_command_after_the_episode_ended_keeps_the_record_aligned(tmp_path):
    """The `Episode over` reply is a tool result the server never counted or recorded."""
    bench = _FakeBench()
    dom = resolve("textgame", "alfworld")
    dom.benchmark = bench
    sample = _sample()
    sample.spec = {"task": "put the key in the box"}
    runner = FakeTraverseRunner(_play(["examine table", "take key", "put key in box", "look"]))
    outcome = run_textgame_episode(sample, runner, None, tmp_path / "sessions", bench=bench)
    r = outcome.result
    assert r.success and r.steps == 3 and len(r.actions) == 3
    assert r.tool_steps[3].result == "Episode over"
    text = dom.render_trajectory_for_compiler(r)
    assert "act    │ put key in box\n  obs    │ You put the key in the box. Task complete." in text
    assert "act    │ look\n  obs    │ Episode over" in text


def test_compiler_renders_use_the_servers_observations(tmp_path):
    bench = _FakeBench()
    dom = resolve("textgame", "alfworld")
    sample = _sample()
    sample.spec = {"task": "put the key in the box"}
    runner = FakeTraverseRunner(_play(["dance", "examine table"]))
    outcome = run_textgame_episode(sample, runner, None, tmp_path / "sessions", bench=bench)
    text = dom.render_trajectory_for_compiler(outcome.result)
    assert "act    │ dance" in text and "obs    │ Nothing happens." in text
    assert "reason │ try dance" in text
    assert "call   │ read corpus/knowledge/document-0001.md" in text
    outcome_text = dom.render_outcome_for_compiler(outcome.result)
    # The fake runner reports no guard, so the agent ended the session: the bullet says so.
    assert outcome_text == (
        "## Outcome\noutcome: failure\n"
        "steps: 2 act calls (ended by the agent)"
    )


# --- the evaluation block each simulator adapter reports ---------------------------

# `get_goal_progress()` of scienceworld 1.3.0 on task boil, variation 22, easy: captured.
_PROGRESS = (
    "Completed keys: \n"
    "----------------------------------------------------------------------------------------------------\n"
    "Sequential Subgoals:\n"
    "----------------------------------------------------------------------------------------------------\n"
    "0\t{s0}\tGoalFind\tfocus on substance\n"
    "1\tfalse\tGoalChangeStateOfMatter\tsubstance is in a liquid state\n"
    "2\tfalse\tGoalChangeStateOfMatter\tsubstance is in a gaseous state (or combusting)\n"
    "----------------------------------------------------------------------------------------------------\n"
    "Unordered and Optional Subgoals:\n"
    "----------------------------------------------------------------------------------------------------\n"
    "0\t{o0}\tGoalInRoomWithObject\tbe in same location as tin\n"
    "1\tfalse\tGoalObjectsInSingleContainer\thave substance alone in a single container\n"
    "2\t{o2}\tGoalActivateDeviceWithName\tactivate heater (stove)\n"
    "----------------------------------------------------------------------------------------------------\n"
)


def test_goal_progress_parses_both_sections_in_the_simulators_order():
    from wikitrace.envs.textgame.envs.scienceworld import parse_goal_progress

    parsed = parse_goal_progress(_PROGRESS.format(s0="true", o0="false", o2="true"))
    assert parsed == [
        ("sequential", "GoalFind", "focus on substance", True),
        ("sequential", "GoalChangeStateOfMatter", "substance is in a liquid state", False),
        ("sequential", "GoalChangeStateOfMatter", "substance is in a gaseous state (or combusting)", False),
        ("optional", "GoalInRoomWithObject", "be in same location as tin", False),
        ("optional", "GoalObjectsInSingleContainer", "have substance alone in a single container", False),
        ("optional", "GoalActivateDeviceWithName", "activate heater (stove)", True),
    ]
    assert parse_goal_progress("Completed keys: \n") == []
    # Some goal classes carry no description (change-the-state-of-matter-of, variation 22).
    empty = "Sequential Subgoals:\n0\ttrue\tGoalIsStateOfMatter\t\n"
    assert parse_goal_progress(empty) == [("sequential", "GoalIsStateOfMatter", "", True)]


class _Progress:
    """A simulator stub answering `get_goal_progress` from a queue of captured texts."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)

    def get_goal_progress(self) -> str:
        return self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]


def test_scienceworld_evaluation_records_when_each_subgoal_first_turned_true():
    from wikitrace.envs.textgame.envs.scienceworld import ScienceworldEnvironment

    env = ScienceworldEnvironment.__new__(ScienceworldEnvironment)
    env.env = _Progress([
        _PROGRESS.format(s0="false", o0="false", o2="false"),  # reset
        _PROGRESS.format(s0="false", o0="true", o2="false"),   # action 1
        _PROGRESS.format(s0="true", o0="true", o2="false"),    # action 2
        _PROGRESS.format(s0="true", o0="false", o2="false"),   # action 3: a subgoal can revert
    ])
    env.score, env.actions, env.subgoals, env.achieved_at = 0.0, 0, [], []
    env._track_subgoals()
    for _ in range(3):
        env.actions += 1
        env._track_subgoals()
    env.score = 0.4
    assert env.evaluation() == {
        "score": 0.4,
        "subgoals": {
            "sequential": {"achieved": 1, "total": 3},
            "optional": {"achieved": 1, "total": 3},
            "timeline": [
                {"kind": "sequential", "class": "GoalFind", "name": "focus on substance",
                 "achieved_at_step": 2},
                {"kind": "sequential", "class": "GoalChangeStateOfMatter",
                 "name": "substance is in a liquid state", "achieved_at_step": None},
                {"kind": "sequential", "class": "GoalChangeStateOfMatter",
                 "name": "substance is in a gaseous state (or combusting)", "achieved_at_step": None},
                {"kind": "optional", "class": "GoalInRoomWithObject",
                 "name": "be in same location as tin", "achieved_at_step": 1},
                {"kind": "optional", "class": "GoalObjectsInSingleContainer",
                 "name": "have substance alone in a single container", "achieved_at_step": None},
                {"kind": "optional", "class": "GoalActivateDeviceWithName",
                 "name": "activate heater (stove)", "achieved_at_step": None},
            ],
        },
    }


def test_alfworld_evaluation_block_is_empty():
    from wikitrace.envs.textgame.envs.alfworld import AlfworldEnvironment

    env = AlfworldEnvironment.__new__(AlfworldEnvironment)
    assert env.evaluation() == {}


def test_failed_outcome_carries_the_harness_error_and_no_success():
    dom = resolve("textgame", "scienceworld")
    outcome = dom.failed_traverse_outcome(_sample(), "boom")
    assert outcome.generation_failed and not outcome.result.success
    assert outcome.result.step_limit == 0 and "boom" in outcome.result.generation_error
    text = dom.render_outcome_for_compiler(outcome.result)
    assert "A harness outcome, not the agent's." in text and "outcome: failure" in text and "score" not in text


def test_compiler_render_holds_every_step_and_every_observation_whole():
    """An 80-step episode reaches the compiler entire: no step dropped, no field cut."""
    from wikitrace.config import ToolStep
    from wikitrace.envs.textgame.models import Action

    n = 80
    obs = ["o" * 900 + f" #{i}" for i in range(n)]
    result = _result(
        tool_steps=[
            ToolStep(command=f"> take item {i}", result="", reasoning="r" * 500)
            for i in range(n)
        ],
        actions=[Action(step=i, command=f"take item {i}", observation=obs[i]) for i in range(n)],
    )
    text = tg_adapter.build_trajectory_text(result)
    assert "omitted" not in text and "chars]" not in text
    for i in range(n):
        assert f"[{i + 1}]" in text
        assert obs[i] in text
    assert "r" * 500 in text


# --- prompts -------------------------------------------------------------------


def test_registered_benchmarks_include_both_textgame_benchmarks():
    assert {"alfworld", "scienceworld"} <= set(_BENCHMARKS)


# More turns than any per-case cap the renders ever carried.
_LONG_EPISODE = 60


def _long_frozen_step(corpus: Path) -> None:
    """One frozen textgame step whose episode is longer than any former render cap."""
    step = corpus / "raw" / "step-0000"
    step.mkdir(parents=True)
    actions = [Action(step=i, command=f"take item {i}", observation=f"item {i} is in your hand")
               for i in range(1, _LONG_EPISODE + 1)]
    tool_steps = [ToolStep(reasoning=f"thinking about item {i}", command=f"> take item {i}",
                           result=f"item {i} is in your hand")
                  for i in range(1, _LONG_EPISODE + 1)]
    turns = [{"type": "message", "role": "system", "content": "system"},
             {"type": "message", "role": "user", "content": "Here is the task. put the key in the box"}]
    turns += [{"type": "function_call", "name": "act", "arguments": {"command": f"take item {i}"},
               "reasoning": f"thinking about item {i}", "result": f"item {i} is in your hand"}
              for i in range(1, _LONG_EPISODE + 1)]
    turns.append({"type": "answer", "role": "assistant", "parsed": None})
    (step / "solver.full.jsonl").write_text(
        "".join(json.dumps(t) + "\n" for t in turns), encoding="utf-8")
    (step / "metadata.md").write_text(
        "# alfworld_0000_x\n\n- instruction: put the key in the box\n", encoding="utf-8")
    append_result(corpus, "step-0000", TextGameResult(
        sample_id="alfworld_0000_x", db_id="alfworld", question="put the key in the box",
        task_type="pick_and_place_simple", success=True, steps=_LONG_EPISODE,
        actions=actions, tool_steps=tool_steps,
    ))


def _assert_outcome_only(text: str) -> None:
    for i in range(1, _LONG_EPISODE + 1):
        assert f"> take item {i}\n" in text
    assert "is in your hand" not in text
    assert "thinking about item" not in text


def _assert_full_trajectory(text: str) -> None:
    for i in range(1, _LONG_EPISODE + 1):
        assert f"  act    │ take item {i}\n" in text
        assert f"  obs    │ item {i} is in your hand" in text
        assert f"  reason │ thinking about item {i}" in text
    assert "omitted" not in text


