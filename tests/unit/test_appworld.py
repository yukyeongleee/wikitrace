"""appworld domain: manifest loading, the episode flow over the fake environment, the raw
step it persists, its metrics, and the prompts it renders."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest
from jinja2 import Template

import wikitrace.envs  # noqa: F401  (the built-in registrations)
import wikitrace.envs.appworld.prompts  # noqa: F401
from wikitrace.config import append_result
from wikitrace.config import DatasetConfig
from wikitrace.config import make_split
from wikitrace.envs import resolve
from wikitrace.envs.appworld import adapter as adapter_mod
from wikitrace.envs.appworld import workflow
from wikitrace.envs.appworld.benchmarks import appworld as bench_mod
from wikitrace.envs.appworld.benchmarks.appworld import (
    AppWorldResult,
    AppWorldSample,
    load_suite_samples,
)
from wikitrace.config import ToolStep
from wikitrace.envs.base import Metrics
from wikitrace.harness.environment import EnvironmentSession, fake
from wikitrace.harness.environment.protocol import request
from wikitrace.harness.environment.tool import SOCKET_NAME
from wikitrace.harness.prompts import render_system
from wikitrace.harness.terminal import FakeTraverseRunner, TerminalResult

REPO = Path(__file__).resolve().parents[2]
GENERATOR = REPO / "scripts" / "setup" / "build_appworld_manifest.py"

_SUPERVISOR = {"first_name": "Joyce", "last_name": "Weaver",
               "email": "joyce-weav@gmail.com", "phone_number": "3155673041"}
_SECRET = "assert model changes match venmo.PaymentRequest"


def _dom():
    return resolve("appworld", "appworld")


def _sample(task_id: str = "82e2fac_1", role: str = "test", root: str = "") -> AppWorldSample:
    return AppWorldSample(
        id=task_id, db_id="test-normal", question="Request $13 from Stacy on Venmo.",
        scenario_id=task_id.rsplit("_", 1)[0], difficulty=2, split_role=role,
        supervisor=_SUPERVISOR, world_datetime="2023-05-18T12:00:00", root=root,
    )


# ---------------------------------------------------------------------------
# harness + manifest
# ---------------------------------------------------------------------------


def test_harness_is_the_environment_kind():
    spec = _dom().benchmark.harness_spec("test-normal")
    assert spec.kind == "environment" and spec.environment and spec.terminal
    assert not spec.input_on_disk


def test_both_suites_share_one_train_group():
    b = _dom().benchmark
    assert b.train_group("test-normal") == b.train_group("test-challenge") == "appworld"


def _fake_install(tmp_path: Path, task_ids: list[str]) -> Path:
    root = tmp_path / "appworld"
    for t in task_ids:
        d = root / "data" / "tasks" / t
        d.mkdir(parents=True)
        (d / "specs.json").write_text(json.dumps({
            "instruction": f"Task {t}", "supervisor": _SUPERVISOR,
            "datetime": "2023-05-18T12:00:00", "db_version": "0.2.0",
        }))
    return root


def _entries(ids: list[str], difficulty: int = 2) -> list[dict]:
    return [{"id": t, "scenario_id": t.rsplit("_", 1)[0], "difficulty": difficulty} for t in ids]


def test_manifest_loads_as_an_explicit_role_split(tmp_path, monkeypatch):
    train = ["aaa_1", "aaa_2", "aaa_3", "bbb_1", "bbb_2", "bbb_3"]
    seen = ["ccc_1", "ccc_2", "ccc_3"]
    unseen = ["ddd_1", "ddd_2", "ddd_3"]
    root = _fake_install(tmp_path, train + seen + unseen)
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({
        "train": _entries(train), "seen_test": _entries(seen),
        "unseen_test": _entries(unseen, 3), "unseen_excluded": [],
    }))
    monkeypatch.setattr(bench_mod, "SPLITS_PATH", splits)
    cfg = DatasetConfig(appworld_root=str(root))
    cfg = cfg.model_copy(update={"appworld_root": str(root)})

    samples = load_suite_samples(cfg, "test-normal")
    assert [s.id for s in samples] == train + seen
    assert [s.split_role for s in samples] == ["train"] * 6 + ["test"] * 3
    assert samples[0].question == "Task aaa_1" and samples[0].root == str(root)
    assert samples[0].supervisor == _SUPERVISOR
    assert samples[0].world_datetime == "2023-05-18T12:00:00"

    unseen_samples = load_suite_samples(cfg, "test-challenge")
    assert [s.id for s in unseen_samples][6:] == unseen
    assert unseen_samples[-1].difficulty == 3

    # The explicit-role split ignores the config sizes; the smoke opt-in slices the head
    # in whole scenarios because the manifest keeps a scenario's tasks adjacent.
    sp = make_split(samples, cfg, "test-normal")
    assert len(sp.train) == 6 and len(sp.test) == 3
    smoke = cfg.model_copy(update={"limit_explicit_split": True, "train_size": 3, "test_size": 3})
    sp = make_split(samples, smoke, "test-normal")
    assert {s.scenario_id for s in sp.train} == {"aaa"}

    with pytest.raises(ValueError, match="unknown AppWorld suite"):
        load_suite_samples(cfg, "appworld-dev")


def test_committed_manifest_keeps_scenarios_whole_and_interleaves_difficulty():
    assert bench_mod.SPLITS_PATH.exists(), "the frozen manifest is a committed artifact"
    m = json.loads(bench_mod.SPLITS_PATH.read_text())
    assert (len(m["train"]), len(m["seen_test"]), len(m["unseen_test"])) == (54, 111, 99)
    for key in ("train", "seen_test", "unseen_test"):
        ids = [e["id"] for e in m[key]]
        assert len(set(ids)) == len(ids)
        # Three adjacent tasks per scenario, numbered 1..3.
        for i in range(0, len(m[key]), 3):
            block = m[key][i:i + 3]
            assert len({e["scenario_id"] for e in block}) == 1
            assert [e["id"].rsplit("_", 1)[1] for e in block] == ["1", "2", "3"]
    # The head slice spans both difficulties: the smoke's two train scenarios differ.
    assert {e["difficulty"] for e in m["train"][:6]} == {2, 3}
    assert {e["difficulty"] for e in m["unseen_test"]} == {3}
    assert len(m["unseen_excluded"]) == 32
    assert not (set(m["unseen_excluded"]) & {e["scenario_id"] for e in m["unseen_test"]})


def test_manifest_generator_is_seeded_and_interleaves(tmp_path):
    spec = importlib.util.spec_from_file_location("build_appworld_manifest", GENERATOR)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    data = tmp_path / "data"
    (data / "datasets").mkdir(parents=True)
    (data / "version.txt").write_text("0.2.0")

    def scenario(split: str, sid: str, difficulty: int) -> None:
        for n in (1, 2, 3):
            d = data / "tasks" / f"{sid}_{n}" / "ground_truth"
            d.mkdir(parents=True)
            (d / "metadata.json").write_text(json.dumps({"difficulty": difficulty}))
        with (data / "datasets" / f"{split}.txt").open("a") as fh:
            fh.write("".join(f"{sid}_{n}\n" for n in (1, 2, 3)))

    for i, level in enumerate((1, 2, 2, 3, 3, 2)):
        scenario("train", f"t{i}", level)
    for i, level in enumerate((2, 3, 1)):
        scenario("test_normal", f"n{i}", level)
    for i in range(40):
        scenario("test_challenge", f"c{i:02d}", 3)
    gen.DATA = data
    first, second = gen.build(), gen.build()
    assert first == second
    train_levels = [e["difficulty"] for e in first["train"]]
    assert train_levels[:12] == [2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3] and 1 not in train_levels
    assert [e["difficulty"] for e in first["seen_test"]] == [2, 2, 2, 3, 3, 3]
    assert len(first["unseen_test"]) == 33 * 3 and len(first["unseen_excluded"]) == 7


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _result(**kw) -> AppWorldResult:
    base = dict(sample_id="82e2fac_1", db_id="test-normal", question="q",
                scenario_id="82e2fac", difficulty=2)
    base.update(kw)
    return AppWorldResult(**base)


def test_metrics_carry_tgc_partial_credit_and_the_scenario():
    b = _dom().benchmark
    m = b.metrics_from_result(_result(tgc=False, tests_passed=5, tests_total=7))
    assert m.primary == 0.0 and m.secondary == pytest.approx(5 / 7) and not m.success
    assert m.extra == {"scenario_id": "82e2fac", "difficulty": 2,
                       "tests_passed": 5, "tests_total": 7}
    assert b.metrics_from_result(_result(tests_total=0)).secondary is None
    # A resumed run reads the row's `metrics` dict straight back into Metrics.
    assert Metrics(**json.loads(m.model_dump_json())).extra == m.extra


def test_summarize_reports_tgc_over_tasks_and_sgc_over_scenarios():
    b = _dom().benchmark
    rows = [
        Metrics(primary=1.0, success=True, extra={"scenario_id": "a"}),
        Metrics(primary=1.0, success=True, extra={"scenario_id": "a"}),
        Metrics(primary=0.0, success=False, extra={"scenario_id": "a"}),
        Metrics(primary=1.0, success=True, extra={"scenario_id": "b"}),
        Metrics(primary=1.0, success=True, extra={"scenario_id": "b"}),
        Metrics(primary=1.0, success=True, extra={"scenario_id": "b"}),
    ]
    s = b.summarize(rows)
    assert s["tgc"] == pytest.approx(5 / 6) and s["sgc"] == 0.5 and s["n_scenarios"] == 2
    assert s["success_rate"] == s["tgc"] and s["refusal_rate"] == 0.0


# ---------------------------------------------------------------------------
# the episode over the fake environment, and the raw step it leaves
# ---------------------------------------------------------------------------


# The tail Pi appends when it size-caps a tool result, so the result reads as "truncated".
_PI_CAPPED = "You take the key fr\n\n[...truncated, 14 chars omitted]"


def _turns_and_context():
    calls = [
        ("c1", "exec", {"code": "print(apis.api_docs.show_app_descriptions())"}, "examine table",
         "On the table you see a key."),
        ("c2", "read", {"path": "/work/corpus/knowledge/document-0001.md"}, None,
         "# document-0001"),
        # Pi size-capped this output in the agent's context; the server recorded it whole.
        ("c3", "exec", {"code": "take_key()"}, "take key", _PI_CAPPED),
        ("c4", "exec", {"code": "apis.supervisor.complete_task()"}, "put key in box",
         "You put the key in the box. Task complete."),
    ]
    turns, context = [], []
    for i, (cid, name, args, _cmd, out) in enumerate(calls):
        turns.append({"turn": i, "type": "function_call", "role": "assistant",
                      "call_id": cid, "name": name, "arguments": args})
        context.append({"role": "assistant", "content": [
            {"type": "thinking", "thinking": f"reason {cid}"},
            {"type": "toolCall", "id": cid, "name": name, "arguments": args}]})
        context.append({"role": "toolResult", "toolCallId": cid,
                        "content": [{"type": "text", "text": out}]})
    return calls, turns, context


# The fake game with AppWorld's evaluation block: a test report handed over at close.
_SCORED_FAKE_ENV = f"""
from wikitrace.harness.environment.fake import FakeEnvironment

class Env(FakeEnvironment):
    def evaluation(self):
        return {{"success": True, "tests": {{
            "passed": 2, "total": 2,
            "passes": [{{"requirement": "assert answers match.", "label": "no_op_fail"}}],
            "failures": [{{"requirement": {_SECRET!r}, "label": None, "trace": "AssertionError"}}],
        }}}}

def make_env(spec):
    return Env(task=str(spec.get("task", "put the key in the box")))
"""


def _serve_on_host(monkeypatch, factory: str) -> None:
    """Run the workflow's episodes on a host server over ``factory``, not in the AppWorld image."""
    monkeypatch.setattr(workflow, "AppWorldSession", lambda install, task_id, out_dir: EnvironmentSession(
        factory=factory, spec={"appworld_root": str(install), "task_id": task_id},
        out_dir=out_dir, step_limit=bench_mod.MAX_STEPS,
    ))


def _run_fake_episode(tmp_path, monkeypatch, *, with_evaluation: bool = True,
                      lenient: bool = False, **extra):
    """Drive one episode: the fake env behind the socket, the fake runner acting on it."""
    factory = f"{fake.__file__}:make_env"
    if with_evaluation:
        (tmp_path / "scored_env.py").write_text(_SCORED_FAKE_ENV, encoding="utf-8")
        factory = f"{tmp_path / 'scored_env.py'}:make_env"
    _serve_on_host(monkeypatch, factory)
    calls, turns, context = _turns_and_context()
    seen: dict = {}

    def handler(**kw):
        seen.update(kw)
        sock = str(kw["env_tool"].socket_dir / SOCKET_NAME)
        for _cid, name, _args, cmd, _out in calls:
            if name == "exec":
                request(sock, {"op": "step", "input": cmd})
        return TerminalResult(
            turns=turns, read_set=["knowledge/document-0001.md"], final_context=context,
            usage={"cost": 0.01}, usage_calls=[{"model_returned": "m", "cost": 0.01}],
        )

    sample = _sample(root=str(tmp_path / "install")).model_copy(update={"lenient_answer": lenient})
    outcome = _dom().run_traverse(
        sample, FakeTraverseRunner(handler), None, tmp_path / "sessions",
        tools="read,bash", max_turns=70, sentinel_name="__out.json",
        max_retries=1, repair_fn=None, prompt_strategy="free-routing", **{"memory_text": "", **extra},
    )
    return sample, outcome, seen


def test_a_user_prefix_opens_the_user_message_and_nothing_else_moves(tmp_path, monkeypatch):
    """The seam that moves an method's memory to the start of the user message."""
    _, _, plain = _run_fake_episode(tmp_path, monkeypatch)
    _, _, seen = _run_fake_episode(tmp_path, monkeypatch, user_prefix="\n\nWORKFLOWS\n")
    assert seen["user_msg"] == "WORKFLOWS\n\n" + plain["user_msg"]
    assert seen["system_prompt"] == plain["system_prompt"]


def test_episode_scores_from_the_environment_and_records_the_calls(tmp_path, monkeypatch):
    sample, outcome, seen = _run_fake_episode(tmp_path, monkeypatch, memory_layers=("R", "C", "K"))
    r = outcome.result
    assert not outcome.generation_failed
    assert r.tgc and (r.tests_passed, r.tests_total) == (2, 2)
    assert r.steps == 3 and r.ended_by == "done" and not r.step_limit_hit
    assert r.report["failures"][0]["requirement"] == _SECRET
    # The agent's calls, an `exec` call spelled as its code, paired with their results.
    assert [s.command for s in r.tool_steps][:2] == [
        "print(apis.api_docs.show_app_descriptions())",
        "read /work/corpus/knowledge/document-0001.md",
    ]
    assert r.tool_steps[0].reasoning == "reason c1"
    assert r.tool_steps[-1].result == "You put the key in the box. Task complete."
    # An executed chunk's output is the server's whole record, not Pi's truncated copy.
    # The server's record stays whole; the compiler's copy is the capped one the agent read.
    assert r.outputs[1] == "You take the key from the table."
    assert r.tool_steps[2].result == _PI_CAPPED and r.tool_steps[2].state == "truncated"
    assert r.tool_steps[1].result == "# document-0001"
    # No sentinel and no declared citation: the read set is the whole record.
    assert r.citations[0].cited["K"] == [] and r.citations[0].read["K"] == ["document-0001"]
    # The runner was handed the exec tool over the server's socket and the suite's turn cap.
    assert seen["env_tool"].name == "exec" and seen["max_turns"] == bench_mod.MAX_TURNS == 150
    assert seen["label"] == "solver"
    # The session dir holds the server's log and one evaluation file, tests block included.
    sdir = tmp_path / "sessions" / sample.id / "solver"
    assert sorted(p.name for p in sdir.iterdir()) == ["env_server.log", "evaluation.json"]
    evaluation = json.loads((sdir / "evaluation.json").read_text())
    assert list(evaluation) == ["success", "ended_by", "steps", "invalid_commands", "tests"]
    assert evaluation["success"] is True and evaluation["steps"] == 3
    assert evaluation["tests"]["passed"] == 2 and evaluation["tests"]["failures"][0]["requirement"] == _SECRET
    assert r.report == evaluation["tests"]


def test_episode_is_judged_leniently_only_when_its_sample_asks(tmp_path, monkeypatch):
    _, plain, _ = _run_fake_episode(tmp_path, monkeypatch)
    assert plain.result.tgc_lenient is None
    seen = []
    monkeypatch.setattr(workflow, "gold_answer_is_null", lambda root, t: seen.append(t) or True)
    _, judged, _ = _run_fake_episode(tmp_path, monkeypatch, lenient=True)
    assert judged.result.tgc_lenient is True and seen == ["82e2fac_1"]


def _fake_appworld(monkeypatch, **modules) -> None:
    """Stand an `appworld` package made of ``modules`` in for the simulator."""
    monkeypatch.setitem(sys.modules, "appworld", types.ModuleType("appworld"))
    for name, attrs in modules.items():
        module = types.ModuleType(f"appworld.{name}")
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, f"appworld.{name}", module)
        setattr(sys.modules["appworld"], name, module)
    monkeypatch.setenv("APPWORLD_ROOT", "unset")
    monkeypatch.delenv("APPWORLD_CACHE", raising=False)
    # The host grader guards sqlite3's connect for its whole process; restored after the test.
    import sqlite3

    monkeypatch.setattr(sqlite3, "connect", sqlite3.connect)
    monkeypatch.setattr(sqlite3.dbapi2, "connect", sqlite3.dbapi2.connect)


def test_host_evaluation_reports_the_tracker_as_the_tests_block(tmp_path, monkeypatch):
    from wikitrace.envs.appworld.envs import appworld as env_mod

    seen: dict = {}
    tracker = types.SimpleNamespace(
        success=False, pass_count=5, num_tests=6,
        passes=[{"requirement": "assert no unexpected changes.", "label": "no_op_fail"}],
        failures=[{"requirement": "assert answers match.", "label": "no_op_fail", "trace": "..."}],
    )
    _fake_appworld(monkeypatch, evaluator={"evaluate_task": lambda **kw: seen.update(kw) or tracker})
    assert env_mod.evaluate(str(tmp_path), "82e2fac_1") == {"success": False, "tests": {
        "passed": 5, "total": 6,
        "passes": [{"requirement": "assert no unexpected changes.", "label": "no_op_fail"}],
        "failures": [{"requirement": "assert answers match.", "label": "no_op_fail", "trace": "..."}],
    }}
    assert seen == {"task_id": "82e2fac_1", "experiment_name": env_mod.EXPERIMENT_NAME,
                    "suppress_errors": True, "save_report": False}
    assert os.environ["APPWORLD_ROOT"] == str(tmp_path)


def test_the_episode_world_loads_without_the_gold_and_saves_its_state_on_close(monkeypatch):
    from wikitrace.envs.appworld.envs import appworld as env_mod

    calls: list = []

    class World:
        def __init__(self, **kw):
            calls.append(("init", kw))

        def save(self):
            calls.append("save")

        def close(self):
            calls.append("close")

    _fake_appworld(monkeypatch)
    monkeypatch.setattr(sys.modules["appworld"], "AppWorld", World, raising=False)
    env = env_mod.make_env({"appworld_root": "/appworld", "task_id": "82e2fac_1"})
    env.close()
    assert calls == [("init", {"task_id": "82e2fac_1", "experiment_name": env_mod.EXPERIMENT_NAME,
                               "load_ground_truth": False}), "save", "close"]
    assert not hasattr(env, "evaluation")


def test_an_episode_without_an_evaluation_is_a_generation_failure(tmp_path, monkeypatch):
    _, outcome, _ = _run_fake_episode(tmp_path, monkeypatch, with_evaluation=False)
    assert outcome.generation_failed and not outcome.result.tgc
    assert "without an evaluation" in outcome.result.generation_error
    assert outcome.result.steps == 3  # the work done is still on the record


def test_persisted_raw_step_withholds_the_report_and_round_trips(tmp_path, monkeypatch):
    sample, outcome, _ = _run_fake_episode(tmp_path, monkeypatch)
    dom = _dom()
    corpus = tmp_path / "corpus"
    step_dir = corpus / "raw" / "step-0000"
    dom.persist_traverse_trajectory(step_dir, sample, outcome)
    append_result(corpus, "step-0000", outcome.result)

    assert sorted(p.name for p in step_dir.iterdir()) == ["metadata.md", "solver.full.jsonl"]
    for p in step_dir.iterdir():
        assert _SECRET not in p.read_text(encoding="utf-8"), p.name
    meta = (step_dir / "metadata.md").read_text()
    assert meta == (
        f"# {sample.id}\n\n"
        f"- instruction: {sample.question}\n"
        "- supervisor: Joyce Weaver <joyce-weav@gmail.com>, phone 3155673041\n"
        "- world_datetime: 2023-05-18T12:00:00\n"
        "- success: True\n"
        "- steps: 3 exec calls\n"
    )
    log = [json.loads(ln) for ln in (step_dir / "solver.full.jsonl").read_text().splitlines()]
    assert log[0]["role"] == "system" and log[1]["role"] == "user"
    assert log[2]["type"] == "function_call" and log[2]["reasoning"] == "reason c1"
    assert log[-1]["type"] == "answer" and log[-1]["parsed"] is None

    back = dom.read_result(step_dir)
    assert back == outcome.result
    facts = dom.read_raw_step(step_dir)
    assert facts.outcome == "success" and facts.metrics == {"success": 1.0}


def test_metadata_names_what_ended_an_episode_the_task_did_not():
    r = AppWorldResult(sample_id="d194965_1", db_id="test-normal", question="q",
                       tests_passed=3, tests_total=5, steps=50, step_limit_hit=True,
                       ended_by="budget_exhausted")
    steps = "50 exec calls (ended by the harness: exec budget ran out)"
    meta = adapter_mod._metadata_text(r)
    assert meta == (
        "# d194965_1\n\n- instruction: q\n- supervisor: \n- world_datetime: \n"
        f"- success: False\n- steps: {steps}\n"
    )
    dom = _dom()
    assert dom.render_outcome_for_compiler(r) == (
        f"## Outcome\noutcome: failure\nsteps: {steps}"
    )
    assert dom.compilation_inputs(r).prediction == f"success=0; steps={steps}"
    # `complete_task` with a wrong result is still the task's own end: no suffix.
    r.steps, r.step_limit_hit, r.ended_by = 12, False, "done"
    assert adapter_mod._steps_text(r) == "12 exec calls"
    r.steps, r.ended_by = 17, "harness_stopped"
    assert adapter_mod._steps_text(r) == "17 exec calls (ended by the harness: session cut off)"


def test_compiler_renders_show_the_outcome_but_not_the_tests(tmp_path, monkeypatch):
    sample, outcome, _ = _run_fake_episode(tmp_path, monkeypatch)
    dom = _dom()
    outcome_text = dom.render_outcome_for_compiler(outcome.result, sample=sample)
    assert outcome_text == "## Outcome\noutcome: success\nsteps: 3 exec calls"
    trajectory = dom.render_trajectory_for_compiler(outcome.result)
    assert "print(apis.api_docs.show_app_descriptions())" in trajectory
    assert "reason c1" in trajectory
    ci = dom.compilation_inputs(outcome.result, sample=sample)
    assert ci.success and ci.metrics == {"success": 1.0, "steps": 3.0, "step_limit_hit": 0.0}
    assert _SECRET not in json.dumps(ci.__dict__) and "tests" not in json.dumps(ci.__dict__)
    row = dom.sample_row(outcome.result, "x/metadata.md")
    assert row["report"]["failures"][0]["requirement"] == _SECRET  # the run's record keeps it
    assert (row["tgc"], row["tests_passed"], row["tests_total"]) == (True, 2, 2)
    # The episode fields every environment domain's row carries.
    assert (row["steps"], row["ended_by"], row["invalid_commands"]) == (3, "done", 0)


def test_compiler_render_holds_a_whole_episode_and_drops_hook_refused_calls():
    from wikitrace.config import ToolStep

    steps = [ToolStep(command=f"print({i})", result=f"out {i}") for i in range(60)]
    text = adapter_mod._render_tool_steps(steps)
    assert "[1]\n  call   │ print(0)\n  result │ out 0" in text and "[60]" in text
    assert "omitted" not in text
    steps.append(ToolStep(command="print(60)", result="x" * 400))
    text = adapter_mod._render_tool_steps(steps)
    assert "[61]" in text and "omitted" not in text
    assert f"result │ {'x' * 400}" in text
    # A refused call never reached the shell: not numbered, not shown.
    blocked = ToolStep(command="print(1)", is_error=True,
                       result="one exec per turn: read this output before sending the next chunk")
    text = adapter_mod._render_tool_steps([steps[0], blocked, steps[1]])
    assert "one exec per turn" not in text and "[3]" not in text
    assert text == "[1]\n  call   │ print(0)\n  result │ out 0\n[2]\n  call   │ print(1)\n  result │ out 1"


def test_index_entries_carry_the_outcome_and_no_score_column():
    from wikitrace.config import Compilation, Outcome
    from wikitrace.memory.store import RawStepFacts, render_compilation_index, render_raw_index_from_steps

    shape = _dom().raw_layer_shape
    assert shape.compilation_metric_keys == () and shape.index_score == ""
    comp = Compilation(step_id="step-0000", outcome=Outcome.SUCCESS, metrics={"success": 1.0},
                       title="[success] Venmo request", question="Request $13.", body="")
    idx = render_compilation_index([comp], entry_source="title", raw_shape=shape)
    assert "1. [step-0000](/work/corpus/compilation/step-0000.md): success | Venmo request" in idx
    assert "(success=..)" not in idx and "()" not in idx and "success: 0 | 1" not in idx
    raw = render_raw_index_from_steps(
        [("step-0000", RawStepFacts(outcome="success", tail="Request $13."))], raw_shape=shape)
    assert "1. [step-0000](/work/corpus/raw/step-0000/): success | Request $13." in raw


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------


def _system(**kw) -> str:
    ctx = dict(exemplar=workflow.exemplar_text(_SUPERVISOR, '[{"name": "spotify"}]'),
               access="none", prompt_strategy="free-routing", substrate="RCK",
               keep_index=True, keep_links=True)
    ctx.update(kw)
    return render_system("appworld_solver", "traverse", **ctx)


def test_exemplar_is_the_shipped_prompt_minus_its_task_block():
    text = workflow.exemplar_text(_SUPERVISOR, '[{"name": "spotify"}]')
    assert text.startswith("I am your supervisor")
    assert "USER:" not in text and "ASSISTANT:" not in text
    assert "**Key instructions**" in text and "My name is: Joyce Weaver" in text
    assert '[{"name": "spotify"}]' in text
    assert "solve the actual task" not in text and "{{" not in text
    assert bench_mod.EXEMPLAR_PATH.read_text().count(workflow._TASK_MARKER) == 1
    assert bench_mod.LICENSE_PATH.exists()
    hashes = bench_mod.data_file_hashes()
    assert set(hashes) >= {"react_code_agent_instructions.txt"} and all(len(h) == 64 for h in hashes.values())


def test_the_compiler_excerpt_is_the_solver_prompt_without_its_worked_example():
    excerpt = _dom().prompt_profile("test-normal").solver_prompt
    intro, keys = workflow.authors_instructions()
    solver = workflow.exemplar_text(_SUPERVISOR, "[]")
    assert intro in solver and keys in solver
    assert excerpt.startswith("I am your supervisor, and you are an AI Assistant")
    assert intro.endswith("Let's start with the task")
    assert excerpt.index(intro) < excerpt.index(keys) < excerpt.index("Runtime notes:")
    assert keys.startswith("**Key instructions**:") and "D. Task-completion instructions:" in keys
    assert "- Send each chunk of code as one `exec` call. Code written in your reply text does not run." in keys
    assert "code blocks" not in keys and "in every step" not in keys
    for example in ("My name is", "Task: How many playlists", "reasoning: ", "\noutput:\n", "USER:"):
        assert example not in excerpt


# The exemplar as the solver reads it, for the test supervisor; moves only with a prompt change.
EXEMPLAR_SHA256 = "baa879f22f2fc9b759d12f08ea7e174832d7299d326446fbfbd7fd4c04dd6be6"


def test_the_exemplar_turns_are_exec_calls_and_the_authors_text_is_kept_byte_for_byte():
    text = workflow.exemplar_text(_SUPERVISOR, '[{"name": "spotify"}]')
    head = bench_mod.EXEMPLAR_PATH.read_text().lstrip().rpartition(workflow._TASK_MARKER)[0]
    shipped = Template(head).render(main_user=_SUPERVISOR, app_descriptions='[{"name": "spotify"}]')
    # Putting the role markers back recovers the shipped example exactly, minus the prose
    # the authors wrote before each code block.
    minus_prose = re.sub(r"^ASSISTANT:\n.*?```python", "ASSISTANT:\n```python", shipped, flags=re.M | re.S)
    back = re.sub(r"^> exec\n", "ASSISTANT:\n", text, flags=re.M)
    back = re.sub(r"^output:\n", "USER:\nOutput:\n", back, flags=re.M)
    back = "USER:\n" + back.replace(
        "\nThe example ends here.\n\n**Key instructions**", "\nUSER:\n**Key instructions**"
    )
    assert back == minus_prose.rstrip()
    assert minus_prose.count("ASSISTANT:\n") == shipped.count("ASSISTANT:\n") == 10
    assert text.count("\n> exec\n```python\n") == 10 and "reasoning: " not in text
    assert "Okay. Let's first find which APIs are available" in shipped and "Okay. Let's first find" not in text
    assert text.count("\noutput:\n") == 10 and "Output:" not in text
    assert ("\noutput:\n```\nExecution successful.\n\nEpisode over\n```\n\nThe example ends here.\n\n"
            "**Key instructions**:\n") in text
    turn = text[text.index("> exec"):].split("\n")[:13]
    assert turn == [
        "> exec",
        "```python",
        "print(apis.api_docs.show_api_descriptions(app_name='spotify'))",
        "```",
        "",
        "output:",
        "```",
        "[",
        " ...",
        ' "login : Login to your account.",',
        ' "logout : Logout from your account.",',
        " ...",
        "]",
    ]
    assert hashlib.sha256(text.encode()).hexdigest() == EXEMPLAR_SHA256


def test_system_prompt_names_the_exec_tool_and_no_wiki_under_no_memory():
    text = _system(substrate="no-memory")
    assert "`exec` tool" in text and "when the exec\n  budget runs out" in text
    # No budget number reaches the agent; the three notes hold the tool, the sandbox, the end.
    assert "50" not in text.split("Runtime notes:")[1]
    assert text.split("Runtime notes:")[1].count("\n- ") == 3
    assert "KNOWLEDGE WIKI" not in text and "CASE LIBRARY" not in text
    assert "__out.json" not in text and "document_ids" not in text


@pytest.mark.parametrize("access", ["none", "inject", "traverse"])
@pytest.mark.parametrize("substrate", ["no-memory", "R00", "0C0", "00K", "RC0", "RCK"])
@pytest.mark.parametrize("prompt_strategy", ["free-routing", "top-down"])
@pytest.mark.parametrize("knobs", [(True, True), (False, False), (True, False)])
def test_every_prompt_combination_renders(access, substrate, prompt_strategy, knobs):
    ki, kl = knobs
    text = _system(access=access, substrate=substrate, prompt_strategy=prompt_strategy,
                   keep_index=ki, keep_links=kl)
    assert text.strip() and "{{" not in text


def test_env_adapter_imports_only_the_standard_library_and_the_simulator():
    import ast

    src = Path(adapter_mod.__file__).with_name("envs") / "appworld.py"
    names = set()
    for node in ast.walk(ast.parse(src.read_text())):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert "wikitrace" not in names and "appworld" in names


# More calls than any per-case cap the renders ever carried.
_LONG_EPISODE = 60


def _long_frozen_step(corpus: Path) -> None:
    """One frozen appworld step whose episode is longer than any former render cap."""
    step = corpus / "raw" / "step-0000"
    step.mkdir(parents=True)
    tool_steps = [ToolStep(reasoning=f"thinking about item {i}",
                           command=f"apis.venmo.request(amount={i})",
                           result=f"request {i} sent")
                  for i in range(1, _LONG_EPISODE + 1)]
    turns = [{"type": "message", "role": "system", "content": "system"},
             {"type": "message", "role": "user", "content": "Here is the task. request the money"}]
    turns += [{"type": "function_call", "name": "exec",
               "arguments": {"code": f"apis.venmo.request(amount={i})"},
               "reasoning": f"thinking about item {i}", "result": f"request {i} sent"}
              for i in range(1, _LONG_EPISODE + 1)]
    turns.append({"type": "answer", "role": "assistant", "parsed": None})
    (step / "solver.full.jsonl").write_text(
        "".join(json.dumps(t) + "\n" for t in turns), encoding="utf-8")
    (step / "metadata.md").write_text(
        "# 82e2fac_1\n\n- instruction: request the money\n", encoding="utf-8")
    append_result(corpus, "step-0000", AppWorldResult(
        sample_id="82e2fac_1", db_id="test-normal", question="request the money",
        tgc=True, steps=_LONG_EPISODE, tool_steps=tool_steps,
    ))


def _assert_outcome_only(text: str) -> None:
    for i in range(1, _LONG_EPISODE + 1):
        assert f"apis.venmo.request(amount={i})\n" in text
    assert "sent" not in text
    assert "thinking about item" not in text


def _assert_full_trajectory(text: str) -> None:
    for i in range(1, _LONG_EPISODE + 1):
        assert f"  call   │ apis.venmo.request(amount={i})" in text
        assert f"  result │ request {i} sent" in text
        assert f"  reason │ thinking about item {i}" in text
    assert "omitted" not in text


# ---------------------------------------------------------------------------
# the episode container and the host grading
# ---------------------------------------------------------------------------


def _install(tmp_path: Path) -> Path:
    """A stand-in AppWorld install: the venv's dist-info and one task's data."""
    install = tmp_path / "install"
    (install / ".venv/lib/python3.11/site-packages/appworld-0.2.0.dev0.dist-info").mkdir(parents=True)
    (install / "data/tasks/82e2fac_1/dbs").mkdir(parents=True)
    (install / "data/tasks/82e2fac_1/ground_truth").mkdir()
    (install / "data/tasks/82e2fac_1/specs.json").write_text("{}")
    return install


def test_the_episode_container_holds_one_task_without_its_gold(tmp_path, monkeypatch):
    from wikitrace.envs.appworld.session import AppWorldSession, image_tag

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-leak")
    install = _install(tmp_path)
    sdir = tmp_path / "sessions" / "82e2fac_1" / "solver"
    env = AppWorldSession(install, "82e2fac_1", sdir)
    data, task = install / "data", install / "data/tasks/82e2fac_1"
    assert env.argv() == [
        "docker", "run", "--rm", "-i", "--pull", "never", "--name", env.container.name,
        "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "APPWORLD_ROOT=/appworld",
        "--mount", f"type=bind,source={env.socket_dir},target=/env",
        "--mount", f"type=bind,source={sdir / 'appworld/experiments/outputs'},target=/appworld/experiments/outputs",
        "--mount", f"type=bind,source={data / 'api_docs'},target=/appworld/data/api_docs,readonly",
        "--mount", f"type=bind,source={data / 'base_dbs'},target=/appworld/data/base_dbs,readonly",
        "--mount", f"type=bind,source={task / 'specs.json'},target=/appworld/data/tasks/82e2fac_1/specs.json,readonly",
        "--mount", f"type=bind,source={task / 'dbs'},target=/appworld/data/tasks/82e2fac_1/dbs,readonly",
        image_tag(install), "python", "/srv/server.py", "--factory", "/srv/envs/appworld.py:make_env",
        "--socket", f"/env/{SOCKET_NAME}", "--out", "/tmp/out",
        "--step-limit", str(bench_mod.MAX_STEPS), "--end-command", "", "--stdin-lifeline",
    ]
    assert not any("ground_truth" in a or "sk-leak" in a for a in env.argv())
    assert env.spec == {"appworld_root": "/appworld", "task_id": "82e2fac_1"}
    assert re.fullmatch(r"wikitrace-appworld-env:0\.2\.0\.dev0-[0-9a-f]{12}", image_tag(install))


def test_a_relative_session_dir_is_mounted_by_its_absolute_path(tmp_path, monkeypatch):
    from wikitrace.envs.appworld.session import AppWorldSession

    install = _install(tmp_path)
    monkeypatch.chdir(tmp_path)
    env = AppWorldSession(install, "82e2fac_1", Path("sessions/82e2fac_1/solver"))
    sources = [a.split(",")[1].removeprefix("source=") for a in env.argv() if a.startswith("type=bind")]
    assert sources and all(os.path.isabs(s) for s in sources)
    assert str(tmp_path / "sessions/82e2fac_1/solver/appworld/experiments/outputs") in sources


def test_a_missing_episode_image_names_its_build_script(monkeypatch):
    from wikitrace.envs.appworld import session as session_mod

    monkeypatch.setattr(session_mod.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, b"", b"no such image"))
    with pytest.raises(RuntimeError, match="appworld_env_image.sh"):
        session_mod.require_image("wikitrace-appworld-env:absent")


def _graded_session(tmp_path, monkeypatch, stdout: str, saved: bool = True):
    from wikitrace.envs.appworld import session as session_mod

    install = _install(tmp_path)
    sdir = tmp_path / "solver"
    env = session_mod.AppWorldSession(install, "82e2fac_1", sdir)
    if saved:
        (env.episode_root / "experiments/outputs/wikitrace/tasks/82e2fac_1/dbs").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(session_mod.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw.get("cwd")))
                        or subprocess.CompletedProcess(cmd, 0, stdout, "evaluator stderr"))
    return env, calls


_HEAD = {"success": False, "ended_by": "done", "steps": 3, "invalid_commands": 0}


def test_the_host_grades_the_saved_state_into_the_servers_evaluation(tmp_path, monkeypatch):
    from wikitrace.envs.appworld.session import ENV_MODULE

    block = {"success": True, "tests": {"passed": 2, "total": 2, "passes": [], "failures": []}}
    env, calls = _graded_session(tmp_path, monkeypatch, "a warning\n" + json.dumps(block) + "\n")
    evaluation = env._grade(dict(_HEAD))
    assert list(evaluation) == ["success", "ended_by", "steps", "invalid_commands", "tests"]
    assert evaluation["success"] is True and evaluation["tests"] == block["tests"]
    assert calls == [([str(env.install / ".venv/bin/python"), "-P", str(ENV_MODULE), ".", "82e2fac_1"],
                      env.episode_root)]
    assert (env.episode_root / "data").resolve() == env.install / "data"


def test_an_episode_without_saved_state_or_a_report_keeps_the_head(tmp_path, monkeypatch):
    env, calls = _graded_session(tmp_path, monkeypatch, "", saved=False)
    assert env._grade(dict(_HEAD)) == _HEAD and calls == []
    env, calls = _graded_session(tmp_path / "b", monkeypatch, "Traceback: boom\n")
    assert env._grade(dict(_HEAD)) == _HEAD and len(calls) == 1
    assert "host evaluation failed" in (env.out_dir / "env_server.log").read_text()


def test_a_host_evaluator_timeout_is_a_recorded_failure(tmp_path, monkeypatch):
    from wikitrace.envs.appworld import session as session_mod

    env, _ = _graded_session(tmp_path, monkeypatch, "")

    def run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])

    monkeypatch.setattr(session_mod.subprocess, "run", run)
    assert env._grade(dict(_HEAD)) == _HEAD
    assert "host evaluation failed: timed out" in (env.out_dir / "env_server.log").read_text()


def _hostile_tree(root: Path, outside: Path) -> None:
    """What an episode's code can leave in its outputs dir: dirs and files it took all rights from."""
    deep = root / "experiments/outputs/x/y"
    (deep / "z").mkdir(parents=True)
    (deep / "z" / "f").write_text("")
    os.symlink(outside, deep / "link")
    os.chmod(deep / "z", 0)
    os.chmod(deep, 0)


def test_an_episode_dir_is_removed_whatever_permissions_its_code_left(tmp_path):
    from wikitrace.envs.appworld.session import remove_tree

    outside = tmp_path / "host.txt"
    outside.write_text("host")
    os.chmod(outside, 0o640)
    _hostile_tree(tmp_path / "appworld", outside)
    remove_tree(tmp_path / "appworld")
    assert not (tmp_path / "appworld").exists()
    assert outside.read_text() == "host" and outside.stat().st_mode & 0o777 == 0o640
    remove_tree(tmp_path / "appworld")  # absent is fine


def test_a_session_starts_over_a_hostile_leftover_and_cleans_up_a_failed_start(tmp_path, monkeypatch):
    from wikitrace.envs.appworld import session as session_mod

    monkeypatch.setattr(session_mod, "require_image", lambda image: None)
    env = session_mod.AppWorldSession(_install(tmp_path), "82e2fac_1", tmp_path / "solver")
    _hostile_tree(env.episode_root, tmp_path / "host.txt")
    monkeypatch.setattr(EnvironmentSession, "__enter__", lambda self: self)
    assert env.__enter__() is env
    assert [p.name for p in (env.episode_root / "experiments/outputs").iterdir()] == []

    _hostile_tree(env.episode_root / "experiments/outputs", tmp_path / "host.txt")

    def fail(self):
        raise RuntimeError("no server")

    monkeypatch.setattr(EnvironmentSession, "__enter__", fail)
    with pytest.raises(RuntimeError, match="no server"):
        env.__enter__()
    assert not env.episode_root.exists()


def test_the_host_graders_sqlite_refuses_attach(tmp_path, monkeypatch):
    import sqlite3

    from wikitrace.envs.appworld.envs import appworld as env_mod

    tracker = types.SimpleNamespace(success=True, pass_count=1, num_tests=1, passes=[], failures=[])
    _fake_appworld(monkeypatch, evaluator={"evaluate_task": lambda **kw: tracker})
    env_mod.evaluate(str(tmp_path), "82e2fac_1")
    target = tmp_path / "planted.db"
    for sql in (f"ATTACH DATABASE '{target}' AS p", f"VACUUM INTO '{target}'"):
        with pytest.raises(sqlite3.OperationalError, match="too many attached"):
            sqlite3.dbapi2.connect(":memory:").execute(sql)
    assert not target.exists()


def _dataset(tmp_path: Path) -> Path:
    install = tmp_path / "install"
    task = install / "data/tasks/82e2fac_1"
    for rel, text in [
        ("data/version.txt", "0.2.0"), ("data/api_docs/a.json", "{}"), ("data/base_dbs/b.db", "b"),
        ("data/datasets/train.txt", "ids"), ("data/tasks/82e2fac_1/specs.json", "{}"),
        ("data/tasks/82e2fac_1/dbs/c.jsonl", ""), ("data/tasks/82e2fac_1/ground_truth/answer.json", "null"),
    ]:
        (install / rel).parent.mkdir(parents=True, exist_ok=True)
        (install / rel).write_text(text)
    (task / "ground_truth/__pycache__").mkdir()
    return install


def test_the_dataset_check_names_the_files_that_differ_from_the_record(tmp_path, monkeypatch):
    install = _dataset(tmp_path)
    digests = bench_mod.dataset_digests(install)
    assert sorted(digests) == [
        "data/api_docs/a.json", "data/base_dbs/b.db", "data/tasks/82e2fac_1/dbs/c.jsonl",
        "data/tasks/82e2fac_1/ground_truth/answer.json", "data/tasks/82e2fac_1/specs.json",
        "data/version.txt",
    ]
    monkeypatch.setattr(bench_mod, "DATA_DIGEST", bench_mod.aggregate_digest(digests))
    (install / "integrity.json").write_text(json.dumps({"files": digests}))
    (install / "data/tasks/82e2fac_1/ground_truth/__pycache__/evaluation.cpython-311.pyc").write_text("")
    (install / "data/datasets/train.txt").write_text("other ids")
    bench_mod.verify_dataset(install)

    (install / "data/tasks/82e2fac_1/ground_truth/answer.json").write_text('"42"')
    (install / "data/tasks/82e2fac_1/ground_truth/extra.py").write_text("")
    (install / "data/base_dbs/b.db").unlink()
    with pytest.raises(RuntimeError) as err:
        bench_mod.verify_dataset(install)
    assert ("3 file(s) differ" in str(err.value) and "base_dbs/b.db (missing)" in str(err.value)
            and "ground_truth/answer.json (changed)" in str(err.value)
            and "ground_truth/extra.py (added)" in str(err.value))


def test_an_appworld_run_checks_its_dataset_and_its_image(tmp_path, monkeypatch):
    from wikitrace.envs.appworld import session as session_mod

    install = _install(tmp_path)
    calls = []
    monkeypatch.setattr(bench_mod, "verify_dataset", lambda root: calls.append(("dataset", root)))
    monkeypatch.setattr(session_mod, "require_image", lambda image: calls.append(("image", image)))
    bench_mod.AppWorldBenchmarkAdapter().verify_install(DatasetConfig(appworld_root=str(install)))
    assert calls == [("dataset", install), ("image", session_mod.image_tag(install))]
