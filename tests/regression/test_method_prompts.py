"""The three methods' rendered prompts, end to end over the offline games, match the pinned text."""

from __future__ import annotations

import hashlib
import json
import sys
import textwrap
from pathlib import Path

import pytest

from wikitrace.harness.environment import fake
from wikitrace.harness.environment.protocol import request
from wikitrace.harness.llm.openrouter import EmbedResponse
from wikitrace.harness.terminal import FakeTraverseRunner, TerminalResult
from wikitrace.run.build import build
from wikitrace.run.cell import run_cell
from tests.conftest import FakeTransport, fake_game_samples, play_fake_game


# The simulated user's first message per task: two train conversations and one test.
FIRST = {
    "10": "Hi, I want to return the shoes from order #W1234567.",
    "20": "Hello, please cancel my pending order #W2 and refund my gift card.",
    "30": "Hi, I want to return the lamp from order #W3.",
}
LOOKUP = {"action": "get_user_details", "argument": {"user_id": "chen_silva_7485"}}

# A retail environment whose first message is the task, scored at close, with no tau2 import.
_FAKE_ENV = textwrap.dedent(f'''
    FIRST = {FIRST!r}


    class Env:
        def __init__(self, task_id):
            self.task_id, self.n = task_id, 0

        def reset(self):
            return FIRST[self.task_id], {{"source": "user", "task": FIRST[self.task_id]}}

        def step(self, command):
            self.n += 1
            return f"reply {{self.n}}", False, {{"source": "env", "invalid_command": False}}

        def close(self):
            pass

        def evaluation(self):
            return {{"success": True, "reward": 1.0, "score": 1.0, "orchestrator_steps": 2 * self.n + 1,
                    "messages": 0, "num_errors": 0, "stop_token": "", "judge_attempts": 0,
                    "calls": {{}}, "cost": {{}}}}


    def make_env(spec):
        return Env(str(spec["task_id"]))
''')

GOLDEN = Path(__file__).parents[1] / "fixtures/prompts/method_prompts.json"
METHODS = ("no-memory", "kinj", "wikitrace")


class _Transport(FakeTransport):
    """Canned compiler payloads with two insights, and a two-axis embedder."""

    def _env_insight(self):
        return {"title": "take before placing", "guideline": "take the object first",
                "applicability": "when a task moves an object"}

    def embed(self, *, model, texts):
        (text,) = texts
        return EmbedResponse(vectors=[[1.0, 0.0] if "box" in text or "return" in text else [0.0, 1.0]],
                             model_returned=model, raw={"usage": {"prompt_tokens": 5, "cost": 0.001}})


def _recorder(seen: list, play):
    def handle(**kw):
        record = {k: kw[k] for k in ("system_prompt", "user_msg", "citation_prompt")}
        if kw["corpus_dir"] is not None:
            root = Path(kw["corpus_dir"])
            record["corpus"] = {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8")
                                for p in sorted(root.rglob("*")) if p.is_file() and p.name != "corpus.json"}
        record["readonly_files"] = {k: hashlib.sha256(Path(v).read_bytes()).hexdigest()
                                    for k, v in (kw["readonly_files"] or {}).items()}
        seen.append(record)
        sock_dir = kw["env_tool"].socket_dir
        (sock,) = sock_dir.glob("*.sock")
        if sock.stem != kw["env_tool"].name:
            (sock_dir / f"{kw['env_tool'].name}.sock").symlink_to(sock)
        return play(**kw)
    return handle


def _run_arm(cfg_path, benchmark, suite, method, runner, transport):
    return run_cell(cfg_path, benchmark, suite, 0, method, transport=transport, traverse_runner=runner)


def _build(cfg_path, benchmark, suite, runner, transport):
    build(cfg_path, benchmark, suite, 0, transport=transport, traverse_runner=runner)


def _capture(cfg_path, benchmark, suite, play, tmp_path):
    transport, record = _Transport(), {}
    seen: list = []
    runner = FakeTraverseRunner(_recorder(seen, play))
    _build(cfg_path, benchmark, suite, runner, transport)
    record["compiler"] = [m["content"] for c in transport.calls for m in c["messages"]]
    for method in METHODS:
        seen.clear()
        _run_arm(cfg_path, benchmark, suite, method, runner, transport)
        record[method] = seen[-1]
    return json.loads(json.dumps(record).replace(str(tmp_path), "<TMP>"))


def _check(key, record):
    assert record == json.loads(GOLDEN.read_text())[key]


def _first_of_each_role(load):
    """The suite's first two train and first test samples, so a capture runs three episodes."""
    def first(self, dataset_cfg, suite_id):
        samples = load(self, dataset_cfg, suite_id)
        return ([s for s in samples if s.split_role == "train"][:2]
                + [s for s in samples if s.split_role == "test"][:1])
    return first


@pytest.mark.slow
@pytest.mark.parametrize("benchmark,suite", [
    ("alfworld", "valid-seen"), ("scienceworld", "scienceworld"), ("plancraft", "plancraft-seen"),
])
def test_text_game_method_prompts(benchmark, suite, tmp_path, monkeypatch, temp_config_writer):
    from wikitrace.envs.textgame.benchmarks import alfworld, plancraft, scienceworld

    adapter = {"alfworld": alfworld.AlfworldBenchmarkAdapter,
               "scienceworld": scienceworld.ScienceworldBenchmarkAdapter,
               "plancraft": plancraft.PlancraftBenchmarkAdapter}[benchmark]
    if benchmark == "alfworld":
        monkeypatch.setattr(adapter, "load_suite", lambda self, d, s: fake_game_samples(s))
    else:
        monkeypatch.setattr(adapter, "load_suite", _first_of_each_role(adapter.load_suite))
    monkeypatch.setattr(adapter, "env_factory", f"{fake.__file__}:make_env")
    monkeypatch.setattr(adapter, "python", sys.executable)
    monkeypatch.setattr(adapter, "verify_install", lambda self, cfg: None)
    java = tmp_path / "scienceworld" / "jdk" / "bin" / "java"
    java.parent.mkdir(parents=True)
    java.touch()
    (tmp_path / "plancraft").mkdir()
    cfg_path = temp_config_writer({"dataset": {
        "test_size": 1, "train_size": 2, "scienceworld_root": str(tmp_path / "scienceworld"),
        "plancraft_root": str(tmp_path / "plancraft")}})
    _check(benchmark, _capture(cfg_path, benchmark, suite, play_fake_game, tmp_path))


@pytest.mark.slow
def test_tau3_method_prompts(tmp_path, monkeypatch, temp_config_writer):
    from wikitrace.envs.tau3.benchmarks import tau3 as bench_mod

    env = tmp_path / "fake_tau3_env.py"
    env.write_text(_FAKE_ENV, encoding="utf-8")
    manifest = tmp_path / "tau3.manifest.json"
    manifest.write_text(json.dumps({
        "train": [{"task_id": "10", "user_id": "u"}, {"task_id": "20", "user_id": "u"}],
        "test": [{"task_id": "30", "user_id": "u"}],
    }))
    adapter = bench_mod.Tau3BenchmarkAdapter
    monkeypatch.setattr(bench_mod, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(adapter, "env_factory", f"{env}:make_env")
    monkeypatch.setattr(adapter, "python", sys.executable)
    monkeypatch.setattr(adapter, "verify_install", lambda self, cfg: None)

    def play(*, env_tool, **_):
        reply = request(str(env_tool.socket_dir / "act.sock"),
                        {"op": "step", "input": json.dumps(LOOKUP, separators=(",", ":"))})
        turns = [{"type": "function_call", "name": "take_action", "arguments": LOOKUP, "call_id": "c0"}]
        ctx = [{"role": "assistant", "content": [{"type": "toolCall", "id": "c0"}]},
               {"role": "toolResult", "toolCallId": "c0",
                "content": [{"type": "text", "text": reply["observation"]}]}]
        return TerminalResult(final_text="done", turns=turns, final_context=ctx, read_set=[],
                              usage={"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.05})

    cfg_path = temp_config_writer({"dataset": {"tau3_root": str(tmp_path),
                                               "tau3_user_model": "openrouter/openai/gpt-4.1"}})
    _check("tau3", _capture(cfg_path, "tau3", "retail", play, tmp_path))
