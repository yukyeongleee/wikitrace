"""The environment server round-trip, its evaluation file, and the runner's tool wiring."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from wikitrace.harness.environment import EPISODE_OVER, EnvTool, EnvironmentSession
from wikitrace.harness.environment import fake
from wikitrace.harness.environment.tool import (
    CITATION_ENV_VAR, EXTENSIONS_DIR, SOCKET_ENV_VAR, SOCKET_NAME,
)
from wikitrace.harness.terminal import FakeTraverseRunner, PiTraverseRunner, TerminalResult
from wikitrace.memory.citation import citation_message
from wikitrace.harness.terminal.docker import ENV_SOCKET_MOUNT, docker_run_argv
from wikitrace.harness.terminal.runner import _is_episode_over

FACTORY = f"{fake.__file__}:make_env"


def _session(tmp_path: Path, step_limit: int = 10) -> EnvironmentSession:
    return EnvironmentSession(
        factory=FACTORY, spec={"task": "put the key in the box"}, out_dir=tmp_path / "ep",
        step_limit=step_limit, python=sys.executable, startup_timeout=30.0,
    )


def _evaluation(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "ep" / "evaluation.json").read_text(encoding="utf-8"))


def test_episode_runs_to_success_and_leaves_one_evaluation_file(tmp_path):
    with _session(tmp_path) as env:
        obs, info = env.reset()
        assert "Your task is to: put the key in the box." in obs
        assert info["task"] == "put the key in the box"
        assert info["success"] is False and info["score"] == 0.0
        # The socket dir is what the agent's sandbox mounts: it holds the socket alone.
        assert [p.name for p in env.socket_dir.iterdir()] == [SOCKET_NAME]
        assert env.step("dance")["observation"] == "Nothing happens."
        assert env.step("examine table")["observation"] == "On the table you see a key."
        # A step reply is what the agent's tool sees: no score, no limit, no info.
        assert env.step("take key") == {"observation": "You take the key from the table.",
                                        "done": False, "over": False}
        assert env.status()["info"]["score"] == 0.5
        last = env.step("put key in box")
        assert last == {"observation": f"You put the key in the box. Task complete.\n\n{EPISODE_OVER}",
                        "done": True, "over": False}
        assert env.status()["info"]["success"] is True
        # After the episode ends a step is answered, not executed.
        after = env.step("look")
        assert after == {"observation": EPISODE_OVER, "done": True, "over": True}
        result = env.finish()
    assert result["success"] is True and result["score"] == 1.0
    assert result["steps"] == 4 and result["invalid_commands"] == 1
    assert result["ended_by"] == "done" and result["step_limit_hit"] is False
    # The step record rides back in the close reply; nothing of it touches the disk.
    lines = result["trajectory"]
    assert [line["role"] for line in lines] == ["environment"] + ["agent", "environment"] * 4
    assert lines[0]["step"] == 0 and lines[-1]["step"] == 4
    assert lines[-1]["observation"] == "You put the key in the box. Task complete."
    assert sorted(p.name for p in (tmp_path / "ep").iterdir()) == ["env_server.log", "evaluation.json"]
    evaluation = _evaluation(tmp_path)
    assert list(evaluation) == ["success", "ended_by", "steps", "invalid_commands"]
    assert evaluation == {"success": True, "ended_by": "done", "steps": 4, "invalid_commands": 1}
    assert result["evaluation"] == evaluation
    assert not env.socket_dir.exists()


def test_step_limit_ends_the_episode_without_success(tmp_path):
    with _session(tmp_path, step_limit=2) as env:
        env.reset()
        first = env.step("look")
        assert first["done"] is False
        second = env.step("look")
        assert second == {"observation": f"{fake.ROOM}\n\n{EPISODE_OVER}", "done": True, "over": False}
        result = env.finish()
    assert result["ended_by"] == "budget_exhausted" and result["step_limit_hit"] is True
    assert result["success"] is False and result["steps"] == 2
    assert _evaluation(tmp_path) == {
        "success": False, "ended_by": "budget_exhausted", "steps": 2, "invalid_commands": 0,
    }


def test_no_step_limit_leaves_the_cutoff_to_the_simulator(tmp_path):
    with _session(tmp_path, step_limit=0) as env:
        env.reset()
        for _ in range(12):
            assert env.step("look")["done"] is False
        result = env.finish()
    assert result["ended_by"] == "harness_stopped" and result["steps"] == 12 and result["step_limit"] == 0


def test_close_names_who_ended_a_session_the_environment_had_not(tmp_path):
    with _session(tmp_path) as env:
        env.reset()
        env.step("look")
        result = env.finish("agent_stopped")
    assert result["ended_by"] == "agent_stopped"
    assert _evaluation(tmp_path)["ended_by"] == "agent_stopped"
    # An unknown reason is the harness's; a reason never overrides the environment's own end.
    with _session(tmp_path) as env:
        env.reset()
        assert env.finish("whatever")["ended_by"] == "harness_stopped"
    with _session(tmp_path) as env:
        env.reset()
        env.step("take key")
        env.step("put key in box")
        assert env.finish("agent_stopped")["ended_by"] == "done"


def test_the_end_command_ends_the_episode_as_the_agents_stop_without_a_step(tmp_path):
    session = EnvironmentSession(
        factory=FACTORY, spec={}, out_dir=tmp_path / "ep", step_limit=10,
        end_command="done", python=sys.executable, startup_timeout=30.0,
    )
    with session as env:
        env.reset()
        env.step("examine table")
        env.step("take key")
        # Stripped and lower-cased; never served, never counted, recorded as an unserved agent row.
        assert env.step(" Done \n") == {"observation": EPISODE_OVER, "done": True, "over": True}
        assert env.step("look") == {"observation": EPISODE_OVER, "done": True, "over": True}
        assert env.status()["steps"] == 2 and env.status()["ended_by"] == "agent_stopped"
        result = env.finish("harness_stopped")
    assert result["ended_by"] == "agent_stopped" and result["steps"] == 2
    assert result["step_limit_hit"] is False and result["score"] == 0.5 and result["end_command"] is True
    assert [line["role"] for line in result["trajectory"]] == (
        ["environment"] + ["agent", "environment"] * 2 + ["agent"]
    )
    assert result["trajectory"][-1] == {"step": 2, "role": "agent", "command": " Done \n", "served": False}
    assert _evaluation(tmp_path) == {
        "success": False, "ended_by": "agent_stopped", "steps": 2, "invalid_commands": 0,
    }
    # Once the environment has ended the episode, `done` is answered like any other command.
    with EnvironmentSession(
        factory=FACTORY, spec={}, out_dir=tmp_path / "ep2", step_limit=1,
        end_command="done", python=sys.executable, startup_timeout=30.0,
    ) as env:
        env.reset()
        env.step("look")
        assert env.step("done") == {"observation": EPISODE_OVER, "done": True, "over": True}
        result = env.finish()
    assert result["ended_by"] == "budget_exhausted" and result["end_command"] is False


@pytest.mark.parametrize("command,ends", [
    ("impossible", True), ("Impossible: no planks in the inventory", True),
    (" impossible:x ", True), ("impossible no planks", False), ("impossibles", False),
    ("move: from [I1] to [A1] with quantity 1", False),
])
def test_the_end_command_matches_alone_or_before_a_colon(tmp_path, command, ends):
    from wikitrace.harness.environment import server as srv

    episode = srv.Episode(lambda: fake.make_env({}), tmp_path, 0, "impossible")
    episode.reset()
    episode.step(command)
    assert (episode.ended_by == "agent_stopped") is ends
    assert episode.result()["end_command"] is ends
    assert episode.steps == (0 if ends else 1)
    # The end command is recorded, never served: an agent row and no environment row.
    assert [r["role"] for r in episode.trajectory[1:]] == (["agent"] if ends else ["agent", "environment"])
    assert episode.trajectory[-1].get("served", True) is not ends


def test_a_harness_or_agent_close_is_not_an_end_command(tmp_path):
    from wikitrace.harness.environment import server as srv

    episode = srv.Episode(lambda: fake.make_env({}), tmp_path, 0, "done")
    episode.reset()
    episode.step("look")
    episode.close("agent_stopped")
    assert episode.result()["ended_by"] == "agent_stopped" and episode.result()["end_command"] is False


def test_without_the_option_done_is_a_command_like_any_other(tmp_path):
    with _session(tmp_path) as env:
        env.reset()
        assert env.step("done") == {"observation": "Nothing happens.", "done": False, "over": False}
        result = env.finish()
    assert result["steps"] == 1 and result["invalid_commands"] == 1
    assert result["ended_by"] == "harness_stopped"


# An environment that ends the task on one action, the ScienceWorld way (`failed` + done).
_FAILING_ENV = textwrap.dedent("""
    class Env:
        def reset(self):
            return "start", {"task": "t"}
        def step(self, command):
            if command == "focus on air":
                return "wrong focus", True, {"failed": True, "score": 0.2}
            return "ok", False, {"score": 0.2}
        def close(self):
            pass
        def evaluation(self):
            return {"score": 0.2}
    def make_env(spec):
        return Env()
""")


def test_an_unrecoverable_action_ends_the_episode_as_task_failed(tmp_path):
    (tmp_path / "failing.py").write_text(_FAILING_ENV)
    session = EnvironmentSession(
        factory=f"{tmp_path / 'failing.py'}:make_env", spec={}, out_dir=tmp_path / "ep",
        step_limit=10, python=sys.executable, startup_timeout=30.0,
    )
    with session as env:
        env.reset()
        assert env.step("look")["done"] is False
        # The ending step's own reply carries the marker; the step still ran (`over` is False).
        assert env.step("focus on air") == {
            "observation": f"wrong focus\n\n{EPISODE_OVER}", "done": True, "over": False}
        result = env.finish("agent_stopped")
    assert result["ended_by"] == "task_failed" and result["failed"] is True
    assert result["step_limit_hit"] is False and result["success"] is False
    assert _evaluation(tmp_path) == {
        "success": False, "score": 0.2, "ended_by": "task_failed", "steps": 2, "invalid_commands": 0,
    }


def test_a_step_before_reset_is_an_error_not_a_crash(tmp_path):
    with _session(tmp_path) as env:
        with pytest.raises(RuntimeError, match="reset first"):
            env.step("look")
        env.reset()
        assert env.status()["started"] is True


def test_exiting_without_finish_still_closes_and_evaluates(tmp_path):
    with _session(tmp_path) as env:
        env.reset()
        env.step("take key")
    result = env.finish()
    assert result["ended_by"] == "harness_stopped" and result["steps"] == 1
    assert _evaluation(tmp_path) == {
        "success": False, "ended_by": "harness_stopped", "steps": 1, "invalid_commands": 0,
    }


def test_a_rerun_in_the_same_dir_starts_from_a_clean_record(tmp_path):
    with _session(tmp_path) as env:
        env.reset()
        env.step("take key")
    with _session(tmp_path) as env:
        env.reset()
        env.step("examine table")
        result = env.finish()
    assert result["steps"] == 1
    lines = result["trajectory"]
    assert [line["step"] for line in lines] == [0, 1, 1]
    assert lines[-1]["observation"] == "On the table you see a key."
    assert _evaluation(tmp_path)["steps"] == 1


# An environment whose simulator raises on one command and reports its own move cutoff.
_FAULTY_ENV = textwrap.dedent("""
    class Env:
        def reset(self):
            return "start", {"task": "t"}
        def step(self, command):
            if command == "crash":
                raise RuntimeError("JVM gone")
            if command == "cutoff":
                return "too many moves", True, {"step_limit_hit": True}
            return "ok", False, {}
        def close(self):
            open(__file__ + ".closed", "w").close()
        def evaluation(self):
            return {"score": 0.25, "subgoals": {"achieved": 1, "total": 4}}
    def make_env(spec):
        return Env()
""")


def _faulty_session(tmp_path: Path) -> EnvironmentSession:
    (tmp_path / "faulty.py").write_text(_FAULTY_ENV)
    return EnvironmentSession(
        factory=f"{tmp_path / 'faulty.py'}:make_env", spec={}, out_dir=tmp_path / "ep",
        step_limit=10, python=sys.executable, startup_timeout=30.0,
    )


def test_a_simulator_that_raises_ends_the_episode_as_a_harness_fault(tmp_path):
    with _faulty_session(tmp_path) as env:
        env.reset()
        env.step("look")
        with pytest.raises(RuntimeError, match="JVM gone"):
            env.step("crash")
        assert env.step("look")["observation"] == EPISODE_OVER
        result = env.finish()
    assert result["ended_by"] == "simulator_error" and "RuntimeError: JVM gone" in result["env_error"]
    assert result["steps"] == 2 and result["success"] is False
    # `env_error` joins the head only here; the block's `score` sits in the head.
    evaluation = _evaluation(tmp_path)
    assert list(evaluation) == ["success", "score", "ended_by", "steps", "invalid_commands", "env_error", "subgoals"]
    assert evaluation["env_error"].startswith("RuntimeError: JVM gone")
    assert evaluation["score"] == 0.25 and evaluation["subgoals"] == {"achieved": 1, "total": 4}


def test_the_simulators_own_cutoff_is_game_time_exhausted(tmp_path):
    with _faulty_session(tmp_path) as env:
        env.reset()
        assert env.step("cutoff") == {"observation": f"too many moves\n\n{EPISODE_OVER}",
                                      "done": True, "over": False}
        result = env.finish()
        # The simulator was closed before the close reply, so its close-time work is done.
        assert (tmp_path / "faulty.py.closed").exists()
    assert result["ended_by"] == "game_time_exhausted" and result["step_limit_hit"] is True
    assert _evaluation(tmp_path) == {
        "success": False, "score": 0.25, "ended_by": "game_time_exhausted", "steps": 1,
        "invalid_commands": 0, "subgoals": {"achieved": 1, "total": 4},
    }


def test_a_server_that_cannot_start_raises(tmp_path):
    session = EnvironmentSession(
        factory=f"{tmp_path / 'missing.py'}:make_env", spec={}, out_dir=tmp_path / "ep",
        step_limit=3, python=sys.executable,
    )
    with pytest.raises(RuntimeError, match="exited during startup"):
        with session:
            pass


def test_docker_argv_mounts_the_socket_dir_and_names_the_socket(tmp_path):
    cmd = docker_run_argv(
        container_name="c", pi_cli=str(tmp_path / "pi-mono/packages/coding-agent/dist/cli.js"),
        work=tmp_path / "work", corpus_host=None, files_host=None, pi_args=["--mode", "rpc"],
        env={}, image="img", provider="openrouter", cmd_timeout=90.0,
        env_socket_host=tmp_path / "sock",
    )
    assert f"{tmp_path / 'sock'}:{ENV_SOCKET_MOUNT}:rw" in cmd
    assert f"{SOCKET_ENV_VAR}={ENV_SOCKET_MOUNT}/{SOCKET_NAME}" in cmd
    assert ("--network", "none") == tuple(cmd[cmd.index("--network"):cmd.index("--network") + 2])
    assert CITATION_ENV_VAR not in cmd
    with_citation = docker_run_argv(
        container_name="c", pi_cli=str(tmp_path / "pi-mono/packages/coding-agent/dist/cli.js"),
        work=tmp_path / "work", corpus_host=None, files_host=None, pi_args=["--mode", "rpc"],
        env={CITATION_ENV_VAR: "cite\nnow"}, image="img", provider="openrouter", cmd_timeout=90.0,
        env_socket_host=tmp_path / "sock",
    )
    # By name only: the multi-line prompt is inherited from the environment, never in argv.
    at = with_citation.index(CITATION_ENV_VAR)
    assert with_citation[at - 1] == "-e"
    assert not any("cite" in a for a in with_citation)


def test_docker_argv_without_an_environment_is_unchanged(tmp_path):
    kw = dict(
        container_name="c", pi_cli=str(tmp_path / "pi-mono/packages/coding-agent/dist/cli.js"),
        work=tmp_path / "work", corpus_host=None, files_host=None, pi_args=["--mode", "rpc"],
        env={}, image="img", provider="openrouter", cmd_timeout=90.0,
    )
    assert docker_run_argv(**kw) == docker_run_argv(**kw, env_socket_host=None)
    assert ENV_SOCKET_MOUNT not in " ".join(docker_run_argv(**kw))
    assert SOCKET_ENV_VAR not in " ".join(docker_run_argv(**kw))


@pytest.mark.parametrize("name,required", [
    ("act", '["command"]'), ("exec", '["code"]'), ("take_action", '["action", "argument"]'),
])
def test_each_extension_registers_its_tool_over_the_socket_env_var(tmp_path, name, required):
    tool = EnvTool(name=name, socket_dir=tmp_path)
    assert tool.extension.name == f"{name}.ts"
    text = tool.extension.read_text()
    assert f'name: "{name}"' in text and "registerTool" in text
    assert f"required: {required}" in text
    assert "additionalProperties: false" in text and "enum" not in text
    assert f"process.env.{SOCKET_ENV_VAR}" in text
    assert f"process.env.{CITATION_ENV_VAR}" in text
    assert 'op: "step"' in text
    # Only node built-ins: the file must load from a directory with no node_modules.
    for line in text.splitlines():
        if line.startswith("import "):
            assert 'from "node:' in line, line


_JITI = EXTENSIONS_DIR.parents[4] / "tools" / "pi-mono" / "node_modules" / "jiti" / "lib" / "jiti.mjs"

# Drives one extension through Pi's hook surface: a second call of the tool in one turn is
# blocked and never reaches the server; a new turn admits the next one.
_LOAD_EXT = """
import {{ createJiti }} from "{jiti}";
import net from "node:net";
const sock = process.argv[2];
process.env.AGENT_WIKI_ENV_SOCKET = sock;
let served = 0;
const inputs = [];
const srv = net.createServer((c) => c.on("data", (d) => {{ served += 1; inputs.push(JSON.parse(d).input); c.end('{{"observation":"ok","done":false,"over":false,"info":{{"score":0.5}}}}\\n'); }}));
await new Promise((r) => srv.listen(sock, r));
const tools = [], handlers = {{}};
const mod = await createJiti(import.meta.url).import("{ext}");
(mod.default ?? mod)({{ registerTool: (t) => tools.push(t), on: (ev, fn) => {{ handlers[ev] = fn; }} }});
const call = (toolName) => handlers.tool_call({{ toolName, toolCallId: "x", input: {{}} }}) ?? null;
handlers.turn_start({{ turnIndex: 0 }});
const results = [call("{name}"), call("{name}"), call("read")];
handlers.turn_start({{ turnIndex: 1 }});
results.push(call("{name}"));
const result = await tools[0].execute("id", {{ {args} }});
srv.close();
console.log(JSON.stringify({{ results, served, inputs, result }}));
"""


# Per tool: its call arguments, and the one step string the server must receive for them.
_EXT_CALLS = [
    ("act", 'command: "look"', "look"),
    ("exec", 'code: "look"', "look"),
    ("take_action", 'action: "search", argument: "iron_ingot"', "search: iron_ingot"),
]


@pytest.mark.skipif(not shutil.which("node") or not _JITI.exists(), reason="needs node and jiti")
@pytest.mark.parametrize("name,args,sent", _EXT_CALLS)
def test_each_extension_admits_one_call_per_turn(tmp_path, name, args, sent):
    script = tmp_path / "load_ext.mjs"
    script.write_text(_LOAD_EXT.format(
        jiti=_JITI, ext=EXTENSIONS_DIR / f"{name}.ts", name=name, args=args,
    ))
    proc = subprocess.run(
        ["node", str(script), str(tmp_path / SOCKET_NAME)], capture_output=True, text=True,
        timeout=60, check=True,
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    first, second, other, third = out["results"]
    assert first is None and other is None and third is None
    assert second["block"] is True and second["reason"].startswith(f"one {name} per turn")
    assert out["served"] == 1 and out["inputs"] == [sent]
    # The tool result is the observation text alone: Pi persists `details` into the turn
    # log, which becomes a raw step, so the reply's other fields must not ride along.
    assert out["result"] == {"content": [{"type": "text", "text": "ok"}]}


# The extension queues the citation steer itself, from inside the tool call that ends the
# episode, so Pi reads it before it collects the turn's steering messages.
_STEER_EXT = """
import {{ createJiti }} from "{jiti}";
import net from "node:net";
const sock = process.argv[2];
process.env.AGENT_WIKI_ENV_SOCKET = sock;
const replies = ['{{"observation":"ok","done":false}}', '{{"observation":"You win.\\\\n\\\\nEpisode over","done":true}}',
                 '{{"observation":"Episode over","done":true,"over":true}}'];
const srv = net.createServer((c) => c.on("data", () => c.end(replies.shift() + "\\n")));
await new Promise((r) => srv.listen(sock, r));
const tools = [], steers = [];
const mod = await createJiti(import.meta.url).import("{ext}");
(mod.default ?? mod)({{ registerTool: (t) => tools.push(t), on: () => {{}},
                     sendUserMessage: (m, o) => steers.push([m, o]) }});
const first = await tools[0].execute("id", {{ {args} }});
const second = await tools[0].execute("id", {{ {args} }});
const third = await tools[0].execute("id", {{ {args} }});
srv.close();
console.log(JSON.stringify({{ first, second, third, steers }}));
"""


@pytest.mark.skipif(not shutil.which("node") or not _JITI.exists(), reason="needs node and jiti")
@pytest.mark.parametrize("name,args,sent", _EXT_CALLS)
def test_each_extension_steers_the_citation_in_when_the_episode_ends(
    tmp_path, name, args, sent, monkeypatch
):
    monkeypatch.setenv(CITATION_ENV_VAR, "cite now")
    script = tmp_path / "steer_ext.mjs"
    script.write_text(_STEER_EXT.format(
        jiti=_JITI, ext=EXTENSIONS_DIR / f"{name}.ts", args=args,
    ))
    proc = subprocess.run(
        ["node", str(script), str(tmp_path / SOCKET_NAME)], capture_output=True, text=True,
        timeout=60, check=True,
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["first"] == {"content": [{"type": "text", "text": "ok"}]}
    assert out["second"] == {"content": [{"type": "text", "text": "You win.\n\nEpisode over"}]}
    # Queued once, though a call after the end is answered with the marker too.
    assert out["third"] == {"content": [{"type": "text", "text": "Episode over"}]}
    assert out["steers"] == [["cite now", {"deliverAs": "steer"}]]


# The citation rule through Pi's hooks: mid-episode no call is blocked by it, a sibling call
# in the ending turn still runs, and from the next turn (the steer's) every tool is blocked.
_CITATION_BLOCK_EXT = """
import {{ createJiti }} from "{jiti}";
import net from "node:net";
const sock = process.argv[2];
process.env.AGENT_WIKI_ENV_SOCKET = sock;
const replies = ['{{"observation":"ok","done":false}}', '{{"observation":"You win.\\\\n\\\\nEpisode over","done":true}}'];
const srv = net.createServer((c) => c.on("data", () => c.end(replies.shift() + "\\n")));
await new Promise((r) => srv.listen(sock, r));
const tools = [], handlers = {{}}, steers = [];
const mod = await createJiti(import.meta.url).import("{ext}");
(mod.default ?? mod)({{ registerTool: (t) => tools.push(t), on: (ev, fn) => {{ handlers[ev] = fn; }},
                     sendUserMessage: (m, o) => steers.push([m, o]) }});
const call = (toolName) => handlers.tool_call({{ toolName, toolCallId: "x", input: {{}} }}) ?? null;
const mid = [];
handlers.turn_start({{ turnIndex: 0 }});
mid.push(call("read"), call("bash"), call("{name}"));
await tools[0].execute("id", {{ {args} }});
handlers.turn_start({{ turnIndex: 1 }});
mid.push(call("read"), call("{name}"));
await tools[0].execute("id", {{ {args} }});
const ending = [call("read"), call("bash")];
handlers.turn_start({{ turnIndex: 2 }});
const after = [call("read"), call("bash"), call("{name}")];
srv.close();
console.log(JSON.stringify({{ mid, ending, after, steers }}));
"""


@pytest.mark.skipif(not shutil.which("node") or not _JITI.exists(), reason="needs node and jiti")
@pytest.mark.parametrize("name,args,sent", _EXT_CALLS)
def test_each_extension_blocks_every_tool_once_the_citation_steer_is_out(
    tmp_path, name, args, sent, monkeypatch
):
    monkeypatch.setenv(CITATION_ENV_VAR, "cite now")
    script = tmp_path / "block_ext.mjs"
    script.write_text(_CITATION_BLOCK_EXT.format(
        jiti=_JITI, ext=EXTENSIONS_DIR / f"{name}.ts", name=name, args=args,
    ))
    proc = subprocess.run(
        ["node", str(script), str(tmp_path / SOCKET_NAME)], capture_output=True, text=True,
        timeout=60, check=True,
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["mid"] == [None] * 5 and out["ending"] == [None, None]
    assert out["steers"] == [["cite now", {"deliverAs": "steer"}]]
    blocked = {"block": True, "reason": "The episode is over. Answer with the JSON object only."}
    assert out["after"] == [blocked] * 3


def test_fake_runner_forwards_the_env_tool_to_its_handler(tmp_path):
    seen = {}

    def handler(**kw):
        seen.update(kw)
        return TerminalResult(sentinel={"ok": 1})

    tool = EnvTool(name="act", socket_dir=tmp_path)
    FakeTraverseRunner(handler).run_session(
        system_prompt="s", user_msg="u", corpus_dir=None, out_dir=tmp_path / "out", env_tool=tool,
    )
    assert seen["env_tool"] is tool


def _tool_end(text: str, is_error: bool = False) -> dict:
    return {"type": "tool_execution_end", "toolName": "act", "isError": is_error,
            "result": {"content": [{"type": "text", "text": text}]}}


def test_episode_over_is_matched_on_the_marker_paragraph():
    assert _is_episode_over(_tool_end(EPISODE_OVER))
    assert _is_episode_over(_tool_end(f"You put the key in the box.\n\n{EPISODE_OVER}"))
    assert not _is_episode_over(_tool_end("You take the key from the table."))
    assert not _is_episode_over(_tool_end(f"{EPISODE_OVER}\nextra"))
    # Another tool's output ending in the same words is not the environment speaking.
    assert not _is_episode_over({**_tool_end(EPISODE_OVER), "toolName": "bash"})
    assert _is_episode_over({**_tool_end(EPISODE_OVER), "toolName": "exec"}, "exec")
    assert not _is_episode_over(_tool_end(EPISODE_OVER, is_error=True))
    assert not _is_episode_over({"type": "tool_execution_end", "result": "Episode over"})


# A stand-in for Pi's RPC loop: after the tool reports the episode over it waits for the
# runner's reply, recording what arrived, and ends the session without another model call.
_FAKE_PI = textwrap.dedent("""
    import json, os, sys
    def emit(ev):
        sys.stdout.write(json.dumps(ev) + "\\n"); sys.stdout.flush()
    sys.stdin.readline()  # the prompt
    emit({"type": "turn_start"})
    emit({"type": "tool_execution_start", "toolCallId": "c1", "toolName": "act",
          "args": {"command": "put key in box"}})
    emit({"type": "tool_execution_end", "toolName": "act", "isError": False,
          "result": {"content": [{"type": "text", "text": "You put the key in the box.\\n\\nEpisode over"}]}})
    got = json.loads(sys.stdin.readline())
    ext = sys.argv[sys.argv.index("--extension") + 1]
    with open(os.environ["FAKE_PI_RECORD"], "w") as fh:
        json.dump({"got": got, "socket": os.environ.get("AGENT_WIKI_ENV_SOCKET"),
                   "extension": ext, "extension_exists": os.path.exists(ext)}, fh)
    emit({"type": "agent_end"})
""")


def test_pi_runner_aborts_as_soon_as_the_episode_is_over(tmp_path, monkeypatch):
    fake_pi = tmp_path / "fake_pi.py"
    fake_pi.write_text(_FAKE_PI, encoding="utf-8")
    record = tmp_path / "record.json"
    monkeypatch.setenv("FAKE_PI_RECORD", str(record))
    with pytest.warns(RuntimeWarning, match="runs Pi directly on the host"):
        runner = PiTraverseRunner(
            pi_cli=str(fake_pi), model="m", node_bin=sys.executable, sandbox="none",
            idle_timeout=5.0, max_seconds=20.0,
        )
    tool = EnvTool(name="act", socket_dir=tmp_path / "sock")
    result = runner.run_session(
        system_prompt="s", user_msg="u", corpus_dir=None, out_dir=tmp_path / "out",
        max_turns=50, env_tool=tool,
    )
    seen = json.loads(record.read_text())
    assert seen["got"] == {"id": "py-abort", "type": "abort"}
    assert seen["socket"] == str((tmp_path / "sock").resolve() / SOCKET_NAME)
    assert seen["extension"].endswith("/_ext/act.ts") and seen["extension_exists"]
    assert result.stopped_by == ""
    assert [t["type"] for t in result.turns] == ["function_call", "episode_over"]


# The same loop with the citation: the extension queues the prompt from the environment,
# Pi delivers it as its own user message at the end of the turn, and the runner aborts
# after the agent's one JSON answer. Pi's real message order is kept.
_FAKE_PI_CITATION = textwrap.dedent("""
    import json, os, sys
    def emit(ev):
        sys.stdout.write(json.dumps(ev) + "\\n"); sys.stdout.flush()
    sys.stdin.readline()  # the prompt
    emit({"type": "turn_start"})
    emit({"type": "provider_request_context", "messages": [{"role": "user", "content": "episode"}]})
    emit({"type": "tool_execution_start", "toolCallId": "c1", "toolName": "act",
          "args": {"command": "put key in box"}})
    emit({"type": "tool_execution_end", "toolName": "act", "isError": False,
          "result": {"content": [{"type": "text", "text": "You put the key in the box.\\n\\nEpisode over"}]}})
    steer = os.environ["AGENT_WIKI_CITATION_PROMPT"]
    with open(os.environ["FAKE_PI_RECORD"], "w") as fh:
        json.dump({"message": steer}, fh)
    emit({"type": "message_start", "message": {"role": "toolResult", "toolCallId": "c1"}})
    emit({"type": "message_end", "message": {"role": "toolResult", "toolCallId": "c1"}})
    emit({"type": "message_start", "message": {"role": "user", "content": steer}})
    emit({"type": "message_end", "message": {"role": "user", "content": steer}})
    emit({"type": "turn_start"})
    emit({"type": "provider_request_context", "messages": [{"role": "user", "content": "citation"}]})
    emit({"type": "message_update", "assistantMessageEvent":
          {"type": "text_delta", "delta": '{"K": ["document-0007"], "R": ["step-0023"]}'}})
    emit({"type": "message_end", "message": {"role": "assistant", "stopReason": "end_turn", "content": []}})
    json.loads(sys.stdin.readline())  # the abort
    emit({"type": "agent_end"})
""")

# Two tool calls in the ending assistant message: the sibling's start must not fire the
# re-armed abort while the extension's steer is still waiting for the end of the turn, so
# the first line the runner writes after the episode is the abort, after the answer.
_FAKE_PI_CITATION_SIBLING = textwrap.dedent("""
    import json, os, sys
    def emit(ev):
        sys.stdout.write(json.dumps(ev) + "\\n"); sys.stdout.flush()
    sys.stdin.readline()  # the prompt
    emit({"type": "turn_start"})
    emit({"type": "tool_execution_start", "toolCallId": "c1", "toolName": "act",
          "args": {"command": "put key in box"}})
    emit({"type": "tool_execution_end", "toolName": "act", "isError": False,
          "result": {"content": [{"type": "text", "text": "You put the key in the box.\\n\\nEpisode over"}]}})
    emit({"type": "tool_execution_start", "toolCallId": "c2", "toolName": "act",
          "args": {"command": "look"}})
    emit({"type": "tool_execution_end", "toolName": "act", "isError": False,
          "result": {"content": [{"type": "text", "text": "Episode over"}]}})
    steer = os.environ["AGENT_WIKI_CITATION_PROMPT"]
    emit({"type": "message_start", "message": {"role": "user", "content": steer}})
    emit({"type": "message_end", "message": {"role": "user", "content": steer}})
    emit({"type": "turn_start"})
    emit({"type": "provider_request_context", "messages": []})
    emit({"type": "message_update", "assistantMessageEvent":
          {"type": "text_delta", "delta": '{"R": ["step-0001"]}'}})
    emit({"type": "message_end", "message": {"role": "assistant", "stopReason": "end_turn",
                                             "responseId": "gen-cite", "content": []}})
    first = json.loads(sys.stdin.readline())
    with open(os.environ["FAKE_PI_RECORD"], "w") as fh:
        json.dump({"first": first["type"]}, fh)
    emit({"type": "agent_end"})
""")


def test_a_sibling_tool_call_after_the_ending_one_does_not_cut_the_citation_off(
    tmp_path, monkeypatch
):
    fake_pi = tmp_path / "fake_pi_sibling.py"
    fake_pi.write_text(_FAKE_PI_CITATION_SIBLING, encoding="utf-8")
    record = tmp_path / "rec.json"
    monkeypatch.setenv("FAKE_PI_RECORD", str(record))
    with pytest.warns(RuntimeWarning, match="runs Pi directly on the host"):
        runner = PiTraverseRunner(
            pi_cli=str(fake_pi), model="m", node_bin=sys.executable, sandbox="none",
            idle_timeout=5.0, max_seconds=20.0,
        )
    tool = EnvTool(name="act", socket_dir=tmp_path / "sock")
    result = runner.run_session(
        system_prompt="s", user_msg="u", corpus_dir=None, out_dir=tmp_path / "out",
        max_turns=50, env_tool=tool,
        citation_prompt=citation_message(("R",)), citation_layers=("R",),
    )
    assert json.loads(record.read_text()) == {"first": "abort"}
    assert [t["type"] for t in result.turns] == [
        "function_call", "episode_over", "function_call", "episode_over", "citation",
    ]
    assert result.citation == {"R": ["step-0001"]}
    assert result.turns[-1]["response_id"] == "gen-cite"


def test_the_citation_turn_is_steered_in_after_the_episode_and_then_aborted(
    tmp_path, monkeypatch
):
    """The citation answer becomes its own turn, and the last request context is kept."""
    fake_pi = tmp_path / "fake_pi_citation.py"
    fake_pi.write_text(_FAKE_PI_CITATION, encoding="utf-8")
    record = tmp_path / "steer.json"
    monkeypatch.setenv("FAKE_PI_RECORD", str(record))
    with pytest.warns(RuntimeWarning, match="runs Pi directly on the host"):
        runner = PiTraverseRunner(
            pi_cli=str(fake_pi), model="m", node_bin=sys.executable, sandbox="none",
            idle_timeout=5.0, max_seconds=20.0,
        )
    tool = EnvTool(name="act", socket_dir=tmp_path / "sock")
    result = runner.run_session(
        system_prompt="s", user_msg="u", corpus_dir=None, out_dir=tmp_path / "out",
        max_turns=50, env_tool=tool,
        citation_prompt=citation_message(("R", "K")), citation_layers=("R", "K"),
    )
    steer = json.loads(record.read_text())
    assert steer["message"] == citation_message(("R", "K"))
    assert "Do not call any tool" in steer["message"]
    assert [t["type"] for t in result.turns] == [
        "function_call", "episode_over", "citation",
    ]
    assert result.citation == {"R": ["step-0023"], "K": ["document-0007"]}
    # The answer text is the citation's, not the episode's, and the persisted context is
    # the citation request's, the only one holding the ending action's result.
    assert result.final_text == ""
    assert result.final_context == [{"role": "user", "content": "citation"}]


# The citation turn answered with a tool call first: the extension blocks it, the runner
# keeps the citation open, skips the blocked call, and closes on the next text answer.
_FAKE_PI_CITATION_RETRY = textwrap.dedent("""
    import json, os, sys
    def emit(ev):
        sys.stdout.write(json.dumps(ev) + "\\n"); sys.stdout.flush()
    misses = int(os.environ["FAKE_PI_MISSES"])
    sys.stdin.readline()  # the prompt
    emit({"type": "turn_start"})
    emit({"type": "provider_request_context", "messages": [{"role": "user", "content": "episode"}]})
    emit({"type": "tool_execution_start", "toolCallId": "c1", "toolName": "act",
          "args": {"command": "put key in box"}})
    emit({"type": "tool_execution_end", "toolName": "act", "isError": False,
          "result": {"content": [{"type": "text", "text": "Episode over"}]}})
    steer = os.environ["AGENT_WIKI_CITATION_PROMPT"]
    emit({"type": "message_end", "message": {"role": "user", "content": steer}})
    for i in range(misses):
        emit({"type": "turn_start"})
        emit({"type": "provider_request_context", "messages": [{"role": "user", "content": f"after {i}"}]})
        emit({"type": "message_end", "message": {"role": "assistant", "stopReason": "toolUse",
                                                 "responseId": f"gen-miss-{i}", "content": []}})
        emit({"type": "tool_execution_start", "toolCallId": f"b{i}", "toolName": "read",
              "args": {"path": "/work/corpus/raw/step-0001/metadata.md"}})
        emit({"type": "tool_execution_end", "toolName": "read", "isError": True,
              "result": {"content": [{"type": "text", "text": "The episode is over."}]}})
    if misses < 3:
        emit({"type": "turn_start"})
        emit({"type": "message_update", "assistantMessageEvent":
              {"type": "text_delta", "delta": '{"R": ["step-0001"]}'}})
        emit({"type": "message_end", "message": {"role": "assistant", "stopReason": "stop",
                                                 "responseId": "gen-cite", "content": []}})
    first = json.loads(sys.stdin.readline())
    with open(os.environ["FAKE_PI_RECORD"], "w") as fh:
        json.dump({"first": first["type"]}, fh)
    emit({"type": "agent_end"})
""")


@pytest.mark.parametrize("misses", [1, 2, 3])
def test_a_citation_answered_by_a_tool_call_stays_open_for_two_more_turns(
    tmp_path, monkeypatch, misses
):
    fake_pi = tmp_path / "fake_pi_retry.py"
    fake_pi.write_text(_FAKE_PI_CITATION_RETRY, encoding="utf-8")
    record = tmp_path / "rec.json"
    monkeypatch.setenv("FAKE_PI_RECORD", str(record))
    monkeypatch.setenv("FAKE_PI_MISSES", str(misses))
    with pytest.warns(RuntimeWarning, match="runs Pi directly on the host"):
        runner = PiTraverseRunner(
            pi_cli=str(fake_pi), model="m", node_bin=sys.executable, sandbox="none",
            idle_timeout=5.0, max_seconds=20.0,
        )
    result = runner.run_session(
        system_prompt="s", user_msg="u", corpus_dir=tmp_path / "corpus", out_dir=tmp_path / "out",
        max_turns=50, env_tool=EnvTool(name="act", socket_dir=tmp_path / "sock"),
        citation_prompt=citation_message(("R",)), citation_layers=("R",),
    )
    assert json.loads(record.read_text()) == {"first": "abort"}
    # The blocked calls are no action and no read, and the kept context is the citation request's.
    assert [t["type"] for t in result.turns] == ["function_call", "episode_over", "citation"]
    assert result.read_set == [] and result.final_context == [{"role": "user", "content": "after 0"}]
    cite = result.turns[-1]
    if misses < 3:
        assert result.citation == {"R": ["step-0001"]} and cite["response_id"] == "gen-cite"
        assert cite["missed_response_ids"] == [f"gen-miss-{i}" for i in range(misses)]
    else:
        # The bound: the third textless answer closes the citation empty, as before.
        assert cite["text"] == "" and cite["response_id"] == "gen-miss-2"
        assert cite["missed_response_ids"] == ["gen-miss-0", "gen-miss-1"]


# A short-lived parent: starts a server the way the client does, then dies without closing it.
_ORPHANING_PARENT = textwrap.dedent("""
    import subprocess, sys, time
    from pathlib import Path
    factory, sock, out = sys.argv[1:4]
    proc = subprocess.Popen([sys.executable, {server!r}, "--factory", factory,
                             "--socket", sock, "--out", out, "--step-limit", "3"],
                            stdin=subprocess.PIPE, start_new_session=True)
    proc.stdin.write(b"{{}}")
    proc.stdin.close()
    while not Path(sock).exists():
        time.sleep(0.05)
    print(proc.pid, flush=True)
    time.sleep(60)
""")


def test_a_server_exits_and_evaluates_when_its_parent_dies(tmp_path):
    from wikitrace.harness.environment.client import SERVER_PATH
    from wikitrace.harness.environment.protocol import request

    parent_py = tmp_path / "parent.py"
    parent_py.write_text(_ORPHANING_PARENT.format(server=str(SERVER_PATH)))
    sock = str(tmp_path / SOCKET_NAME)
    parent = subprocess.Popen(
        [sys.executable, str(parent_py), FACTORY, sock, str(tmp_path / "ep")],
        stdout=subprocess.PIPE, text=True,
    )
    server_pid = int(parent.stdout.readline())
    assert request(sock, {"op": "reset"})["info"]["success"] is False
    parent.kill()
    parent.wait()
    deadline = time.monotonic() + 10
    while os.path.exists(f"/proc/{server_pid}") and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not os.path.exists(f"/proc/{server_pid}"), "server outlived its parent"
    assert not os.path.exists(sock)
    assert _evaluation(tmp_path)["ended_by"] == "harness_stopped"


def test_the_ppid_poll_ends_the_episode_when_the_death_signal_did_not(tmp_path, monkeypatch):
    from wikitrace.harness.environment import server as srv

    monkeypatch.setattr(srv.os, "getppid", lambda: 1)
    episode = srv.Episode(lambda: fake.make_env({}), tmp_path, 0)
    with pytest.raises(srv.ParentGone):
        srv.serve(episode, str(tmp_path / SOCKET_NAME), parent_pid=4242, poll_interval=0.05)


# The citation turn disobeys: its answer is cut at the output budget and it calls a corpus
# read and the environment tool. Neither call is part of the episode, and the solver's stop
# statistics stay the ones its own last message left.
_FAKE_PI_CITATION_TOOLS = textwrap.dedent("""
    import json, os, sys
    def emit(ev):
        sys.stdout.write(json.dumps(ev) + "\\n"); sys.stdout.flush()
    sys.stdin.readline()  # the prompt
    emit({"type": "turn_start"})
    emit({"type": "provider_request_context", "messages": [{"role": "user", "content": "episode"}]})
    emit({"type": "message_end", "message": {"role": "assistant", "stopReason": "toolUse",
          "content": [{"type": "toolCall", "id": "c1", "name": "act"}]}})
    emit({"type": "tool_execution_start", "toolCallId": "c1", "toolName": "act",
          "args": {"command": "put key in box"}})
    emit({"type": "tool_execution_end", "toolName": "act", "isError": False,
          "result": {"content": [{"type": "text", "text": "You put the key in the box.\\n\\nEpisode over"}]}})
    steer = os.environ["AGENT_WIKI_CITATION_PROMPT"]
    emit({"type": "message_end", "message": {"role": "toolResult", "toolCallId": "c1"}})
    emit({"type": "message_end", "message": {"role": "user", "content": steer}})
    emit({"type": "turn_start"})
    emit({"type": "provider_request_context", "messages": [{"role": "user", "content": "citation"}]})
    emit({"type": "message_update", "assistantMessageEvent":
          {"type": "text_delta", "delta": '{"K": ["document-0007"]}'}})
    emit({"type": "message_end", "message": {"role": "assistant", "stopReason": "length",
          "responseId": "gen-cite",
          "content": [{"type": "toolCall", "id": "c2", "name": "read"},
                      {"type": "toolCall", "id": "c3", "name": "act"}]}})
    emit({"type": "tool_execution_start", "toolCallId": "c2", "toolName": "read",
          "args": {"path": "/work/corpus/knowledge/document-0007.md"}})
    emit({"type": "tool_execution_end", "toolName": "read", "isError": False,
          "result": {"content": [{"type": "text", "text": "a document"}]}})
    emit({"type": "tool_execution_start", "toolCallId": "c3", "toolName": "act",
          "args": {"command": "look"}})
    emit({"type": "tool_execution_end", "toolName": "act", "isError": False,
          "result": {"content": [{"type": "text", "text": "Episode over"}]}})
    json.loads(sys.stdin.readline())  # the abort
    emit({"type": "agent_end"})
""")


def test_a_tool_the_citation_turn_runs_is_neither_an_action_nor_a_read(tmp_path):
    """The episode ends at its ending step; the citation turn adds only its answer."""
    fake_pi = tmp_path / "fake_pi_citation_tools.py"
    fake_pi.write_text(_FAKE_PI_CITATION_TOOLS, encoding="utf-8")
    with pytest.warns(RuntimeWarning, match="runs Pi directly on the host"):
        runner = PiTraverseRunner(
            pi_cli=str(fake_pi), model="m", node_bin=sys.executable, sandbox="none",
            idle_timeout=5.0, max_seconds=20.0,
        )
    tool = EnvTool(name="act", socket_dir=tmp_path / "sock")
    result = runner.run_session(
        system_prompt="s", user_msg="u", corpus_dir=None, out_dir=tmp_path / "out",
        max_turns=50, env_tool=tool,
        citation_prompt=citation_message(("K",)), citation_layers=("K",),
    )
    assert [t["type"] for t in result.turns] == ["function_call", "episode_over", "citation"]
    assert result.citation == {"K": ["document-0007"]}
    assert not any("document-0007" in p for p in result.read_set)
    assert result.last_stop_reason == "toolUse"
    assert result.last_had_tool_call is True
    assert result.truncated_calls == 0


def test_a_host_server_is_the_interpreter_running_the_server_script(tmp_path):
    from wikitrace.harness.environment.client import SERVER_PATH

    env = _session(tmp_path, step_limit=7)
    assert env.argv() == [
        sys.executable, str(SERVER_PATH), "--factory", FACTORY,
        "--socket", str(env.socket_path), "--out", str(tmp_path / "ep"),
        "--step-limit", "7", "--end-command", "",
    ]


def test_a_container_server_gets_only_its_mounts_and_its_named_environment(tmp_path, monkeypatch):
    from wikitrace.harness.environment.client import CONTAINER_PREFIX, Container

    monkeypatch.setenv("AGENT_WIKI_TEST_SECRET", "leak")
    env = EnvironmentSession(
        factory="/srv/envs/sim.py:make_env", spec={}, out_dir=tmp_path / "ep", step_limit=5,
        container=Container(image="img:1", mounts=(("/host/a", "/a", True), ("/host/b", "/b", False)),
                            env=(("SIM_ROOT", "/a"),)),
    )
    name = env.container.name
    assert name.startswith(f"{CONTAINER_PREFIX}{os.getpid()}-")
    assert env.argv() == [
        "docker", "run", "--rm", "-i", "--pull", "never", "--name", name,
        "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "SIM_ROOT=/a",
        "--mount", f"type=bind,source={env.socket_dir},target=/env",
        "--mount", "type=bind,source=/host/a,target=/a,readonly",
        "--mount", "type=bind,source=/host/b,target=/b",
        "img:1", "python", "/srv/server.py", "--factory", "/srv/envs/sim.py:make_env",
        "--socket", f"/env/{SOCKET_NAME}", "--out", "/tmp/out",
        "--step-limit", "5", "--end-command", "", "--stdin-lifeline",
    ]
    assert "leak" not in " ".join(env.argv()) and "--env-file" not in env.argv()


def test_each_container_session_has_its_own_name():
    from wikitrace.harness.environment.client import Container

    assert Container(image="i").name != Container(image="i").name


def test_killing_a_container_session_removes_the_container(tmp_path, monkeypatch):
    from wikitrace.harness.environment import client
    from wikitrace.harness.environment.client import Container

    calls = []
    monkeypatch.setattr(client.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    env = EnvironmentSession(factory="f", spec={}, out_dir=tmp_path, container=Container(image="i"))
    env._kill()
    assert calls == [["docker", "rm", "-f", env.container.name]]


def test_a_host_session_never_calls_docker(tmp_path, monkeypatch):
    from wikitrace.harness.environment import client

    calls = []
    real_run = client.subprocess.run
    monkeypatch.setattr(client.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or real_run(cmd, **kw))
    with _session(tmp_path) as env:
        env.reset()
        env.finish()
    assert calls == []


def test_the_stdin_lifeline_ends_the_episode_when_stdin_closes(tmp_path):
    from wikitrace.harness.environment.client import SERVER_PATH
    from wikitrace.harness.environment.protocol import request

    sock = str(tmp_path / SOCKET_NAME)
    server = subprocess.Popen(
        [sys.executable, str(SERVER_PATH), "--factory", FACTORY, "--socket", sock,
         "--out", str(tmp_path / "ep"), "--stdin-lifeline"],
        stdin=subprocess.PIPE,
    )
    server.stdin.write(json.dumps({"task": "put the key in the box"}).encode() + b"\n")
    server.stdin.flush()
    deadline = time.monotonic() + 30
    while not os.path.exists(sock) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert request(sock, {"op": "reset"})["info"]["task"] == "put the key in the box"
    assert request(sock, {"op": "step", "input": "take key"})["done"] is False
    server.stdin.close()
    assert server.wait(timeout=10) == 0
    assert not os.path.exists(sock)
    assert _evaluation(tmp_path) == {"success": False, "ended_by": "harness_stopped", "steps": 1,
                                     "invalid_commands": 0}


def test_a_container_session_that_fails_to_start_leaves_nothing_behind(tmp_path, monkeypatch):
    from wikitrace.harness.environment import client
    from wikitrace.harness.environment.client import Container

    calls = []
    monkeypatch.setattr(client.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    env = EnvironmentSession(factory="f", spec={}, out_dir=tmp_path, container=Container(image="i"))
    monkeypatch.setattr(env, "argv", lambda: [sys.executable, "-c", "raise SystemExit(3)"])
    with pytest.raises(RuntimeError, match="exited during startup"):
        env.__enter__()
    assert calls == [["docker", "rm", "-f", env.container.name]]
    assert env._proc.stdin.closed and env._log.closed and not env.socket_dir.exists()


@pytest.mark.parametrize("error", [subprocess.TimeoutExpired("docker", 60), OSError("no docker")])
def test_a_failing_docker_rm_does_not_stop_a_kill(tmp_path, monkeypatch, error):
    from wikitrace.harness.environment import client
    from wikitrace.harness.environment.client import Container

    def run(cmd, **kw):
        raise error

    monkeypatch.setattr(client.subprocess, "run", run)
    EnvironmentSession(factory="f", spec={}, out_dir=tmp_path, container=Container(image="i"))._kill()


class _Chat:
    """An environment with a user: a message is answered by the next scripted reply."""

    def __init__(self, replies, ask_ended_by: bool = True):
        self.replies = list(replies)
        self.ended = None
        if not ask_ended_by:
            self.evaluation = lambda: {"asked": False}

    def reset(self):
        return "hi", {"source": "user"}

    def step(self, command):
        return "tool output", False, {"source": "env"}

    def message(self, text):
        reply = self.replies.pop(0)
        return reply, reply.endswith("STOP"), {"source": "user"}

    def close(self):
        pass

    def evaluation(self, ended_by: str = ""):
        self.ended = ended_by
        return {"asked": True, "ended_by_seen": ended_by}


def test_a_message_is_answered_by_the_user_and_is_not_a_step(tmp_path: Path) -> None:
    from wikitrace.harness.environment.server import Episode

    ep = Episode(lambda: _Chat(["sure", "bye STOP"]), tmp_path)
    ep.reset()
    assert ep.message("Hello?") == {"observation": "sure", "done": False, "over": False}
    assert ep.step("look")["observation"] == "tool output"
    assert ep.message("Anything else?") == {"observation": "bye STOP", "done": True, "over": False}
    assert ep.message("never sent") == {"observation": EPISODE_OVER, "done": True, "over": True}
    assert ep.steps == 1 and ep.ended_by == "done"
    assert [(r["role"], r.get("message"), r.get("source")) for r in ep.trajectory] == [
        ("environment", None, "user"), ("message", "Hello?", "user"), ("agent", None, None),
        ("environment", None, "env"), ("message", "Anything else?", "user")]


def test_an_environment_without_a_user_refuses_a_message_and_keeps_the_episode(tmp_path: Path) -> None:
    from wikitrace.harness.environment.server import Episode

    ep = Episode(lambda: fake.make_env({}), tmp_path)
    ep.reset()
    assert ep.message("hello") == {"error": "this environment takes no messages"} and ep.ended_by == ""


@pytest.mark.parametrize("asks", [True, False])
def test_the_close_reason_reaches_an_evaluation_that_names_it(tmp_path: Path, asks: bool) -> None:
    from wikitrace.harness.environment.server import Episode

    ep = Episode(lambda: _Chat([], ask_ended_by=asks), tmp_path)
    ep.reset()
    ep.close("harness_stopped")
    ev = json.loads((tmp_path / "evaluation.json").read_text())
    assert ev["ended_by"] == "harness_stopped" and ev["asked"] is asks
    assert ev.get("ended_by_seen") == ("harness_stopped" if asks else None)
