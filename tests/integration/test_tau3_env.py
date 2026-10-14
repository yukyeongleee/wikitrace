"""The tau3 environment under its own venv: the authors' code answers, counts and scores as their loop does."""
# No model call: the user simulator and the judge replay scripts from the spec.

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO / "data" / "tau3" / ".venv" / "bin" / "python"
ENV = REPO / "src" / "wikitrace" / "envs" / "tau3" / "envs" / "tau3.py"

pytestmark = pytest.mark.skipif(not PYTHON.exists(), reason=f"{PYTHON} is absent: run scripts/setup/tau3.sh")

STOP, TRANSFER, OOS = "###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"


def _run(script: str, *args: str, timeout: int = 600) -> dict:
    """Run a script under the tau3 venv with the env module path first; the last stdout line is JSON."""
    proc = subprocess.run([str(PYTHON), "-c", script, str(ENV), *args], capture_output=True, text=True,
                          timeout=timeout, check=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


_HEAD = r"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("tau3env", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
def call(action, argument):
    return json.dumps({"action": action, "argument": argument})
"""


# Task 10's gold actions with a scripted user, through `make_env` and the authors' evaluator.
_GOLD = _HEAD + r"""
task = m.load_task("10")
env = m.make_env({"task_id": "10", "user_script": ["I want to return everything I ordered.", "sure", "yes", "###STOP###"]})
first, info = env.reset()
out = {"first": first, "reset_info": info, "steps": []}
for a in task.evaluation_criteria.actions:
    obs, done, info = env.step(call(a.name, a.arguments))
    out["steps"].append([obs, done, info])
for text in ("Anything else?", "Goodbye", "Bye"):
    obs, done, info = env.message(text)
    out["steps"].append([obs, done, info])
env.close()
out["evaluation"] = env.evaluation()
out["messages"] = [[type(x).__name__, x.content, [t.name for t in (x.tool_calls or [])] if hasattr(x, "tool_calls") else None] for x in env.messages]
print(json.dumps(out))
"""


def test_gold_replay_of_task_10_scores_reward_one():
    out = _run(_GOLD)
    assert out["first"] == "I want to return everything I ordered."
    info = out["reset_info"]
    assert (info["task"], info["source"], info["orchestrator_steps"], info["messages"]) == (out["first"], "user", 1, 0)
    obs_tool = [s for s in out["steps"][:5]]
    assert all(i["source"] == "env" and not i["invalid_command"] and not d for _, d, i in obs_tool)
    # The authors' `to_json_str` passes a string result through as it is.
    assert obs_tool[0][0] == "mia_garcia_4516" and json.loads(obs_tool[1][0])["user_id"] == "mia_garcia_4516"
    assert [i["orchestrator_steps"] for _, _, i in out["steps"]] == [3, 5, 7, 9, 11, 13, 15, 17]
    replies = out["steps"][5:]
    assert [o for o, _, _ in replies] == ["sure", "yes", STOP]
    assert [d for _, d, _ in replies] == [False, False, True]
    assert replies[-1][2]["stop_token"] == STOP and replies[-1][2]["termination_reason"] == "user_stop"
    ev = out["evaluation"]
    assert ev["success"] is True and ev["reward"] == 1.0 and ev["score"] == 1.0
    assert ev["reward_basis"] == ["DB", "NL_ASSERTION"] and ev["reward_breakdown"] == {"DB": 1.0, "NL_ASSERTION": 1.0}
    assert ev["db_check"] == {"db_match": True, "db_reward": 1.0}
    assert ev["nl_assertions"] == [] and ev["judge_attempts"] == 0
    assert (ev["orchestrator_steps"], ev["messages"], ev["num_errors"]) == (17, 3, 0)
    assert ev["termination_reason"] == "user_stop" and ev["stop_token"] == STOP
    assert ev["calls"] == {"user_simulator": 4, "judge": 0} and ev["cost"] == {"user_simulator": 0.0, "judge": 0.0}
    assert ev["scored"] is True and ev["judge_failed"] is False and ev["judge_error"] == ""
    # One row per simulator call; a scripted reply carries no usage, provider or generation id.
    rows = ev["llm_calls"]
    assert [r["role"] for r in rows] == ["user_simulator"] * 4 and all(r["model_requested"] == "scripted" for r in rows)
    assert all((r["cost"], r["generation_id"], r["provider"], r["prompt_tokens"]) == (0.0, "", "", 0) for r in rows)
    for key in ("action_checks", "communicate_checks", "env_assertions", "info"):
        assert key not in ev
    # The authors' message list: the greeting, then user, tool call + tool result pairs, replies.
    kinds = [k for k, _, _ in out["messages"]]
    assert kinds[:2] == ["AssistantMessage", "UserMessage"]
    assert kinds.count("ToolMessage") == 5 and kinds[-1] == "UserMessage"
    assert out["messages"][0][1] == "Hi! How can I help you today?"


# Each rejection, ours and the authors', with the counters after it.
_ERRORS = _HEAD + r"""
env = m.make_env({"task_id": "10", "user_script": ["hello", "###STOP###"]})
env.reset()
out = []
for action, argument in [("nope", {}), ("get_user_details", {}), ("get_user_details", {"user_id": "x", "k": 1}),
                         ("get_user_details", {"user_id": 5}), ("get_user_details", "x"), ("respond", {"message": "hi"}),
                         ("respond", " \n"), ("calculate", {"expression": "2+2"})]:
    obs, done, info = env.step(call(action, argument))
    out.append([obs, done, info["orchestrator_steps"], info["num_errors"], info["invalid_command"], info["source"]])
obs, done, info = env.step("not json")
out.append([obs, done, info["orchestrator_steps"], info["num_errors"], info["invalid_command"], info["source"]])
print(json.dumps({"out": out, "messages": len(env.messages)}))
"""


def test_each_error_message_is_verbatim_and_counts_once():
    out = _run(_ERRORS)
    rows = out["out"]
    assert [r[0] for r in rows] == [
        "Error: Tool 'nope' not found.",
        "Error: RetailTools.get_user_details() missing 1 required positional argument: 'user_id'",
        "Error: RetailTools.get_user_details() got an unexpected keyword argument 'k'",
        "Error: `get_user_details` parameter `user_id` must be string, got integer.",
        "Error: `get_user_details` takes its parameters as a JSON object, got string.",
        # `respond` is no tool: the authors' unknown-tool message, whatever the argument.
        "Error: Tool 'respond' not found.",
        "Error: Tool 'respond' not found.",
        "4.0",
        "Error: take_action expects a JSON object with `action` and `argument`.",
    ]
    assert [r[3] for r in rows] == [1, 2, 3, 4, 5, 6, 7, 7, 8]
    assert [r[4] for r in rows] == [False, False, False, True, True, False, False, False, True]
    assert [r[2] for r in rows] == [3, 5, 7, 9, 11, 13, 15, 17, 19]
    assert all(r[5] == "env" and r[1] is False for r in rows)
    # The authors' errors are in their message list (5 call/result pairs + calculate); ours are not.
    assert out["messages"] == 2 + 2 * 6


# The same scripted conversation through the authors' Orchestrator and through our environment.
_COMPARE = _HEAD + r"""
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry

class ScriptedAgent(LLMAgent):
    def __init__(self, script):
        super().__init__(tools=[], domain_policy="", llm="scripted")
        self.script = list(script)
    def _generate_next_message(self, message, state):
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        action, argument = self.script.pop(0)
        if action == "respond":
            return AssistantMessage(role="assistant", content=argument, cost=0.0)
        return AssistantMessage(role="assistant", tool_calls=[ToolCall(id=f"c{len(state.messages)}", name=action, arguments=argument)], cost=0.0)

def theirs(task_id, agent_script, user_script):
    task = m.load_task(task_id)
    o = Orchestrator(domain="retail", agent=ScriptedAgent(agent_script), user=m.scripted_user(user_script),
                     environment=registry.get_env_constructor("retail")(), task=task, max_steps=m.MAX_STEPS, max_errors=m.MAX_ERRORS)
    o.run()
    return [o.step_count, o.num_errors, o.termination_reason.value]

def ours(task_id, agent_script, user_script):
    env = m.make_env({"task_id": task_id, "user_script": user_script})
    env.reset()
    for action, argument in agent_script:
        obs, done, info = env.message(argument) if action == "respond" else env.step(call(action, argument))
        if done:
            break
    return [env.step_count, env.num_errors, env.termination.value]

scripts = json.loads(sys.argv[2])
print(json.dumps([[theirs(*s), ours(*s)] for s in scripts]))
"""


def test_step_and_error_counts_match_the_authors_orchestrator():
    look = ["get_user_details", {"user_id": "mia_garcia_4516"}]
    bad = ["nope", {}]
    scripts = [
        # A conversation the user ends: greeting, replies, tool calls, one unknown tool.
        ["10", [["respond", "Hello, who is this?"], look, ["get_order_details", {"order_id": "#W5490111"}], bad,
                ["respond", "Anything else?"]], ["I am Mia.", "Return both orders.", STOP]],
        # Ten tool errors.
        ["10", [bad] * 10 + [["respond", "never sent"]], ["hi"] * 5],
        # 200 steps by tool calls alone: the 100th result crosses the budget.
        ["10", [look] * 101, ["hi"]],
        # 99 tool calls then a message bound for the user: the check fires before the user answers.
        ["10", [look] * 99 + [["respond", "Still there?"], ["respond", "never sent"]], ["hi", "yes", "yes"]],
        # The user stops at once.
        ["19", [["respond", "Hello"]], [STOP]],
        # Errors and steps together: 9 errors, then the step budget.
        ["10", [bad] * 9 + [look] * 91 + [["respond", "x"]], ["hi", "y"]],
    ]
    pairs = _run(_COMPARE, json.dumps(scripts))
    for theirs, ours in pairs:
        assert theirs == ours, (theirs, ours)
    assert [p[0][2] for p in pairs] == ["user_stop", "too_many_errors", "max_steps", "max_steps", "user_stop", "max_steps"]
    assert pairs[0][0][:2] == [11, 1] and pairs[1][0][:2] == [21, 10] and pairs[2][0][:2] == [201, 0]
    assert pairs[3][0][:2] == [200, 0] and pairs[4][0][:2] == [1, 0]


KEY = "sk-or-v1-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

# Every end condition through the server, with the `ended_by` it records and the evaluation it
# writes, under the worker's key; then every file and reply is searched for that key.
_ENDINGS = r"""
import json, sys
from pathlib import Path
from wikitrace.harness.environment import EnvironmentSession
env_path, python, out_root, key = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
def call(action, argument):
    return json.dumps({"action": action, "argument": argument})
look = call("get_user_details", {"user_id": "mia_garcia_4516"})
bad = call("nope", {})
cases = {
    "stop": (["hi", "###STOP###"], [("message", "ok")], "agent_stopped"),
    "transfer": (["hi", "###TRANSFER###"], [("message", "ok")], "agent_stopped"),
    "oos": (["hi", "###OUT-OF-SCOPE###"], [("message", "ok")], "agent_stopped"),
    "errors": (["hi"], [bad] * 10 + [look], "agent_stopped"),
    "steps": (["hi"], [look] * 101, "agent_stopped"),
    "agent": (["hi", "sure"], [look, ("message", "bye")], "agent_stopped"),
    "harness": (["hi", "sure"], [look], "harness_stopped"),
    # The scripted user runs out of lines: mid-episode, and at the reset itself.
    "simerror": (["hi"], [("message", "ok")], "agent_stopped"),
    # A reply that is billed and then fails the authors' `validate`.
    "invalid": (["hi", ""], [("message", "ok")], "agent_stopped"),
    "resetfail": ([], [], "agent_stopped"),
}
out = {}
for name, (user_script, commands, reason) in cases.items():
    out_dir = out_root / name
    spec = {"task_id": "10", "user_script": user_script}
    first, replies, errors = None, [], []
    with EnvironmentSession(factory=f"{env_path}:make_env", spec=spec, out_dir=out_dir, python=python,
                            env={"OPENROUTER_API_KEY": key}) as env:
        try:
            first, info = env.reset()
        except RuntimeError as exc:
            errors.append(str(exc))
        for c in commands:
            try:
                replies.append(env.message(c[1]) if isinstance(c, tuple) else env.step(c))
            except RuntimeError as exc:
                errors.append(str(exc))
        served = env.finish(reason)
    ev = json.loads((out_dir / "evaluation.json").read_text())
    trajectory = served["trajectory"]
    out[name] = {"first": first, "replies": replies, "errors": errors, "ended_by": served["ended_by"],
                 "steps": served["steps"], "env_error": served["env_error"], "evaluation": ev,
                 "sources": [r.get("source") for r in trajectory if r["role"] in ("environment", "message")]}
    if key in json.dumps(served):
        out[name]["leak"] = "served"
out["_leaks"] = [str(p.relative_to(out_root)) for p in out_root.rglob("*")
                 if p.is_file() and key in p.read_text(errors="replace")]
print(json.dumps(out))
"""


def test_every_end_condition_through_the_server(tmp_path):
    proc = subprocess.run([sys.executable, "-c", _ENDINGS, str(ENV), str(PYTHON), str(tmp_path), KEY],
                          capture_output=True, text=True, timeout=900, check=True)
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert {Path(p).name for p in out["_leaks"]} == set() and not any("leak" in c for c in out.values() if isinstance(c, dict))
    assert (tmp_path / "simerror" / "env_server.log").stat().st_size > 0
    for name, token in (("stop", STOP), ("transfer", TRANSFER), ("oos", OOS)):
        c = out[name]
        assert c["first"] == "hi"
        # A message's reply comes back whole: the runner reads `done`, not a marker.
        assert c["replies"][0] == {"observation": token, "done": True, "over": False}
        assert c["ended_by"] == "done" and c["evaluation"]["ended_by"] == "done"
        assert c["evaluation"]["termination_reason"] == "user_stop" and c["evaluation"]["stop_token"] == token
        # A message is no `take_action` step, and one of the authors' steps, the reply another.
        assert c["evaluation"]["steps"] == 0 and c["evaluation"]["orchestrator_steps"] == 3
        assert c["sources"] == ["user", "user"]  # the reset record, then the reply to the message
    errors = out["errors"]
    assert errors["replies"][9]["done"] and errors["replies"][9]["observation"].endswith("\n\nEpisode over")
    assert errors["replies"][10] == {"observation": "Episode over", "done": True, "over": True}
    assert errors["ended_by"] == "task_failed"
    assert errors["evaluation"]["termination_reason"] == "too_many_errors" and errors["evaluation"]["num_errors"] == 10
    assert errors["evaluation"]["reward"] == 0.0 and errors["evaluation"]["success"] is False
    assert errors["evaluation"]["db_check"] is None and errors["evaluation"]["reward_basis"] is None
    assert errors["evaluation"]["scored"] is False and errors["evaluation"]["judge_attempts"] == 0
    steps = out["steps"]
    assert steps["ended_by"] == "game_time_exhausted" and steps["evaluation"]["termination_reason"] == "max_steps"
    assert steps["evaluation"]["steps"] == 100 and steps["evaluation"]["orchestrator_steps"] == 201
    assert steps["replies"][99]["done"] and steps["replies"][100]["over"]
    assert steps["evaluation"]["scored"] is False and steps["evaluation"]["reward"] == 0.0
    # A session the agent ended with a reply and no call, and one the harness stopped: the user
    # ended neither, so neither is scored, although task 10's DB is at the gold state.
    for name, ended in (("agent", "agent_stopped"), ("harness", "harness_stopped")):
        c = out[name]
        assert c["ended_by"] == ended and c["evaluation"]["ended_by"] == ended
        # The environment learns the close reason and reports it as the termination.
        assert c["evaluation"]["termination_reason"] == ended and c["evaluation"]["stop_token"] == ""
        assert c["evaluation"]["scored"] is False and c["evaluation"]["success"] is False
        assert c["evaluation"]["reward"] == 0.0 and c["evaluation"]["score"] == 0.0
        assert c["evaluation"]["db_check"] is None and c["evaluation"]["reward_breakdown"] is None
        assert c["evaluation"]["nl_assertions"] is None and c["evaluation"]["judge_attempts"] == 0
        assert "generation_error" not in c["evaluation"]
    assert out["agent"]["sources"] == ["user", "env", "user"] and out["agent"]["evaluation"]["messages"] == 1
    assert out["harness"]["sources"] == ["user", "env"]
    assert out["agent"]["evaluation"]["invalid_commands"] == 0
    # A simulator that raises mid-episode, and one that raises at reset: generation failures,
    # not scored, no judge call, the reason recorded.
    sim = out["simerror"]
    assert sim["errors"] == ["environment message: IndexError: pop from empty list"]
    assert sim["ended_by"] == "simulator_error" and sim["env_error"] == "IndexError: pop from empty list"
    assert sim["evaluation"]["generation_error"] == "user_simulator: IndexError: pop from empty list"
    assert sim["evaluation"]["env_error"] == sim["env_error"]
    reset = out["resetfail"]
    assert reset["first"] is None and reset["errors"][0].startswith("environment reset: IndexError")
    assert reset["evaluation"]["generation_error"].startswith("user_simulator: IndexError")
    for c in (sim, reset):
        ev = c["evaluation"]
        assert ev["success"] is False and ev["reward"] is None and ev["score"] == 0.0 and ev["scored"] is False
        assert ev["judge_attempts"] == 0 and ev["calls"]["judge"] == 0 and ev["db_check"] is None
    assert sim["evaluation"]["calls"]["user_simulator"] == 1 and reset["evaluation"]["calls"]["user_simulator"] == 0
    invalid = out["invalid"]["evaluation"]
    assert invalid["generation_error"].startswith("user_simulator: ValueError: UserMessage must have either content")
    assert invalid["calls"]["user_simulator"] == 2 and invalid["scored"] is False
    assert [r["role"] for r in invalid["llm_calls"]] == ["user_simulator"] * 2


# The judge on task 19 (two NL assertions): a reply that parses on the second try, then three failures.
_JUDGE = _HEAD + r"""
good = json.dumps({"results": [{"expectedOutcome": a, "reasoning": "r", "metExpectation": True}
                               for a in m.load_task("19").evaluation_criteria.nl_assertions]})
out = {}
# No script: the authors' judge call runs, and an unroutable model id fails it before any request.
env = m.make_env({"task_id": "19", "user_script": ["hi", "###STOP###"], "judge_model": "no-such-provider/model",
                  "judge_llm_args": {"timeout": 5, "num_retries": 1}})
env.reset()
env.message("ok")
env.close()
out["transport"] = env.evaluation()
for name, script in (("second", ["not json", good]), ("failed", ["not json", "[1, 2", "{"])):
    env = m.make_env({"task_id": "19", "user_script": ["hi", "###STOP###"], "judge_model": "scripted",
                      "judge_llm_args": {"x": 1}, "judge_script": script})
    env.reset()
    env.message("ok")
    env.close()
    out[name] = env.evaluation()
import tau2.evaluator.evaluator_nl_assertions as nl
out["judge_model"] = [nl.DEFAULT_LLM_NL_ASSERTIONS, nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS]
print(json.dumps(out))
"""


def test_judge_attempts_on_a_second_try_parse_and_on_three_failures():
    out = _run(_JUDGE)
    second = out["second"]
    assert second["judge_attempts"] == 2 and second["calls"]["judge"] == 2
    assert [c["met"] for c in second["nl_assertions"]] == [True, True]
    assert all(c["justification"] == "r" for c in second["nl_assertions"])
    assert second["reward_breakdown"]["NL_ASSERTION"] == 1.0 and second["reward_breakdown"]["DB"] == 0.0
    assert second["judge_failed"] is False and second["judge_error"] == "" and second["scored"] is True
    assert [r["role"] for r in second["llm_calls"]] == ["user_simulator"] * 2 + ["judge"] * 2
    failed = out["failed"]
    assert failed["judge_attempts"] == 3 and failed["calls"]["judge"] == 3
    assert failed["nl_assertions"] is None and failed["reward_breakdown"]["NL_ASSERTION"] == 0.0
    assert failed["reward"] == 0.0 and failed["success"] is False
    assert failed["judge_failed"] is True and failed["judge_error"].startswith("JSONDecodeError")
    assert out["judge_model"] == ["scripted", {"x": 1}]
    # A judge whose request fails is a judge failure too: three attempts, the error kept, NL 0,
    # the episode still scored and the simulator's accounting intact.
    transport = out["transport"]
    assert transport["judge_attempts"] == 3 and transport["calls"]["judge"] == 0
    assert transport["judge_failed"] is True and "no-such-provider/model" in transport["judge_error"]
    assert transport["nl_assertions"] is None and transport["reward_breakdown"]["NL_ASSERTION"] == 0.0
    assert transport["scored"] is True and transport["calls"]["user_simulator"] == 2
    assert "generation_error" not in transport


# A simulator request against a server that never answers, with the benchmark's own retry count.
_ATTEMPTS = _HEAD + r"""
import socket, threading, time
from tau2.data_model.message import UserMessage
from tau2.utils.llm_utils import generate
args = json.loads(sys.argv[2])
srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(8)
held = []
def accept():
    while True:
        try: held.append(srv.accept()[0])
        except OSError: return
threading.Thread(target=accept, daemon=True).start()
t0 = time.monotonic()
try:
    generate(sys.argv[3], [UserMessage(role="user", content="hi")],
             api_base=f"http://127.0.0.1:{srv.getsockname()[1]}", api_key="not-a-key", **args)
    error = ""
except Exception as exc:
    error = type(exc).__name__
print(json.dumps({"connections": len(held), "seconds": time.monotonic() - t0, "error": error}))
"""


def test_a_hung_simulator_request_makes_one_attempt_per_retry_and_then_times_out():
    from wikitrace.envs.tau3.benchmarks.tau3 import SIMULATOR_LLM_ARGS, SIMULATOR_MODEL

    args = {**SIMULATOR_LLM_ARGS, "timeout": 1}
    out = _run(_ATTEMPTS, json.dumps(args), SIMULATOR_MODEL, timeout=60)
    attempts = 1 + args["num_retries"]
    assert out["connections"] == attempts and out["error"] == "Timeout"
    assert attempts * args["timeout"] <= out["seconds"] < attempts * args["timeout"] + 5


_QUIET = _HEAD + r"""
env = m.make_env({"task_id": "10", "user_script": ["hello", "###STOP###"]})
env.reset()
env.step(call("get_user_details", {"user_id": "x"}))
env.message("Goodbye")
print(json.dumps({}))
"""


def test_the_authors_log_stays_quiet_below_warning():
    proc = subprocess.run([str(PYTHON), "-c", _QUIET, str(ENV)], capture_output=True, text=True,
                          timeout=600, check=True)
    assert "| DEBUG" not in proc.stderr and "| INFO" not in proc.stderr, proc.stderr[:500]
