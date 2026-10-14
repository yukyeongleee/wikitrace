"""The AppWorld episode container from inside a real episode: what `exec` code sees and reaches."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from wikitrace.envs.appworld.session import AppWorldSession, image_tag, require_image

REPO = Path(__file__).resolve().parents[2]
INSTALL = REPO / "data" / "appworld"


def _unready() -> str:
    """Why this host cannot run an episode container at all; a missing image is a failure."""
    if not (INSTALL / "data" / "tasks").is_dir() or not list(INSTALL.glob(".venv/lib/python*")):
        return "AppWorld is not installed"
    if shutil.which("docker") is None:
        return "docker is not on PATH"
    if subprocess.run(["docker", "info"], capture_output=True, timeout=60).returncode != 0:
        return "the docker daemon is not reachable"
    return ""


_WHY = _unready()
pytestmark = pytest.mark.skipif(bool(_WHY), reason=_WHY)
TASK = "" if _WHY else sorted(p.name for p in (INSTALL / "data" / "tasks").iterdir())[0]


def test_the_episode_image_of_this_code_is_built():
    require_image(image_tag(INSTALL))


def test_a_real_episode_is_graded_on_the_host(tmp_path):
    # AppWorld refuses a saved change log whose path holds "memory", as a no-memory run's does.
    sdir = tmp_path / "no-memory__trial-00" / "solver"
    with AppWorldSession(INSTALL, TASK, sdir) as env:
        env.reset()
        assert "supervisor" in _exec(env, "print(apis.api_docs.show_app_descriptions())")
        _exec(env, "profile = apis.supervisor.show_profile()\nprint(profile['email'])")
        assert env.step("apis.supervisor.complete_task()")["done"] is True
        episode = env.finish("agent_stopped")
    evaluation = json.loads((sdir / "evaluation.json").read_text())
    assert evaluation == episode["evaluation"]
    assert list(evaluation) == ["success", "ended_by", "steps", "invalid_commands", "tests"]
    assert evaluation["ended_by"] == "done" and evaluation["steps"] == 3
    assert evaluation["tests"]["total"] > 0
    assert not env.episode_root.exists()

# `os.walk` is refused by AppWorld's safety checks; `os.listdir` is not.
_WALK = """
import os, json
def walk(d):
    out = [d]
    for n in os.listdir(d):
        p = os.path.join(d, n)
        out += walk(p) if os.path.isdir(p) and not os.path.islink(p) else [p]
    return out
print(json.dumps(walk('/appworld')))
"""


def _dotenv_names() -> set[str]:
    """The variable names in the runner's `.env`, the nearest one above the repo."""
    for d in (REPO, *REPO.parents):
        if (d / ".env").is_file():
            lines = (d / ".env").read_text(encoding="utf-8").splitlines()
            return {ln.split("=", 1)[0].removeprefix("export ").strip()
                    for ln in lines if "=" in ln and not ln.lstrip().startswith("#")}
    return set()


def _containers(name: str) -> list[str]:
    return subprocess.run(["docker", "ps", "-aq", "--filter", f"name=^{name}$"],
                          capture_output=True, text=True, check=True).stdout.split()


def _exec(env: AppWorldSession, code: str) -> str:
    return env.step(code)["observation"]


def test_exec_code_sees_one_task_without_its_gold_and_nothing_of_the_host(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WIKI_TEST_SECRET", "leak")
    with AppWorldSession(INSTALL, TASK, tmp_path / "solver") as env:
        env.reset()
        name = env.container.name
        network = subprocess.run(["docker", "inspect", "-f", "{{.HostConfig.NetworkMode}}", name],
                                 capture_output=True, text=True, check=True).stdout.strip()
        names = json.loads(_exec(env, "import os, json\nprint(json.dumps(sorted(os.environ)))"))
        cwd, root, home, host_paths = json.loads(_exec(env, (
            "import os, json\nprint(json.dumps([os.getcwd(), os.listdir('/'), os.listdir('/home'), "
            f"[os.path.exists(p) for p in {[str(REPO), str(INSTALL), str(Path.home())]!r}]]))")))
        tree = json.loads(_exec(env, _WALK))
        dial = _exec(env, "import urllib.request\n"
                          "urllib.request.build_opener().open('http://1.1.1.1', timeout=5)")
    assert not set(names) & (_dotenv_names() | {"AGENT_WIKI_TEST_SECRET"})
    assert cwd == "/appworld" and home == [] and host_paths == [False, False, False]
    assert REPO.parts[1] not in root
    assert not any("ground_truth" in p for p in tree)
    assert {p.split("/")[4] for p in tree if p.startswith("/appworld/data/tasks/")} == {TASK}
    assert network == "none" and "Network is unreachable" in dial
    assert _containers(name) == []


def test_a_killed_session_leaves_no_container(tmp_path):
    env = AppWorldSession(INSTALL, TASK, tmp_path / "solver")
    env.__enter__()
    try:
        assert _containers(env.container.name)
        env._kill()
        assert _containers(env.container.name) == []
    finally:
        env.__exit__(None, None, None)
    assert not env.episode_root.exists()


_RUNNER = """
import sys, time
from pathlib import Path
from wikitrace.envs.appworld.session import AppWorldSession
env = AppWorldSession(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
env.__enter__()
print(env.container.name, flush=True)
time.sleep(600)
"""


def test_a_container_does_not_outlive_a_runner_killed_outright(tmp_path):
    runner = subprocess.Popen(
        [sys.executable, "-c", _RUNNER, str(INSTALL), TASK, str(tmp_path / "solver")],
        stdout=subprocess.PIPE, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "src")},
    )
    name = runner.stdout.readline().strip()
    assert _containers(name)
    runner.kill()
    runner.wait()
    deadline = time.monotonic() + 30
    while _containers(name) and time.monotonic() < deadline:
        time.sleep(0.5)
    assert _containers(name) == []
