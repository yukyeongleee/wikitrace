"""Serve one environment episode over a unix socket; runnable under any interpreter."""
# Invoked by file path from the benchmark's own virtualenv: stdlib only, factory loaded by path.

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import inspect
import json
import os
import signal
import socket
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Callable

try:
    from wikitrace.harness.environment.protocol import (
        CLOSE_REASONS, EPISODE_OVER, EVALUATION_FILE, normalize_info, recv_line, send_line,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from protocol import (  # type: ignore[no-redef]
        CLOSE_REASONS, EPISODE_OVER, EVALUATION_FILE, normalize_info, recv_line, send_line,
    )


def load_factory(spec: str) -> Callable[[dict[str, Any]], Any]:
    """``/path/to/module.py:attr`` -> the environment factory it names."""
    path, _, attr = spec.rpartition(":")
    # Prefixed so an adapter named after its simulator (envs/alfworld.py) does not shadow it.
    module_spec = importlib.util.spec_from_file_location(f"wikitrace_env_{Path(path).stem}", path)
    if module_spec is None or module_spec.loader is None:
        raise SystemExit(f"cannot load environment module {path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return getattr(module, attr)


class Episode:
    """One environment behind the socket: step accounting and the in-memory step record."""

    def __init__(self, make_env: Callable[[], Any], out_dir: Path, step_limit: int = 0,
                 end_command: str = "") -> None:
        self._make_env = make_env
        self.env: Any = None
        self.out_dir = out_dir
        self.step_limit = step_limit
        self.end_command = end_command
        self.steps = 0
        self.invalid_commands = 0
        self.info: dict[str, Any] = normalize_info(None)
        self.started = False
        self.ended_by = ""
        self.end_commanded = False
        self.error = ""
        self.closed = False
        self.trajectory: list[dict[str, Any]] = []
        self.evaluation: dict[str, Any] = {}
        (out_dir / EVALUATION_FILE).unlink(missing_ok=True)

    def _record(self, **line: Any) -> None:
        self.trajectory.append(line)

    def reset(self) -> dict[str, Any]:
        if self.started:
            return {"error": "episode already started"}
        self.env = self._make_env()
        observation, info = self.env.reset()
        self.info = normalize_info(info)
        self.started = True
        self._record(step=0, role="environment", observation=observation, info=self.info,
                     **self._source())
        return {"observation": observation, "info": self.info, "task": self.info.get("task", "")}

    def _source(self) -> dict[str, str]:
        """The observation's source when the environment names one (a simulated user, its tools)."""
        return {"source": str(self.info["source"])} if self.info.get("source") else {}

    def step(self, command: str) -> dict[str, Any]:
        if not self.started:
            return {"error": "episode not started: reset first"}
        if self.ended_by:
            return {"observation": EPISODE_OVER, "done": True, "over": True}
        if self._is_end_command(command):
            self._record(step=self.steps, role="agent", command=command, served=False)
            if callable(hook := getattr(self.env, "on_end_command", None)):
                hook(command)
            self.end_commanded = True
            self.finish("agent_stopped")
            return {"observation": EPISODE_OVER, "done": True, "over": True}
        self.steps += 1
        self._record(step=self.steps, role="agent", command=command)
        try:
            observation, done, info = self.env.step(command)
        except Exception as exc:
            # A raising simulator ends the episode as a harness fault, reported instead of scoring the agent.
            self.error = f"{type(exc).__name__}: {exc}"
            self.finish("simulator_error")
            raise
        self.info = normalize_info(info)
        if self.info["invalid_command"]:
            self.invalid_commands += 1
        self._record(step=self.steps, role="environment", observation=observation, info=self.info,
                     **self._source())
        if done:
            self._finish_done()
        elif self.step_limit and self.steps >= self.step_limit:
            self.finish("budget_exhausted")
        if self.ended_by:
            observation = f"{observation}\n\n{EPISODE_OVER}"
        return {"observation": observation, "done": bool(self.ended_by), "over": False}

    def _finish_done(self) -> None:
        """End an episode the environment reported done, by the reason its info gives."""
        if self.info["failed"]:
            self.finish("task_failed")
        else:
            self.finish("game_time_exhausted" if self.info.get("step_limit_hit") else "done")

    def message(self, text: str) -> dict[str, Any]:
        """A message to the user of an environment that has one; the observation is the reply."""
        if not self.started:
            return {"error": "episode not started: reset first"}
        if not callable(getattr(self.env, "message", None)):
            return {"error": "this environment takes no messages"}
        if self.ended_by:
            return {"observation": EPISODE_OVER, "done": True, "over": True}
        try:
            observation, done, info = self.env.message(text)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.finish("simulator_error")
            raise
        self.info = normalize_info(info)
        self._record(step=self.steps, role="message", message=text, observation=observation,
                     info=self.info, **self._source())
        if done:
            self._finish_done()
        return {"observation": observation, "done": bool(self.ended_by), "over": False}

    def _is_end_command(self, command: str) -> bool:
        """The end command alone, or followed by `:` and an argument (a reason)."""
        c = command.strip().lower()
        return bool(self.end_command) and (c == self.end_command or c.startswith(f"{self.end_command}:"))

    def status(self) -> dict[str, Any]:
        return {"started": self.started, "steps": self.steps, "step_limit": self.step_limit,
                "ended_by": self.ended_by, "info": self.info}

    def finish(self, ended_by: str) -> dict[str, Any]:
        """End the episode once (a value of ``protocol.ENDED_BY``); a later call returns the same result."""
        if not self.ended_by:
            self.ended_by = ended_by
        return self.result()

    def result(self) -> dict[str, Any]:
        return {
            "success": self.info["success"],
            "score": self.info["score"],
            "failed": self.info["failed"],
            "steps": self.steps,
            "step_limit": self.step_limit,
            "step_limit_hit": self.ended_by in ("budget_exhausted", "game_time_exhausted"),
            "invalid_commands": self.invalid_commands,
            "ended_by": self.ended_by,
            "end_command": self.end_commanded,
            "env_error": self.error,
            "info": self.info,
        }

    def close(self, reason: str = "harness_stopped") -> None:
        """Release the simulator, then write ``evaluation.json`` once."""
        if self.closed:
            return
        self.closed = True
        self.finish(reason if reason in CLOSE_REASONS else "harness_stopped")
        try:
            if self.env is not None:
                self.env.close()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        self.evaluation = self._evaluate()
        (self.out_dir / EVALUATION_FILE).write_text(
            json.dumps(self.evaluation, ensure_ascii=False, indent=1), encoding="utf-8")

    def _evaluate(self) -> dict[str, Any]:
        """The common head, then the benchmark's block; the block's ``success``/``score`` join the head."""
        block: dict[str, Any] = {}
        if callable(evaluation := getattr(self.env, "evaluation", None)):
            kwargs = {"ended_by": self.ended_by} if "ended_by" in inspect.signature(evaluation).parameters else {}
            try:
                block = dict(evaluation(**kwargs) or {})
            except Exception:  # noqa: BLE001
                traceback.print_exc()
        out: dict[str, Any] = {"success": bool(block.pop("success", self.info["success"]))}
        if "score" in block:
            out["score"] = block.pop("score")
        out.update(ended_by=self.ended_by, steps=self.steps, invalid_commands=self.invalid_commands)
        if self.ended_by == "simulator_error":
            out["env_error"] = self.error
        out.update(block)
        return out

    def close_payload(self) -> dict[str, Any]:
        """The close reply."""
        return {**self.result(), "trajectory": self.trajectory, "evaluation": self.evaluation}


class ParentGone(BaseException):
    """The process that started the server is gone, or told it to stop."""
    # Not an Exception: the request loop's error handler must not answer it as a reply.


def bind_to_parent() -> int:
    """Ask the kernel for SIGTERM when the parent dies (Linux only); return the parent's pid."""
    # The signal is tied to the spawning thread, so start the session from a thread that outlives the episode.
    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.prctl(1, int(signal.SIGTERM), 0, 0, 0)  # PR_SET_PDEATHSIG
        except (OSError, AttributeError):
            pass
    return os.getppid()


def watch_stdin() -> None:
    """SIGTERM this process once stdin closes: the client's lifeline into a container."""
    # A container cannot see its client's pid, but `docker run -i` closes stdin when the client dies.
    sys.stdin.read()
    os.kill(os.getpid(), signal.SIGTERM)


def parent_alive(parent_pid: int) -> bool:
    """The poll fallback for a parent death the death signal did not deliver."""
    return os.getppid() == parent_pid


def serve(episode: Episode, socket_path: str, parent_pid: int, poll_interval: float = 1.0) -> None:
    """Answer one request per connection until ``close`` arrives or the parent is gone."""
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as srv:
        srv.bind(socket_path)
        srv.listen(8)
        srv.settimeout(poll_interval)
        while True:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                if not parent_alive(parent_pid):
                    raise ParentGone(f"parent {parent_pid} is gone")
                continue
            with conn:
                try:
                    req = recv_line(conn)
                except Exception:  # noqa: BLE001
                    continue
                if not req:
                    continue
                op = req.get("op")
                try:
                    if op == "reset":
                        reply = episode.reset()
                    elif op == "step":
                        reply = episode.step(str(req.get("input", "")))
                    elif op == "message":
                        reply = episode.message(str(req.get("text", "")))
                    elif op == "status":
                        reply = episode.status()
                    elif op == "close":
                        # Closed before the reply so close-time work (AppWorld evaluates) is bounded by the request.
                        episode.close(str(req.get("reason", "harness_stopped")))
                        reply = episode.close_payload()
                    else:
                        reply = {"error": f"unknown op {op!r}"}
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc()
                    reply = {"error": f"{type(exc).__name__}: {exc}",
                             "traceback": traceback.format_exc()}
                send_line(conn, reply)
                if op == "close":
                    break


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--factory", required=True, help="/path/to/module.py:factory_name")
    ap.add_argument("--socket", required=True, help="unix socket path to serve on")
    ap.add_argument("--out", required=True, type=Path, help="where evaluation.json goes")
    ap.add_argument("--step-limit", type=int, default=0, help="0 = no server-side limit")
    ap.add_argument("--end-command", default="",
                    help="a step equal to this, or to it plus ':' and text (stripped, lower-cased), "
                         "ends the episode as the agent's stop")
    ap.add_argument("--stdin-lifeline", action="store_true",
                    help="the spec is one stdin line, and stdin closing ends the episode")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # The spec arrives on stdin: the socket dir is bind-mounted into the sandbox and the spec names the task.
    spec = json.loads(sys.stdin.readline()) if args.stdin_lifeline else json.load(sys.stdin)
    factory = load_factory(args.factory)
    episode = Episode(lambda: factory(spec), args.out, args.step_limit, args.end_command)
    parent_pid = os.getppid() if args.stdin_lifeline else bind_to_parent()

    def on_sigterm(*_: Any) -> None:
        raise ParentGone("SIGTERM")

    signal.signal(signal.SIGTERM, on_sigterm)
    if args.stdin_lifeline:
        threading.Thread(target=watch_stdin, daemon=True).start()
    try:
        serve(episode, args.socket, parent_pid)
    except ParentGone as exc:
        print(f"environment server: {exc}; closing the episode", file=sys.stderr)
    finally:
        episode.close()
        if os.path.exists(args.socket):
            os.unlink(args.socket)


if __name__ == "__main__":
    main()
