"""Host-side control of one environment server: start it, reset, hand the socket to Pi, finish."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wikitrace.harness.environment.protocol import EVALUATION_FILE, request
from wikitrace.harness.environment.tool import SOCKET_NAME

SERVER_PATH = Path(__file__).with_name("server.py")
# In-container paths; the server's output dir stays inside, since the session dir holds files its code must not see.
CONTAINER_SERVER = "/srv/server.py"
CONTAINER_SOCKET_DIR = "/env"
CONTAINER_OUT_DIR = "/tmp/out"
CONTAINER_PREFIX = "wikitrace-env-"


@dataclass(frozen=True)
class Container:
    """Run the server in its own `docker run --rm` container instead of a host process."""

    image: str
    mounts: tuple[tuple[str, str, bool], ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    name: str = field(
        default_factory=lambda: f"{CONTAINER_PREFIX}{os.getpid()}-{uuid.uuid4().hex[:12]}")


@dataclass
class EnvironmentSession:
    """One episode's server process, alive from ``__enter__`` to ``__exit__``."""
    # `python` has the simulator installed; the server and factory load by path, so it needs nothing of ours.

    factory: str
    spec: dict[str, Any]
    out_dir: Path
    step_limit: int = 0
    end_command: str = ""
    python: str = sys.executable
    env: dict[str, str] = field(default_factory=dict)
    container: Container | None = None
    startup_timeout: float = 60.0
    request_timeout: float = 600.0
    socket_dir: Path = field(default_factory=lambda: Path(tempfile.mkdtemp(prefix="env-")))
    _proc: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _log: Any = field(default=None, init=False, repr=False)
    _finished: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    @property
    def socket_path(self) -> Path:
        # A tmp dir keeps the path under the unix-socket length limit; it is bind-mounted into the sandbox, so holds the socket alone.
        return self.socket_dir / SOCKET_NAME

    def argv(self) -> list[str]:
        """The command that starts the server, on the host or in its container."""
        tail = ["--step-limit", str(self.step_limit), "--end-command", self.end_command]
        c = self.container
        if c is None:
            return [self.python, str(SERVER_PATH), "--factory", self.factory,
                    "--socket", str(self.socket_path), "--out", str(self.out_dir), *tail]
        cmd = ["docker", "run", "--rm", "-i", "--pull", "never", "--name", c.name,
               "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}"]
        for key, value in c.env:
            cmd += ["-e", f"{key}={value}"]
        # `--mount` refuses a missing or relative source, where `-v` would create the first on the host.
        for host, inside, readonly in [(str(self.socket_dir), CONTAINER_SOCKET_DIR, False), *c.mounts]:
            bind = f"type=bind,source={os.path.abspath(host)},target={inside}"
            cmd += ["--mount", bind + ",readonly" if readonly else bind]
        return [*cmd, c.image, "python", CONTAINER_SERVER, "--factory", self.factory,
                "--socket", f"{CONTAINER_SOCKET_DIR}/{SOCKET_NAME}", "--out", CONTAINER_OUT_DIR,
                *tail, "--stdin-lifeline"]

    def __enter__(self) -> "EnvironmentSession":
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._log = (self.out_dir / "env_server.log").open("ab")
        try:
            self._start()
        except BaseException:
            # No ``__exit__`` follows a failed ``__enter__``, so nothing may outlive this.
            self._kill()
            self._log.close()
            shutil.rmtree(self.socket_dir, ignore_errors=True)
            raise
        return self

    def _start(self) -> None:
        self._proc = subprocess.Popen(
            self.argv(), stdin=subprocess.PIPE, stdout=self._log, stderr=subprocess.STDOUT,
            start_new_session=True, env={**os.environ, **self.env} if self.env else None,
        )
        assert self._proc.stdin is not None
        try:
            spec = json.dumps(self.spec, ensure_ascii=False).encode("utf-8")
            if self.container is None:
                self._proc.stdin.write(spec)
                self._proc.stdin.close()
            else:
                self._proc.stdin.write(spec + b"\n")
                self._proc.stdin.flush()
        except BrokenPipeError:
            pass  # the server died first; the startup wait below reports it
        deadline = time.monotonic() + self.startup_timeout
        while not self.socket_path.exists():
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"environment server exited during startup (code {self._proc.returncode}); "
                    f"see {self.out_dir / 'env_server.log'}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"environment server did not open {self.socket_path}")
            time.sleep(0.05)

    def _request(self, op: str, **payload: Any) -> dict[str, Any]:
        reply = request(str(self.socket_path), {"op": op, **payload}, timeout=self.request_timeout)
        if "error" in reply:
            raise RuntimeError(f"environment {op}: {reply['error']}")
        return reply

    def reset(self) -> tuple[str, dict[str, Any]]:
        reply = self._request("reset")
        return reply["observation"], reply["info"]

    def step(self, command: str) -> dict[str, Any]:
        return self._request("step", input=command)

    def message(self, text: str) -> dict[str, Any]:
        return self._request("message", text=text)

    def status(self) -> dict[str, Any]:
        return self._request("status")

    def finish(self, reason: str = "harness_stopped") -> dict[str, Any]:
        """Close the episode; the reply is its result plus ``trajectory`` and ``evaluation``."""
        if self._proc is None or self._proc.poll() is not None:
            return self._finished
        self._finished = self._request("close", reason=reason)
        if self.container is not None:
            (self.out_dir / EVALUATION_FILE).write_text(
                json.dumps(self._finished.get("evaluation") or {}, ensure_ascii=False, indent=1),
                encoding="utf-8")
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._kill()
        return self._finished

    def _kill(self) -> None:
        if self.container is not None:
            # A signal to the `docker run` client does not reach its container; the stdin closed below still ends it.
            try:
                subprocess.run(["docker", "rm", "-f", self.container.name],
                               capture_output=True, timeout=60)
            except (subprocess.TimeoutExpired, OSError):
                pass
            if self._proc is not None and self._proc.stdin is not None:
                try:
                    self._proc.stdin.close()
                except OSError:
                    pass
        if self._proc is None or self._proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
        except OSError:
            self._proc.kill()
        self._proc.wait(timeout=10)

    def __exit__(self, *exc: Any) -> None:
        try:
            if self._proc is not None and self._proc.poll() is None:
                self.finish()
        finally:
            self._kill()
            if self._log is not None:
                self._log.close()
            shutil.rmtree(self.socket_dir, ignore_errors=True)
