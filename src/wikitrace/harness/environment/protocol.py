"""The environment Protocol and the newline-JSON wire format its server and clients share."""
# Standard library only: the server runs under a benchmark's own interpreter.

from __future__ import annotations

import json
import socket
from typing import Any, Protocol

EPISODE_OVER = "Episode over"
EVALUATION_FILE = "evaluation.json"

# What ended the episode; `env_error` carries the message of a raising `step`.
ENDED_BY = (
    "done", "task_failed", "budget_exhausted", "game_time_exhausted",
    "agent_stopped", "harness_stopped", "simulator_error",
)
CLOSE_REASONS = ("agent_stopped", "harness_stopped")


class Environment(Protocol):
    """One episode of a text environment: reset once, then step until done."""

    def reset(self) -> tuple[str, dict[str, Any]]:
        """Start the episode; return the first observation and the episode info."""
        ...

    def step(self, command: str) -> tuple[str, bool, dict[str, Any]]:
        """Apply one command; return the observation, whether the episode ended, and info."""
        ...

    def close(self) -> None:
        """Release the simulator."""
        ...

    def message(self, text: str) -> tuple[str, bool, dict[str, Any]]:
        """Send the agent's message to the user; return the reply, whether the episode ended, and info. Optional."""
        ...

    def on_end_command(self, command: str) -> None:
        """The agent's end command, which never reaches ``step``. Optional."""
        ...

    def evaluation(self) -> dict[str, Any]:
        """The benchmark's block of ``evaluation.json``; called after ``close``. Optional."""
        ...


INFO_DEFAULTS: dict[str, Any] = {
    "success": False,
    "score": 0.0,
    "failed": False,
    "invalid_command": False,
}


def normalize_info(info: dict[str, Any] | None) -> dict[str, Any]:
    """Fill the shared ``info`` keys so a consumer never branches on their absence."""
    out = dict(INFO_DEFAULTS)
    out.update(info or {})
    out["success"] = bool(out["success"])
    out["score"] = float(out["score"])
    out["failed"] = bool(out["failed"])
    out["invalid_command"] = bool(out["invalid_command"])
    return out


def send_line(conn: socket.socket, payload: dict[str, Any]) -> None:
    conn.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))


def recv_line(conn: socket.socket) -> dict[str, Any] | None:
    """Read one newline-terminated JSON object; ``None`` on a closed connection."""
    buf = bytearray()
    while True:
        chunk = conn.recv(65536)
        if not chunk:
            return None if not buf else json.loads(buf.decode("utf-8"))
        buf.extend(chunk)
        nl = buf.find(b"\n")
        if nl != -1:
            return json.loads(buf[:nl].decode("utf-8"))


def request(socket_path: str, payload: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
    """One request/response round-trip over a fresh connection to the server."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(timeout)
        conn.connect(socket_path)
        send_line(conn, payload)
        reply = recv_line(conn)
    if reply is None:
        raise ConnectionError(f"environment server at {socket_path} closed without replying")
    return reply
