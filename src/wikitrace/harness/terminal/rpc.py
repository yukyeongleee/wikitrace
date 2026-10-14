"""stdio JSON-RPC transport for a live Pi session: bounded line reads, one round-trip."""

from __future__ import annotations

import fcntl
import json
import os
import select
import time
from typing import Any


def _peek_buffered(stream: Any) -> bytes:
    """Whatever is ALREADY in ``stream``'s read buffer, without ever blocking."""
    # ``peek`` issues a raw read when the buffer is empty, so the fd goes non-blocking for the duration.
    peek = getattr(stream, "peek", None)
    if peek is None:
        return b""
    try:
        fd = stream.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    except Exception:
        return b""
    try:
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        return peek(0) or b""
    except Exception:
        return b""
    finally:
        try:
            fcntl.fcntl(fd, fcntl.F_SETFL, flags)
        except Exception:
            pass


def _readline_by(stdout: Any, buf: bytearray, deadline: float) -> bytes | None:
    """One line from ``stdout``; ``b""`` at EOF; ``None`` once ``deadline`` passes."""
    # Owns the line buffer: ``readline()`` could BLOCK after ``select()`` and wedge the phase and every run behind it.
    while True:
        nl = buf.find(b"\n")
        if nl >= 0:
            line = bytes(buf[: nl + 1])
            del buf[: nl + 1]
            return line
        # Bytes already inside the BufferedReader are invisible to select(), so take those first.
        peek = _peek_buffered(stdout)
        if peek:
            buf += stdout.read1(len(peek))
            continue
        budget = deadline - time.monotonic()
        if budget <= 0:
            return None
        ready, _, _ = select.select([stdout], [], [], budget)
        if not ready:
            return None
        chunk = os.read(stdout.fileno(), 65536)
        if not chunk:
            buf.clear()
            return b""
        buf += chunk


def _rpc_roundtrip(proc: Any, command: str, timeout: float, buf: bytearray | None = None) -> Any:
    """Send one RPC command to a LIVE Pi session and return its ``data``, or None."""
    # Asking Pi is the only way its API cost is observable; never raises, never blocks past ``timeout``.
    stdin, stdout = getattr(proc, "stdin", None), getattr(proc, "stdout", None)
    if stdin is None or stdout is None or proc.poll() is not None:
        return None
    try:
        stdin.write((json.dumps({"id": f"py-{command}", "type": command}) + "\n").encode())
        stdin.flush()
    except Exception:
        return None
    deadline = time.monotonic() + timeout
    # The event loop reads in 64 KB chunks, so this response can already sit in the caller's leftover bytes.
    if buf is None:
        buf = bytearray()
    while True:
        try:
            raw = _readline_by(stdout, buf, deadline)
        except Exception:
            return None
        if not raw:
            return None
        try:
            event = json.loads(raw.decode("utf-8").rstrip("\r\n"))
        except Exception:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") != "response" or event.get("command") != command:
            continue
        return event.get("data") if event.get("success") else None
