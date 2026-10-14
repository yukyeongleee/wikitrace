"""Behavioral tests for the in-container egress relay.
It is half the validity control, so its contract is exercised against a local Node and a
unix-socket echo server; skipped where ``node`` is unavailable."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import threading
from pathlib import Path

import pytest

_RELAY = Path(__file__).resolve().parents[2] / "src/wikitrace/harness/terminal/incontainer_relay.mjs"
_NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(_NODE is None, reason="node not available")


def _unix_echo_server(sock_path: Path, reply_prefix: bytes = b"reply:") -> threading.Thread:
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    srv.listen(1)

    def serve() -> None:
        conn, _ = srv.accept()
        data = conn.recv(1024)
        conn.sendall(reply_prefix + data)
        conn.close()
        srv.close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    return t


@pytest.mark.usefixtures("local_sockets_available")
def test_relay_pipes_bytes_and_keeps_stdout_clean(tmp_path: Path):
    # The wrapped "Pi" here is a node one-liner that prints a known marker to
    # stdout. If the relay wrote anything to stdout, the marker would be polluted.
    sock = tmp_path / "up.sock"
    _unix_echo_server(sock)
    port = _free_port()
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps([{"ip": "127.0.0.1", "port": port, "sock": str(sock)}]))

    # Wrapped command: connect to the listener, send bytes, print the echoed reply
    # to stdout, exit. This exercises the pipe AND proves stdout is the child's alone.
    child = (
        "import('node:net').then(({default:net})=>{"
        f"const c=net.connect({port},'127.0.0.1',()=>c.write('ping'));"
        "c.on('data',d=>{process.stdout.write('CHILD:'+d.toString());c.end();});"
        "c.on('end',()=>process.exit(0));});"
    )
    proc = subprocess.run(
        [_NODE, str(_RELAY), str(cfg), "--", _NODE, "-e", child],
        capture_output=True,
        timeout=20,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    # stdout carries ONLY the child's output — the relay never writes there.
    assert proc.stdout.strip() == "CHILD:reply:ping", repr(proc.stdout)


@pytest.mark.usefixtures("local_sockets_available")
def test_relay_exits_nonzero_when_a_listener_cannot_bind(tmp_path: Path):
    # Reserve a port so the bind failure is independent of root privileges.
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps([{"ip": "127.0.0.1", "port": port, "sock": "/nope.sock"}]))
    sentinel = tmp_path / "child_ran"
    child = f"import('node:fs').then(({{default:fs}})=>fs.writeFileSync({str(sentinel)!r},'x'))"
    proc = subprocess.run(
        [_NODE, str(_RELAY), str(cfg), "--", _NODE, "-e", child],
        capture_output=True,
        timeout=20,
        text=True,
    )
    listener.close()
    assert proc.returncode != 0
    assert not sentinel.exists(), "child ran despite a bind failure"


def test_relay_requires_the_separator(tmp_path: Path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text("[]")
    proc = subprocess.run(
        [_NODE, str(_RELAY), str(cfg)], capture_output=True, timeout=20, text=True
    )
    assert proc.returncode == 2


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p
