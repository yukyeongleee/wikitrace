"""Unit tests for the traverse egress relay, host side.
The relay is a plain unix-socket to TCP byte pipe, so its contract is fully exercisable
against a local echo server, with no docker and no provider."""

from __future__ import annotations

from pathlib import Path

import pytest

from wikitrace.harness.terminal.egress import EgressRelay, host_port_from_url


def test_host_port_from_url():
    assert host_port_from_url("https://openrouter.ai/api/v1") == ("openrouter.ai", 443)
    assert host_port_from_url("http://localhost:8080/x") == ("localhost", 8080)
    with pytest.raises(ValueError):
        host_port_from_url("not-a-url")


def test_loopback_ips_are_assigned_per_host_and_deduped(tmp_path: Path):
    relay = EgressRelay(tmp_path, [("a.com", 443), ("b.com", 443), ("a.com", 443)])
    # dedup keeps first occurrence; one loopback IP per distinct host
    assert [e.host for e in relay.entries] == ["a.com", "b.com"]
    assert [e.loopback_ip for e in relay.entries] == ["127.0.0.1", "127.0.0.2"]
    # the container-facing config points at the /net mount, not the host path
    cfg = relay.relay_config()
    assert cfg[0] == {"ip": "127.0.0.1", "port": 443, "sock": "/net/a.com.sock"}
    assert relay.add_host_args() == [
        "--add-host", "a.com:127.0.0.1", "--add-host", "b.com:127.0.0.2",
    ]


def test_allowlist_over_254_hosts_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="too large"):
        EgressRelay(tmp_path, [(f"h{i}.com", 443) for i in range(255)])




def test_start_raises_on_unbindable_socket_path(tmp_path: Path):
    # A socket path under a nonexistent, uncreatable parent fails at bind — the
    # fail-loud contract: a broken control aborts before any container starts.
    missing = tmp_path / "nope"
    missing.write_text("not a dir")  # so mkdir of a child path fails
    relay = EgressRelay(missing / "sub", [("a.com", 443)])
    with pytest.raises((RuntimeError, OSError, NotADirectoryError, FileExistsError)):
        relay.start()
