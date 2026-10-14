"""Unix-socket-to-TCP L4 relay per allowed host, destination fixed in Python so the container cannot steer it."""

from __future__ import annotations

import socket
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class RelayEntry:
    """One allowlisted host and the loopback address the container reaches it at."""

    host: str
    port: int
    loopback_ip: str
    sock_path: str     # the unix socket on the host, bind-mounted into the container


def host_port_from_url(url: str, default_port: int = 443) -> tuple[str, int]:
    """``https://openrouter.ai/api/v1`` -> ``("openrouter.ai", 443)``."""
    parts = urlsplit(url)
    host = parts.hostname
    if not host:
        raise ValueError(f"cannot parse a host from base_url {url!r}")
    return host, parts.port or default_port


class EgressRelay:
    """Runs one unix-socket <-> TCP relay per allowed host, for one Pi session."""
    # Per-session: ``net_dir`` is a fresh tmp dir, so parallel sessions never share sockets.

    def __init__(self, net_dir: Path, hosts: list[tuple[str, int]]) -> None:
        seen: set[tuple[str, int]] = set()
        ordered: list[tuple[str, int]] = []
        for hp in hosts:
            if hp not in seen:
                seen.add(hp)
                ordered.append(hp)
        if len(ordered) > 254:
            raise ValueError(f"egress allowlist too large: {len(ordered)} hosts (max 254)")
        self.net_dir = net_dir
        self.entries: list[RelayEntry] = [
            RelayEntry(
                host=h, port=p, loopback_ip=f"127.0.0.{i + 1}",
                sock_path=str(net_dir / f"{h}.sock"),
            )
            for i, (h, p) in enumerate(ordered)
        ]
        self._servers: list[socket.socket] = []
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()


    def add_host_args(self) -> list[str]:
        """``--add-host host:127.0.0.x`` flags pinning each name to its listener."""
        args: list[str] = []
        for e in self.entries:
            args += ["--add-host", f"{e.host}:{e.loopback_ip}"]
        return args

    def relay_config(self) -> list[dict]:
        """The in-container listener config: {ip, port, sock} per host."""
        # ``port`` must equal what Pi dials: the ``/etc/hosts`` pin changes the address only.
        return [
            {"ip": e.loopback_ip, "port": e.port, "sock": _in_container(e.sock_path)}
            for e in self.entries
        ]


    def start(self) -> None:
        """Bind + serve every host's socket. Raises (before any container) on failure."""
        self.net_dir.mkdir(parents=True, exist_ok=True)
        for e in self.entries:
            Path(e.sock_path).unlink(missing_ok=True)
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                srv.bind(e.sock_path)
                srv.listen(64)
            except OSError as exc:
                srv.close()
                self.stop()
                raise RuntimeError(f"egress relay could not bind {e.sock_path}: {exc}") from exc
            srv.settimeout(0.5)
            self._servers.append(srv)
            t = threading.Thread(
                target=self._accept_loop, args=(srv, e.host, e.port), daemon=True
            )
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        for srv in self._servers:
            try:
                srv.close()
            except OSError:
                pass
        for t in self._threads:
            t.join(timeout=2)
        for e in self.entries:
            Path(e.sock_path).unlink(missing_ok=True)
        self._servers.clear()
        self._threads.clear()

    def __enter__(self) -> EgressRelay:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


    def _accept_loop(self, srv: socket.socket, host: str, port: int) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(
                target=self._bridge, args=(conn, host, port), daemon=True
            ).start()

    def _bridge(self, downstream: socket.socket, host: str, port: int) -> None:
        """Open TCP to the real host and pipe both directions until either closes."""
        try:
            upstream = socket.create_connection((host, port), timeout=30)
        except OSError:
            downstream.close()
            return
        # A CONNECT deadline only: a leftover timeout would truncate a reasoning model's first-token wait.
        upstream.settimeout(None)
        a = threading.Thread(target=self._pump, args=(downstream, upstream), daemon=True)
        b = threading.Thread(target=self._pump, args=(upstream, downstream), daemon=True)
        a.start()
        b.start()
        a.join()
        b.join()

    @staticmethod
    def _pump(src: socket.socket, dst: socket.socket) -> None:
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    break
                dst.sendall(data)
        except OSError:
            pass
        finally:
            # The first pump to see EOF tears both down; sendall completes before the peer's EOF is read.
            try:
                dst.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            for s in (src, dst):
                try:
                    s.close()
                except OSError:
                    pass


def _in_container(host_sock_path: str) -> str:
    """Map a host socket path under the session net dir to its ``/net`` mount point."""
    return "/net/" + Path(host_sock_path).name
