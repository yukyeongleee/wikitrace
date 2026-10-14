"""Host relay byte forwarding using local TCP and Unix sockets."""

import socket
import threading
import time
from pathlib import Path
import pytest
from wikitrace.harness.terminal.egress import EgressRelay

pytestmark = pytest.mark.usefixtures("local_sockets_available")


def test_relay_pipes_bytes_bidirectionally_and_cleans_up(tmp_path: Path):
    # A local echo server stands in for the real upstream host. Bytes sent into the
    # unix socket must arrive at the server and its reply must come back verbatim.
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def echo() -> None:
        conn, _ = srv.accept()
        data = conn.recv(1024)
        conn.sendall(b"reply:" + data)
        conn.close()

    threading.Thread(target=echo, daemon=True).start()

    relay = EgressRelay(tmp_path, [("127.0.0.1", port)])
    relay.start()
    try:
        sock_path = relay.entries[0].sock_path
        assert Path(sock_path).exists()
        client = socket.socket(socket.AF_UNIX)
        client.connect(sock_path)
        client.sendall(b"ping")
        time.sleep(0.2)
        assert client.recv(1024) == b"reply:ping"
        client.close()
    finally:
        relay.stop()
    # stop() removes the socket so a re-run never trips over a stale one
    assert not Path(relay.entries[0].sock_path).exists()
