"""The terminal agent's sandbox: image check, egress pins, `docker run` argv, corpus freeze."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

from wikitrace.harness.environment.tool import CITATION_ENV_VAR, SOCKET_ENV_VAR, SOCKET_NAME
from wikitrace.harness.terminal.egress import EgressRelay, host_port_from_url
from wikitrace.harness.terminal.usage import _PI_AGENT_DIR_NAME, _PI_AGENT_DIR_VAR

# Env var pi-patch 0004 reads: the CEILING on the bash tool's `timeout` and its default when omitted.
_PI_BASH_TIMEOUT_VAR = "PI_BASH_MAX_TIMEOUT_SEC"

# `--network none` even for `allowlist`: it only adds a unix-socket relay, so no interface offers a raw-IP route.
NETWORK_ARGS = ("--network", "none")
ENV_SOCKET_MOUNT = "/work/_env"
ENV_EXT_DIR = "/work/_ext"

_INCONTAINER_RELAY = str(Path(__file__).resolve().parent / "incontainer_relay.mjs")


def _require_docker_image(image: str) -> None:
    """Fail LOUDLY at construction if the sandbox image is not present locally."""
    # A missing image ends a session with an EMPTY answer indistinguishable from a wrong one; the tag is a local build.
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True, text=True, timeout=30,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"traverse sandbox='docker' needs the docker CLI, which is not on PATH: {exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"`docker image inspect {image}` timed out — is the docker daemon up?"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"sandbox image {image!r} is not present locally, so every terminal session "
            f"would die at startup and be scored as a WRONG answer. Build it first:\n"
            f"    scripts/setup/sandbox_image.sh\n"
            f"(docker said: {(proc.stderr or '').strip().splitlines()[-1] if proc.stderr.strip() else 'not found'})"
        )


def _parse_allowed_host(spec: str) -> tuple[str, int]:
    """Normalize one ``allowed_hosts`` entry to ``(host, port)``; raise on garbage."""
    # Ambiguous forms are rejected so a mistyped allowlist is a build error, not a malformed `--add-host`.
    spec = spec.strip()
    if not spec:
        raise ValueError("allowed_hosts entry is empty")
    if "//" in spec:
        return host_port_from_url(spec)
    if ":" in spec:
        host, _, port = spec.partition(":")
        if not host or not port.isdigit():
            raise ValueError(f"malformed allowed_hosts entry {spec!r} (want host or host:port)")
        return host, int(port)
    return spec, 443


def _pi_mono_root(pi_cli: Path) -> Path:
    """The pi-mono repo root to mount at /pi (the dir holding ``packages/``)."""
    p = pi_cli.resolve()
    for parent in p.parents:
        if (parent / "packages").is_dir():
            return parent
    return p.parents[3] if len(p.parents) > 3 else p.parent


def docker_run_argv(
    *,
    container_name: str,
    pi_cli: str,
    work: Path,
    corpus_host: Path | None,
    files_host: Path | None,
    pi_args: list[str],
    env: dict[str, str],
    image: str,
    provider: str,
    cmd_timeout: float,
    relay: EgressRelay | None = None,
    dbref_mounts: list[tuple[Path, str]] | None = None,
    env_socket_host: Path | None = None,
    ro_mounts: list[tuple[Path, str]] | None = None,
) -> list[str]:
    """Build the full `docker run` argv for one session (no side effects)."""
    # Pi sees ONLY the corpus (ro) and a fresh scratch (rw), so `find /` is bounded and `docker rm -f` reaps every child.
    pim = _pi_mono_root(Path(pi_cli))
    rel_cli = Path(pi_cli).resolve().relative_to(pim).as_posix()
    cmd = [
        "docker", "run", "--rm", "-i", "--name", container_name,
        *NETWORK_ARGS,
        "-v", f"{pim}:/pi:ro",
        "-v", f"{work}:/work:rw",
        "-w", "/work",
    ]
    if relay is not None:
        cmd += ["--init"]
        cmd += relay.add_host_args()
        cmd += ["-v", f"{relay.net_dir}:/net:rw"]
        cmd += ["-v", f"{_INCONTAINER_RELAY}:/relay/incontainer.mjs:ro"]
    if corpus_host is not None:
        cmd += ["-v", f"{corpus_host}:/work/corpus:ro"]
    # Reference dumps mount two levels below the corpus mount so `dbref/_index.md` stays unshadowed.
    for host_dir, db_name in dbref_mounts or []:
        cmd += ["-v", f"{host_dir}:/work/corpus/dbref/{db_name}/data:ro"]
    if files_host is not None:
        cmd += ["-v", f"{files_host}:/work/files:ro"]
    for host_dir, container_dir in ro_mounts or []:
        cmd += ["-v", f"{host_dir}:{container_dir}:ro"]
    if env_socket_host is not None:
        # A bind mount carries a unix socket, so the tool reaches the host-side simulator with the network off.
        cmd += ["-v", f"{env_socket_host}:{ENV_SOCKET_MOUNT}:rw"]
        cmd += ["-e", f"{SOCKET_ENV_VAR}={ENV_SOCKET_MOUNT}/{SOCKET_NAME}"]
    if env.get(CITATION_ENV_VAR):
        cmd += ["-e", CITATION_ENV_VAR]
    # Credentials go by NAME only (`-e VAR`): `VAR=secret` in argv would leak to `ps`.
    cred_vars = (
        ["OPENROUTER_API_KEY"]
        if provider == "openrouter"
        else ["OPENAI_API_KEY", "OPENAI_BASE_URL"]
    )
    for var in cred_vars:
        if env.get(var):
            cmd += ["-e", var]
    cmd += ["-e", f"{_PI_AGENT_DIR_VAR}=/work/{_PI_AGENT_DIR_NAME}"]
    cmd += ["-e", f"{_PI_BASH_TIMEOUT_VAR}={cmd_timeout:g}"]
    pi_entry = [
        "node", f"/pi/{rel_cli}",
        *pi_args, "--system-prompt", "/work/_system_prompt.txt",
    ]
    if relay is not None:
        # Start the loopback listeners, THEN exec Pi with stdio inherited so its RPC stream is untouched.
        cmd += [image, "node", "/relay/incontainer.mjs",
                "/net/_relay.json", "--", *pi_entry]
    else:
        cmd += [image, *pi_entry]
    return cmd


# The corpus is a read-only COPY, not a symlink, so the borrowed wiki cannot be touched; one snapshot per run is reused.


def copy_readonly(src: Path, dest: Path) -> Path:
    """Copy a directory tree to ``dest`` and strip write bits (read-only corpus)."""
    src = Path(src)
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest)
    _chmod_readonly(dest)
    return dest


def make_readonly(root: Path) -> None:
    """Strip write bits from a tree in place (public alias of the chmod pass)."""
    _chmod_readonly(Path(root))


def make_writable(root: Path) -> None:
    """Restore owner write bits across a tree (undo ``make_readonly``)."""
    # A SIGKILLed run skips the restore and needs a manual ``chmod -R u+w``.
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        try:
            st = os.stat(dirpath).st_mode
            os.chmod(dirpath, st | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRUSR)
        except OSError:
            pass
        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                os.chmod(p, os.stat(p).st_mode | stat.S_IWUSR | stat.S_IRUSR)
            except OSError:
                pass


def _chmod_readonly(root: Path) -> None:
    ro_file = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    ro_dir = ro_file | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            try:
                os.chmod(Path(dirpath) / fn, ro_file)
            except OSError:
                pass
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        try:
            os.chmod(dirpath, ro_dir)
        except OSError:
            pass


def prepare_session_dir(session_dir: Path, corpus_src: Path, *, corpus_name: str = "corpus") -> tuple[Path, Path]:
    """Create a writable ``session_dir`` with a read-only ``corpus/`` copy inside."""
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = copy_readonly(Path(corpus_src), session_dir / corpus_name)
    return session_dir, corpus_dir
