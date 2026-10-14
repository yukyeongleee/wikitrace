"""Terminal-agent harness contract and implementations, domain-neutral."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
import uuid
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Optional

from wikitrace.harness.environment.protocol import EPISODE_OVER, request
from wikitrace.harness.environment.tool import CITATION_ENV_VAR, SOCKET_ENV_VAR, SOCKET_NAME, EnvTool
from wikitrace.harness.terminal.docker import (
    ENV_EXT_DIR,
    _PI_BASH_TIMEOUT_VAR,
    _parse_allowed_host,
    _require_docker_image,
    docker_run_argv,
)
from wikitrace.harness.terminal.egress import EgressRelay, host_port_from_url
from wikitrace.harness.terminal.rpc import _readline_by
from wikitrace.harness.terminal.usage import (
    PI_MAX_OUTPUT_TOKENS_CEILING,
    _sum_usage,
    _write_pi_agent_dir,
    collect_call_usage,
    collect_session_stats,
)


@dataclass
class TerminalResult:
    """The outcome of one terminal-agent session (schema-agnostic)."""
    # None when the agent wrote no sentinel or garbage; the caller then uses ``final_text``.

    sentinel: Optional[dict] = None
    sentinel_text: str = ""
    final_text: str = ""
    final_reasoning: str = ""
    session_text: str = ""
    turns: list[dict] = field(default_factory=list)
    read_set: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    usage_calls: list[dict[str, Any]] = field(default_factory=list)
    # Non-empty when the session was CUT SHORT; a generation still streaming at the kill was billed but never reported.
    stopped_by: str = ""
    # `stopReason` of the LAST assistant message and whether it carried a tool call; "length" without one is fatal.
    last_stop_reason: str = ""
    last_had_tool_call: bool = False
    truncated_calls: int = 0
    model_returned: str = ""
    final_context: Any = None
    citation_text: str = ""
    citation: Optional[dict] = None
    # The provider's error on the session's last request when Pi's retries did not clear it: a harness outcome, not the agent's.
    provider_error: str = ""
    provider_retries: int = 0


_OWNED_WORK_DIRS = ("/work/corpus", "/work/files", "/work/_ext")


def readonly_groups(files: Mapping[str, Path]) -> dict[str, dict[str, Path]]:
    """Group read-only files by their container directory, refusing a path outside ``/work``."""
    groups: dict[str, dict[str, Path]] = {}
    for path, src in files.items():
        p = PurePosixPath(path)
        directory = str(p.parent)
        if (not p.is_absolute() or ".." in p.parts or p.parts[:2] != ("/", "work") or len(p.parts) < 4
                or any(directory == d or directory.startswith(d + "/") for d in _OWNED_WORK_DIRS)):
            raise ValueError(f"read-only file {path!r} must be an absolute path in a new directory "
                             f"under /work, without ..")
        if not Path(src).is_file():
            raise FileNotFoundError(f"read-only file source {src} does not exist")
        groups.setdefault(directory, {})[p.name] = Path(src)
    return groups


class TraverseRunner(ABC):
    """Runs one terminal-agent session and returns a :class:`TerminalResult`."""

    @abstractmethod
    def run_session(
        self,
        *,
        system_prompt: str,
        user_msg: str,
        corpus_dir: Path | None,
        out_dir: Path,
        tools: str = "read,bash",
        max_turns: int = 50,
        sentinel_name: str = "__out.json",
        label: str = "",
        files_dir: Path | None = None,
        dbref_dirs: list[tuple[Path, str]] | None = None,
        env_tool: EnvTool | None = None,
        citation_prompt: str | None = None,
        citation_layers: tuple[str, ...] = (),
        readonly_files: Mapping[str, Path] | None = None,
    ) -> TerminalResult:
        """Run the agent over ``corpus_dir`` (read-only), writing its sentinel into ``out_dir``."""
        raise NotImplementedError


class FakeTraverseRunner(TraverseRunner):
    """Offline runner: delegates to a handler that returns a TerminalResult."""

    def __init__(self, handler: Callable[..., TerminalResult]) -> None:
        self._handler = handler

    def run_session(
        self,
        *,
        system_prompt: str,
        user_msg: str,
        corpus_dir: Path | None,
        out_dir: Path,
        tools: str = "read,bash",
        max_turns: int = 50,
        sentinel_name: str = "__out.json",
        label: str = "",
        files_dir: Path | None = None,
        dbref_dirs: list[tuple[Path, str]] | None = None,
        env_tool: EnvTool | None = None,
        citation_prompt: str | None = None,
        citation_layers: tuple[str, ...] = (),
        readonly_files: Mapping[str, Path] | None = None,
    ) -> TerminalResult:
        result = self._handler(
            system_prompt=system_prompt,
            user_msg=user_msg,
            corpus_dir=corpus_dir,
            out_dir=out_dir,
            tools=tools,
            max_turns=max_turns,
            sentinel_name=sentinel_name,
            label=label,
            files_dir=files_dir,
            dbref_dirs=dbref_dirs,
            env_tool=env_tool,
            citation_prompt=citation_prompt,
            citation_layers=citation_layers,
            readonly_files=readonly_files,
        )
        if result.sentinel is not None and not result.sentinel_text:
            out_dir.mkdir(parents=True, exist_ok=True)
            text = json.dumps(result.sentinel, ensure_ascii=False)
            (out_dir / sentinel_name).write_text(text, encoding="utf-8")
            result.sentinel_text = text
        return result


# Content blocks meaning "the agent asked for a tool": a `length` truncation that still emitted one keeps the loop running.
_TOOL_CALL_BLOCKS = ("toolCall", "tool_call", "toolUse", "tool_use")

# How far the runner's guard sits ABOVE Pi's command ceiling; overshoot means Pi's kill did not happen.
_CMD_BACKSTOP_SLACK = 30.0
# Opening of the pi-patch 0004 tool-result message, matched as TEXT since nothing else tells a force-kill from a failure.
_HARNESS_KILL_MARKER = "[harness] Command force-killed after"
_PROVIDER_RETRY_GRACE = 3.0

_CITATION_EXTRA_TURNS = 2
_PROMPT_RESENDS = 50
_PROMPT_RESEND_DELAY = 0.2


def _is_harness_kill(event: dict) -> bool:
    """Is this ``tool_execution_end`` pi-patch 0004's per-command force-kill?"""
    if not event.get("isError"):
        return False
    result = event.get("result")
    if not isinstance(result, dict):
        return False
    for block in result.get("content") or []:
        if isinstance(block, dict) and _HARNESS_KILL_MARKER in str(block.get("text") or ""):
            return True
    return False


def _is_episode_over(event: dict, tool: str = "act") -> bool:
    """Is this ``tool_execution_end`` the environment tool reporting the episode over?"""
    # Only the environment tool's reply counts: a `bash` or `exec` output can end with the same words.
    result = event.get("result")
    if event.get("toolName") != tool or event.get("isError") or not isinstance(result, dict):
        return False
    return any(
        isinstance(block, dict) and isinstance(block.get("text"), str)
        and (block["text"] == EPISODE_OVER or block["text"].endswith(f"\n\n{EPISODE_OVER}"))
        for block in result.get("content") or []
    )


class PiTraverseRunner(TraverseRunner):
    """Drive the Pi coding-agent CLI (``cli.js --mode rpc``) as a subprocess."""
    # Defaults must equal ``TraverseConfig``'s (a test pins it), or a divergent provider forwards the wrong key.

    def __init__(
        self,
        *,
        pi_cli: str,
        model: str,
        provider: str = "openrouter",
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: str | None = None,
        node_bin: str = "node",
        extra_args: Optional[list[str]] = None,
        idle_timeout: float = 120.0,
        max_seconds: float = 600.0,
        max_cmd_kills: int = 5,
        # Output budget per provider call, reasoning included; a decoder stop, not a thinking target.
        max_output_tokens: int = 16384,
        cmd_timeout: float = 90.0,
        sandbox: str = "none",
        docker_image: str = "wikitrace-sandbox:node20",
        network_policy: str = "allowlist",
        allowed_hosts: Optional[list[str]] = None,
        openrouter_providers: Optional[list[str]] = None,
    ) -> None:
        self.sandbox = sandbox
        self.docker_image = docker_image
        self.openrouter_providers = list(openrouter_providers or [])
        if sandbox == "docker":
            _require_docker_image(docker_image)
        if network_policy not in ("allowlist", "deny"):
            raise ValueError(f"unknown traverse network_policy: {network_policy!r}")
        self.network_policy = network_policy
        self.allowed_hosts: list[tuple[str, int]] = [
            _parse_allowed_host(h) for h in (allowed_hosts or [])
        ]
        if sandbox != "docker":
            # HOST MODE IS UNCONTAINED: network_policy is enforced only in docker; warned, not raised, for hosts without docker.
            warnings.warn(
                f"traverse sandbox={sandbox!r} runs Pi directly on the host: "
                f"network_policy={network_policy!r} is NOT enforced and the agent has "
                "full network + host filesystem access. Use sandbox='docker' for an "
                "isolated run.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.pi_cli = pi_cli
        self.model = model
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.node_bin = node_bin
        self.extra_args = list(extra_args or [])
        self.max_output_tokens = min(max_output_tokens, PI_MAX_OUTPUT_TOKENS_CEILING)
        self.idle_timeout = idle_timeout
        self.max_seconds = max_seconds
        # Per-command cap, ENFORCED BY PI (``PI_BASH_MAX_TIMEOUT_SEC``): it kills the process tree and the session lives.
        self.cmd_timeout = cmd_timeout
        self.cmd_backstop = cmd_timeout + _CMD_BACKSTOP_SLACK
        # Force-kills allowed per session; each costs ``cmd_timeout`` of wall clock, which ``max_seconds`` must cover.
        self.max_cmd_kills = max_cmd_kills

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env[_PI_BASH_TIMEOUT_VAR] = f"{self.cmd_timeout:g}"
        if self.provider == "openrouter":
            if self.api_key:
                env["OPENROUTER_API_KEY"] = self.api_key
        else:
            env["OPENAI_BASE_URL"] = self.base_url
            if self.api_key:
                env["OPENAI_API_KEY"] = self.api_key
        return env

    def _llm_host_port(self) -> tuple[str, int]:
        """The LLM endpoint host:port Pi actually dials -- always allowlisted."""
        if self.provider == "openrouter":
            return "openrouter.ai", 443
        return host_port_from_url(self.base_url)

    def _docker_cmd(
        self,
        *,
        container_name: str,
        pi_cli: str,
        work: Path,
        corpus_host: Path | None,
        files_host: Path | None,
        pi_args: list[str],
        env: dict[str, str],
        relay: EgressRelay | None = None,
        dbref_mounts: list[tuple[Path, str]] | None = None,
        env_socket_host: Path | None = None,
        ro_mounts: list[tuple[Path, str]] | None = None,
    ) -> list[str]:
        """The `docker run` argv for one session, with this runner's sandbox settings filled in."""
        return docker_run_argv(
            container_name=container_name, pi_cli=pi_cli, work=work,
            corpus_host=corpus_host, files_host=files_host, pi_args=pi_args, env=env,
            relay=relay, dbref_mounts=dbref_mounts,
            image=self.docker_image, provider=self.provider, cmd_timeout=self.cmd_timeout,
            env_socket_host=env_socket_host, ro_mounts=ro_mounts,
        )

    def run_session(
        self,
        *,
        system_prompt: str,
        user_msg: str,
        corpus_dir: Path | None,
        out_dir: Path,
        tools: str = "read,bash",
        max_turns: int = 50,
        sentinel_name: str = "__out.json",
        label: str = "",
        files_dir: Path | None = None,
        dbref_dirs: list[tuple[Path, str]] | None = None,
        env_tool: EnvTool | None = None,
        citation_prompt: str | None = None,
        citation_layers: tuple[str, ...] = (),
        readonly_files: Mapping[str, Path] | None = None,
    ) -> TerminalResult:
        from wikitrace.harness.terminal.sentinel import read_sentinel
        from wikitrace.harness.terminal.turns import (
            event_to_turn,
            parse_citation,
            parse_read_set,
        )

        import shutil
        import tempfile

        out_dir.mkdir(parents=True, exist_ok=True)
        pi_cli = str(Path(self.pi_cli).resolve())
        corpus_host = Path(corpus_dir).resolve() if corpus_dir else None
        files_host = Path(files_dir).resolve() if files_dir else None
        dbref_mounts = [(Path(d).resolve(), name) for d, name in (dbref_dirs or [])]
        work = Path(tempfile.mkdtemp(prefix="pi-traverse-"))
        sys_file = work / "_system_prompt.txt"
        sys_file.write_text(system_prompt, encoding="utf-8")
        _write_pi_agent_dir(work, self.model, self.max_output_tokens, self.openrouter_providers)
        (work / "files").mkdir(exist_ok=True)
        ro_groups = readonly_groups(readonly_files or {})

        pi_args = [
            "--mode", "rpc", "--provider", self.provider, "--model", self.model,
            "--tools", tools, "--no-session", *self.extra_args,
        ]
        env_socket_host: Path | None = None
        env_extra: dict[str, str] = {}
        if env_tool is not None:
            ext_dir = work / "_ext"
            ext_dir.mkdir()
            shutil.copy(env_tool.extension, ext_dir / env_tool.extension.name)
            env_socket_host = Path(env_tool.socket_dir).resolve()
            if self.sandbox == "docker":
                pi_args += ["--extension", f"{ENV_EXT_DIR}/{env_tool.extension.name}"]
            else:
                pi_args += ["--extension", str(ext_dir / env_tool.extension.name)]
                env_extra[SOCKET_ENV_VAR] = str(env_socket_host / SOCKET_NAME)
            if citation_prompt is not None:
                env_extra[CITATION_ENV_VAR] = citation_prompt
        container_name = ""
        relay: EgressRelay | None = None
        net_dir: Path | None = None
        ro_root: Path | None = None
        # Setup runs BEFORE the try/finally that owns cleanup, so a failure here must release the relay and temp dirs.
        try:
            ro_mounts: list[tuple[Path, str]] = []
            if ro_groups:
                ro_root = Path(tempfile.mkdtemp(prefix="pi-readonly-"))
                for i, (directory, files) in enumerate(sorted(ro_groups.items())):
                    host = ro_root / str(i)
                    host.mkdir()
                    for name, src in files.items():
                        shutil.copyfile(src, host / name)
                        (host / name).chmod(0o755)
                    ro_mounts.append((host, directory))
                    (work / directory.removeprefix("/work/")).mkdir(parents=True, exist_ok=True)
            if self.sandbox == "docker":
                container_name = f"pi-traverse-{uuid.uuid4().hex[:12]}"
                env = {**self._env(), **env_extra}
                if self.network_policy == "allowlist":
                    net_dir = Path(tempfile.mkdtemp(prefix="pi-egress-"))
                    hosts = [self._llm_host_port(), *self.allowed_hosts]
                    relay = EgressRelay(net_dir, hosts)
                    relay.start()
                    (net_dir / "_relay.json").write_text(
                        json.dumps(relay.relay_config()), encoding="utf-8"
                    )
                cmd = self._docker_cmd(
                    container_name=container_name, pi_cli=pi_cli, work=work,
                    corpus_host=corpus_host, files_host=files_host,
                    pi_args=pi_args, env=env, relay=relay,
                    dbref_mounts=dbref_mounts, env_socket_host=env_socket_host,
                    ro_mounts=ro_mounts,
                )
            else:
                # Host mode: NO FS isolation or egress restriction; the corpus symlink points at this run's frozen COPY.
                if corpus_host is not None:
                    (work / "corpus").symlink_to(corpus_host, target_is_directory=True)
                if files_host is not None:
                    # Task files point at the SHARED dataset, so they are COPIED: `rm -rf files/*` must not corrupt it.
                    shutil.copytree(files_host, work / "files", dirs_exist_ok=True)
                for host, directory in ro_mounts:
                    target = work / directory.removeprefix("/work/")
                    target.rmdir()
                    target.symlink_to(host, target_is_directory=True)
                cmd = [self.node_bin, pi_cli, *pi_args, "--system-prompt", str(sys_file)]
                env = {**self._env(), **env_extra}
            proc = subprocess.Popen(
                cmd, cwd=str(work), env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True,  # own process group so we can kill the whole tree
            )
        except BaseException:
            if relay is not None:
                relay.stop()
            if net_dir is not None:
                shutil.rmtree(net_dir, ignore_errors=True)
            if ro_root is not None:
                shutil.rmtree(ro_root, ignore_errors=True)
            shutil.rmtree(work, ignore_errors=True)
            raise

        stderr_chunks: list[str] = []

        def _drain() -> None:
            assert proc.stderr is not None
            for raw in proc.stderr:
                stderr_chunks.append(raw.decode("utf-8", errors="replace"))

        drain = threading.Thread(target=_drain, daemon=True)
        drain.start()

        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write((json.dumps({"id": "py-1", "type": "prompt", "message": user_msg}) + "\n").encode())
        proc.stdin.flush()

        turns: list[dict] = []
        text_parts: list[str] = []
        msg_start = 0            # where in text_parts the assistant message in progress began
        final_text = ""
        final_reasoning = ""
        read_set: list[str] = []
        usage: dict[str, Any] = {}
        usage_calls: list[dict[str, Any]] = []
        model_returned = ""
        seen_turns = 0
        aborted_at = 0     # seen_turns when the last abort was sent (0 = never)
        evbuf = bytearray()
        over_cap = False  # past max_turns, waiting for a safe window to abort in
        episode_over = False  # the environment tool answered EPISODE_OVER; same window
        citation_parts: list[str] = []
        citation_open = False
        citation_sent_at = -1   # seen_turns when it was sent (-1 = never)
        citation_text = ""
        citation_misses: list[str] = []  # response ids of citation-turn answers with no text
        context_frozen = False  # the citation request's context is the last one kept
        turn_idx = 0

        def _send_abort() -> None:
            nonlocal aborted_at
            proc.stdin.write((json.dumps({"id": "py-abort", "type": "abort"}) + "\n").encode())
            proc.stdin.flush()
            aborted_at = seen_turns

        pending_prompt: dict[str, Any] = {}
        prompt_resends = 0

        def _send_prompt(text: str) -> None:
            nonlocal pending_prompt, prompt_resends
            pending_prompt = {"id": f"py-message-{turn_idx}", "type": "prompt", "message": text}
            prompt_resends = 0
            proc.stdin.write((json.dumps(pending_prompt) + "\n").encode())
            proc.stdin.flush()

        def _deliver_message() -> bool:
            """Send the last reply to the user; True when the session goes on with Pi."""
            nonlocal turn_idx, episode_over, citation_open, citation_sent_at
            turns.append({"turn": turn_idx, "type": "message", "role": "assistant",
                          "content": final_text, "reasoning": final_reasoning})
            turn_idx += 1
            try:
                reply = request(str(env_socket_host / SOCKET_NAME), {"op": "message", "text": final_text})
            except (OSError, ValueError):
                return False
            if "error" in reply:
                return False
            turns.append({"turn": turn_idx, "type": "message", "role": "user",
                          "content": str(reply.get("observation", ""))})
            turn_idx += 1
            if not reply.get("done"):
                _send_prompt(str(reply.get("observation", "")))
                return True
            turns.append({"turn": turn_idx, "type": "episode_over"})
            turn_idx += 1
            episode_over = True
            if citation_prompt is None:
                return False
            citation_open = True
            citation_sent_at = seen_turns
            _send_prompt(citation_prompt)
            return True

        def _close_citation(*, abort: bool, response_id: str = "") -> None:
            nonlocal citation_text, citation_open, turn_idx
            citation_text = "".join(citation_parts)
            turns.append({
                "turn": turn_idx, "type": "citation", "text": citation_text,
                "parsed": parse_citation(citation_text, citation_layers),
                "response_id": response_id,
                **({"missed_response_ids": list(citation_misses)} if citation_misses else {}),
            })
            turn_idx += 1
            citation_open = False
            if abort:
                _send_abort()
        stopped_by = ""
        last_stop_reason = ""
        last_had_tool_call = False
        truncated_calls = 0
        final_context: Any = None
        deadline = time.monotonic() + self.max_seconds
        tool_started_at: float | None = None
        cmd_kills = 0
        provider_error = ""
        provider_retries = 0
        retry_grace_until: float | None = None  # set while waiting for Pi's auto_retry_start
        try:
            while True:
                now = time.monotonic()
                reason = None
                if now >= deadline:
                    reason = "wall"
                elif tool_started_at is not None and now - tool_started_at > self.cmd_backstop:
                    reason = "command"
                raw: bytes | None = b""
                if reason is None:
                    budget = max(0.0, deadline - now)
                    if tool_started_at is not None:
                        budget = min(budget, self.cmd_backstop - (now - tool_started_at))
                    budget = min(self.idle_timeout, budget)
                    if retry_grace_until is not None:
                        budget = min(budget, retry_grace_until - now)
                    # Buffer-aware and deadline-bounded: a bare select() cannot see events already in the BufferedReader.
                    raw = _readline_by(proc.stdout, evbuf, now + max(0.0, budget))
                    if raw is None and retry_grace_until is not None and time.monotonic() >= retry_grace_until:
                        break  # Pi did not retry the failed request: the session is over
                    if raw is None:
                        now2 = time.monotonic()
                        reason = (
                            "wall" if now2 >= deadline
                            else "command" if tool_started_at is not None
                            and now2 - tool_started_at > self.cmd_backstop
                            else "idle"
                        )
                if reason is not None:
                    turns.append({"turn": turn_idx, "type": "timeout", "reason": reason})
                    stopped_by = reason
                    break
                if not raw:
                    # EOF without agent_end: Pi died; a streaming generation was billed and never reported.
                    stopped_by = "eof"
                    break
                try:
                    event = json.loads(raw.decode("utf-8").rstrip("\r\n"))
                except Exception:
                    continue
                if not isinstance(event, dict):
                    continue
                etype = event.get("type")
                if etype == "provider_request_context":
                    if not context_frozen:
                        final_context = event.get("messages")
                        context_frozen = citation_open
                elif etype == "turn_start":
                    seen_turns += 1
                    if seen_turns > max_turns + 3:
                        turns.append({
                            "turn": turn_idx, "type": "timeout", "reason": "turn_cap",
                        })
                        stopped_by = "turn_cap"
                        break
                    # Do NOT abort here: Pi dispatches right after `turn_start`, so it would cancel a billed, unrecorded stream.
                    over_cap = seen_turns > max_turns
                elif etype == "message_update":
                    ev = event.get("assistantMessageEvent", {})
                    if ev.get("type") == "text_delta":
                        target = citation_parts if citation_open else text_parts
                        target.append(ev.get("delta", ""))
                elif etype == "message_end":
                    msg = event.get("message") or {}
                    if citation_open and msg.get("role") == "assistant":
                        rid = str(msg.get("responseId") or "")
                        if "".join(citation_parts).strip() or len(citation_misses) >= _CITATION_EXTRA_TURNS:
                            _close_citation(abort=True, response_id=rid)
                        else:
                            citation_misses.append(rid)
                    if citation_sent_at >= 0:
                        continue
                    # The ONLY place a truncation is visible: logged usage normalises a `length` end to `tool_calls`.
                    last_stop_reason = str(msg.get("stopReason") or "")
                    provider_error = (
                        str(msg.get("errorMessage") or "unknown provider error")
                        if last_stop_reason == "error" else ""
                    )
                    content = msg.get("content")
                    blocks = [b for b in (content if isinstance(content, list) else []) if isinstance(b, dict)]
                    last_had_tool_call = any(b.get("type") in _TOOL_CALL_BLOCKS for b in blocks)
                    if last_stop_reason == "length":
                        truncated_calls += 1
                    if msg.get("role") == "assistant":
                        final_text = (
                            "".join(str(b.get("text") or "") for b in blocks if b.get("type") == "text")
                            if blocks else "".join(text_parts[msg_start:])
                        )
                        final_reasoning = "".join(
                            str(b.get("thinking") or "") for b in blocks if b.get("type") == "thinking"
                        )
                        msg_start = len(text_parts)
                elif etype in ("tool_execution_start", "tool_execution_end"):
                    tool_started_at = now if etype == "tool_execution_start" else None
                    if citation_sent_at >= 0 and seen_turns > citation_sent_at:
                        continue
                    if env_tool is not None and etype == "tool_execution_end" and _is_episode_over(event, env_tool.name):
                        # The environment ended, so nothing the agent does next can score: abort now (the tool-start window re-arms).
                        turns.append({"turn": turn_idx, "type": "episode_over"})
                        turn_idx += 1
                        episode_over = True
                        if citation_prompt is None:
                            _send_abort()
                        elif citation_sent_at < 0:
                            # The extension queued the citation steer for this turn's end, so the cache still holds the episode; armed once.
                            citation_open = True
                            citation_sent_at = seen_turns
                    if etype == "tool_execution_end" and _is_harness_kill(event):
                        cmd_kills += 1
                        turns.append({
                            "turn": turn_idx, "type": "cmd_kill",
                            "n": cmd_kills, "limit": self.max_cmd_kills,
                        })
                        turn_idx += 1
                        if cmd_kills > self.max_cmd_kills:
                            stopped_by = "cmd_kill_budget"
                            break
                        if cmd_kills == self.max_cmd_kills:
                            proc.stdin.write((json.dumps({
                                "id": "py-kill-budget", "type": "steer",
                                "message": (
                                    f"[harness] Command-kill budget exhausted "
                                    f"({cmd_kills} of {self.max_cmd_kills} commands "
                                    f"force-killed). The session ends at the next one. "
                                    f"Stop computing and write your final answer to "
                                    f"{sentinel_name} now, using what you already have."
                                ),
                            }) + "\n").encode())
                            proc.stdin.flush()
                    if etype == "tool_execution_start":
                        turns.append(event_to_turn(turn_idx, event))
                        turn_idx += 1
                        # THE window to abort in; RE-ARMED, not latched, so a dropped abort cannot make the cap advisory.
                        if (over_cap or episode_over) and seen_turns > aborted_at and not citation_open:
                            _send_abort()
                    read_set.extend(parse_read_set(event, corpus_dir))
                elif etype in ("compaction_start", "compaction_end"):
                    turns.append({
                        "turn": turn_idx,
                        "type": etype,
                        "reason": event.get("reason"),
                        "aborted": event.get("aborted"),
                        "result": event.get("result"),
                    })
                    turn_idx += 1
                elif etype == "agent_end":
                    if last_stop_reason != "error" or over_cap or episode_over:
                        if (
                            env_tool is not None and env_tool.messages and not over_cap
                            and not episode_over and last_stop_reason == "stop"
                            and final_text.strip() and _deliver_message()
                        ):
                            continue
                        break
                    retry_grace_until = time.monotonic() + _PROVIDER_RETRY_GRACE
                elif (
                    etype == "response" and pending_prompt and event.get("id") == pending_prompt["id"]
                    and event.get("success") is False and prompt_resends < _PROMPT_RESENDS
                ):
                    prompt_resends += 1
                    time.sleep(_PROMPT_RESEND_DELAY)
                    proc.stdin.write((json.dumps(pending_prompt) + "\n").encode())
                    proc.stdin.flush()
                elif etype == "auto_retry_start":
                    retry_grace_until = None
                    provider_retries += 1
                    seen_turns -= 1
                    turns.append({
                        "turn": turn_idx, "type": "provider_retry",
                        "attempt": event.get("attempt"), "max_attempts": event.get("maxAttempts"),
                        "delay_ms": event.get("delayMs"), "error": event.get("errorMessage"),
                    })
                    turn_idx += 1
                elif etype == "auto_retry_end" and not event.get("success"):
                    provider_error = str(event.get("finalError") or provider_error)
                    break

            if citation_open:
                _close_citation(abort=False)

            leash = 3.0 if reason else 15.0
            usage_calls = collect_call_usage(proc, timeout=leash * 2, buf=evbuf)
            if usage_calls:
                usage = _sum_usage(usage_calls)
                model_returned = usage_calls[-1].get("model_returned", "")
            else:
                usage = collect_session_stats(proc, timeout=leash, buf=evbuf)

            if len(text_parts) > msg_start and citation_sent_at < 0:
                final_text, final_reasoning = "".join(text_parts[msg_start:]), ""
            if final_context is not None:
                (work / "_final_context.json").write_text(
                    json.dumps(final_context, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            sentinel, sentinel_text = read_sentinel(work, sentinel_name)
            for fn in (sentinel_name, "_system_prompt.txt", "_final_context.json"):
                src = work / fn
                if src.exists():
                    shutil.move(str(src), str(out_dir / fn))
            read_set = list(dict.fromkeys(read_set))
            return TerminalResult(
                sentinel=sentinel, sentinel_text=sentinel_text, final_text=final_text,
                final_reasoning=final_reasoning, session_text="".join(text_parts),
                turns=turns, read_set=read_set, usage=usage, usage_calls=usage_calls,
                stopped_by=stopped_by, model_returned=model_returned,
                final_context=final_context,
                last_stop_reason=last_stop_reason,
                last_had_tool_call=last_had_tool_call,
                truncated_calls=truncated_calls,
                citation_text=citation_text,
                citation=next(
                    (t["parsed"] for t in reversed(turns) if t["type"] == "citation"), None,
                ),
                provider_error=provider_error if last_stop_reason == "error" and not episode_over else "",
                provider_retries=provider_retries,
            )
        finally:
            # ALWAYS, even on exception: kill the whole process group (a wedged child outlives `terminate`) and delete the scratch.
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                try:
                    proc.wait(timeout=3)
                except Exception:
                    os.killpg(pgid, signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            if container_name:
                try:
                    subprocess.run(["docker", "rm", "-f", container_name],
                                   capture_output=True, timeout=15)
                except Exception:
                    pass
            drain.join(timeout=2)
            if stderr_chunks:
                try:
                    (out_dir / "_pi_stderr.log").write_text("".join(stderr_chunks), encoding="utf-8")
                except Exception:
                    pass
            # Tear the egress relay down AFTER the container is reaped, so no in-flight LLM call is cut early.
            if relay is not None:
                relay.stop()
            if net_dir is not None:
                shutil.rmtree(net_dir, ignore_errors=True)
            if ro_root is not None:
                shutil.rmtree(ro_root, ignore_errors=True)
            shutil.rmtree(work, ignore_errors=True)
