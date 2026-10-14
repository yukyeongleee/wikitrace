"""Unit tests for the domain-neutral traverse harness layer."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from wikitrace.harness.terminal import (
    FakeTraverseRunner,
    PiTraverseRunner,
    TerminalResult,
    extract_json_fallback,
    prepare_session_dir,
    read_sentinel,
)
import wikitrace.harness.terminal.runner as runner_mod
from wikitrace.config import write_turns


@pytest.fixture
def docker_image_available(monkeypatch):
    """Command assembly needs no daemon; image validation has separate gate tests."""
    monkeypatch.setattr(runner_mod, "_require_docker_image", lambda image: None)


def test_sentinel_read_and_fallback(tmp_path: Path):
    (tmp_path / "__out.json").write_text('{"final_sql": "SELECT 1"}', encoding="utf-8")
    obj, text = read_sentinel(tmp_path, "__out.json")
    assert obj == {"final_sql": "SELECT 1"}
    # missing file
    assert read_sentinel(tmp_path, "nope.json") == (None, "")
    # fenced fallback
    assert extract_json_fallback('pre ```json\n{"a": 1}\n``` post') == {"a": 1}
    # last balanced object wins
    assert extract_json_fallback('{"a":1} noise {"b":2}') == {"b": 2}
    assert extract_json_fallback("no json here") is None


def test_sandbox_readonly_corpus_writable_scratch(tmp_path: Path):
    corpus = tmp_path / "wiki" / "knowledge"
    corpus.mkdir(parents=True)
    doc = tmp_path / "wiki" / "knowledge" / "document-0000.md"
    doc.write_text("hi", encoding="utf-8")

    session, corpus_dir = prepare_session_dir(tmp_path / "sess", tmp_path / "wiki")
    copied = corpus_dir / "knowledge" / "document-0000.md"
    assert copied.exists()
    # corpus copy is read-only WHILE the agent greps it; scratch is writable
    assert not os.access(copied, os.W_OK)
    # make_writable undoes the freeze so cleanup needs no manual chmod
    from wikitrace.harness.terminal.docker import make_writable

    make_writable(corpus_dir)
    assert os.access(copied, os.W_OK)
    assert os.access(corpus_dir / "knowledge", os.W_OK)
    (session / "__out.json").write_text("{}", encoding="utf-8")
    assert (session / "__out.json").exists()


def test_fake_runner_persists_sentinel(tmp_path: Path):
    def handler(**kw):
        return TerminalResult(sentinel={"final_sql": "SELECT 2"})

    runner = FakeTraverseRunner(handler)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    res = runner.run_session(
        system_prompt="sys", user_msg="u", corpus_dir=tmp_path,
        out_dir=out_dir, label="decomposer",
    )
    assert res.sentinel == {"final_sql": "SELECT 2"}
    # handler-provided sentinel is mirrored to disk so file-based parsing works
    obj, _ = read_sentinel(out_dir, "__out.json")
    assert obj == {"final_sql": "SELECT 2"}


def _fake_pi_cli(tmp_path: Path) -> Path:
    """A pi-mono-shaped tree so _pi_mono_root/relative_to resolve (no Pi needed)."""
    cli = tmp_path / "pi-mono" / "packages" / "coding-agent" / "dist" / "cli.js"
    cli.parent.mkdir(parents=True)
    cli.write_text("// stub", encoding="utf-8")
    return cli


def _docker_argv(tmp_path: Path, **kw) -> list[str]:
    """Build one session's docker argv — pure argv assembly, no daemon, no subprocess."""
    runner = PiTraverseRunner(
        pi_cli=str(_fake_pi_cli(tmp_path)), model="deepseek/deepseek-v4-flash",
        sandbox="docker", **kw,
    )
    return runner._docker_cmd(
        container_name="pi-traverse-deadbeef", pi_cli=str(_fake_pi_cli(tmp_path / "b")),
        work=tmp_path / "work", corpus_host=tmp_path / "corpus", files_host=None,
        pi_args=["--mode", "rpc"], env={"OPENROUTER_API_KEY": "sk-xxx"},
    )


@pytest.mark.usefixtures("docker_image_available")
def test_dbref_dumps_mount_read_only_under_the_corpus(tmp_path: Path):
    # The reference dumps are hundreds of MB and identical across runs, so they are
    # bind-mounted ONCE from the shared install. Each lands two levels below the corpus
    # mount, so the corpus's own index and READMEs stay visible.
    runner = PiTraverseRunner(
        pi_cli=str(_fake_pi_cli(tmp_path)), model="m", sandbox="docker",
    )
    argv = runner._docker_cmd(
        container_name="c", pi_cli=str(_fake_pi_cli(tmp_path / "b")),
        work=tmp_path / "work", corpus_host=tmp_path / "corpus", files_host=None,
        pi_args=[], env={},
        dbref_mounts=[(tmp_path / "dumps" / "clinvar", "clinvar")],
    )
    assert f"{tmp_path / 'dumps' / 'clinvar'}:/work/corpus/dbref/clinvar/data:ro" in argv
    # read-only, and nested strictly deeper than the corpus mount
    corpus_mnt = f"{tmp_path / 'corpus'}:/work/corpus:ro"
    assert corpus_mnt in argv
    assert argv.index(corpus_mnt) < argv.index(
        f"{tmp_path / 'dumps' / 'clinvar'}:/work/corpus/dbref/clinvar/data:ro"
    )


@pytest.mark.usefixtures("docker_image_available")
def test_no_dbref_mounts_by_default(tmp_path: Path):
    # Every non-DbQA suite (BIRD, seqqa, protocolqa) mounts no dumps at all.
    argv = _docker_argv(tmp_path)
    assert not any("/work/corpus/dbref/" in a for a in argv)


@pytest.mark.usefixtures("docker_image_available")
def test_docker_sandbox_is_always_network_none(tmp_path: Path):
    # The container is ALWAYS `--network none` (no NIC, loopback only) regardless
    # of policy — allowlist opens a unix-socket relay, never the network. That is
    # what makes raw-IP bypass impossible: no interface to dial a raw IP from.
    argv = _docker_argv(tmp_path, network_policy="deny")
    assert "--network" in argv
    assert argv[argv.index("--network") + 1] == "none"
    # ...and it must be a `docker run` FLAG, not something after the image name.
    assert argv.index("--network") < argv.index("wikitrace-sandbox:node20")
    # the credential is still forwarded by NAME only — never `VAR=secret` in argv
    assert "-e" in argv and "OPENROUTER_API_KEY" in argv
    assert not any("sk-xxx" in a for a in argv)


def test_docker_image_default_is_the_bio_sandbox_and_defaults_agree():
    # The runtime image comes from the config, which the runner threads through; the
    # runner's own parameter default is only for direct construction. Both are pinned,
    # and that they AGREE, so the authoritative one cannot drift alone.
    import inspect

    from wikitrace.config import TraverseConfig

    prod_default = TraverseConfig().docker_image
    assert prod_default == "wikitrace-sandbox:node20"
    fallback = inspect.signature(PiTraverseRunner.__init__).parameters["docker_image"].default
    assert fallback == prod_default


def test_network_policy_allowlist_is_the_config_default():
    # Allowlist is the default because Pi calls the LLM from inside the container,
    # so a live traversal cannot run under deny (fully offline). See TraverseConfig.
    from wikitrace.config import TraverseConfig

    assert TraverseConfig().network_policy == "allowlist"


@pytest.mark.usefixtures("docker_image_available")
def test_pi_runner_provider_default_matches_traverse_config(tmp_path: Path, monkeypatch):
    # The two provider defaults are pinned EQUAL: a runner built without an explicit
    # provider would otherwise forward one set of credentials and drop the other,
    # landing on a provider whose tool-less turns are a traversal that never happens.
    from wikitrace.config import TraverseConfig

    runner = PiTraverseRunner(
        pi_cli=str(_fake_pi_cli(tmp_path)), model="m", sandbox="docker", api_key="sk-test",
    )
    assert runner.provider == TraverseConfig().provider == "openrouter"
    assert runner.base_url == TraverseConfig().base_url

    # ...and the default must actually WIRE the OpenRouter credential: what matters
    # is which key reaches Pi, not which string the attribute holds.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    env = runner._env()
    assert env["OPENROUTER_API_KEY"] == "sk-test"
    assert "OPENAI_API_KEY" not in env
    assert "OPENAI_BASE_URL" not in env


@pytest.mark.usefixtures("docker_image_available")
def test_allowlist_wires_add_host_socket_mount_and_relay_entrypoint(tmp_path: Path):
    # Under allowlist the container stays --network none but gains exactly one
    # egress hole per host: the loopback pin, the /net socket mount, and the relay
    # wrapping Pi (so it starts the listeners before Pi's first LLM call).
    from wikitrace.harness.terminal.egress import EgressRelay

    runner = PiTraverseRunner(
        pi_cli=str(_fake_pi_cli(tmp_path)), model="m",
        sandbox="docker", network_policy="allowlist",
    )
    net_dir = tmp_path / "net"
    relay = EgressRelay(net_dir, [("openrouter.ai", 443)])
    argv = runner._docker_cmd(
        container_name="c", pi_cli=str(_fake_pi_cli(tmp_path / "b")),
        work=tmp_path / "work", corpus_host=tmp_path / "corpus", files_host=None,
        pi_args=["--mode", "rpc"], env={"OPENROUTER_API_KEY": "sk-xxx"}, relay=relay,
    )
    s = " ".join(argv)
    assert argv[argv.index("--network") + 1] == "none"  # still no NIC
    assert "--init" in argv  # PID-1 reaping (the relay wraps Pi as PID 1)
    assert "--add-host openrouter.ai:127.0.0.1" in s
    assert f"{net_dir}:/net:rw" in s
    assert "/relay/incontainer.mjs:ro" in s
    # the relay is the entrypoint and wraps Pi (its listeners come up first)
    assert "/relay/incontainer.mjs /net/_relay.json --" in s
    assert s.index("incontainer.mjs /net/_relay.json") < s.index("/pi/")


@pytest.mark.usefixtures("docker_image_available")
def test_allowed_hosts_parse_forms_and_rejects_garbage(tmp_path: Path):
    # A malformed allowlist must fail at CONSTRUCTION, not mis-pin --add-host later.
    runner = PiTraverseRunner(
        pi_cli=str(_fake_pi_cli(tmp_path)), model="m", sandbox="docker",
        network_policy="allowlist",
        allowed_hosts=["ncbi.nlm.nih.gov", "eutils.example.org:8443",
                       "https://rest.ensembl.org/x"],
    )
    assert runner.allowed_hosts == [
        ("ncbi.nlm.nih.gov", 443),
        ("eutils.example.org", 8443),
        ("rest.ensembl.org", 443),
    ]
    with pytest.raises(ValueError, match="malformed|empty"):
        PiTraverseRunner(
            pi_cli=str(_fake_pi_cli(tmp_path / "b")), model="m", sandbox="docker",
            network_policy="allowlist", allowed_hosts=["host:notaport"],
        )


@pytest.mark.usefixtures("docker_image_available")
def test_deny_policy_has_no_relay_wiring(tmp_path: Path):
    # Deny is fully offline: no add-host, no /net mount, no relay entrypoint.
    argv = _docker_argv(tmp_path, network_policy="deny")
    s = " ".join(argv)
    assert "--add-host" not in s
    assert "/net" not in s
    assert "incontainer.mjs" not in s


@pytest.mark.usefixtures("docker_image_available")
def test_unknown_network_policy_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="network_policy"):
        PiTraverseRunner(
            pi_cli=str(_fake_pi_cli(tmp_path)), model="m",
            sandbox="docker", network_policy="open",
        )


def test_host_sandbox_warns_that_policy_is_unenforced(tmp_path: Path):
    # sandbox="none" has no way to restrict egress; say so instead of implying
    # the policy applies. Warn (not raise): it is the deliberate no-docker escape
    # hatch, and the default policy must not make it unusable.
    with pytest.warns(RuntimeWarning, match="not enforced|NOT enforced"):
        PiTraverseRunner(pi_cli=str(_fake_pi_cli(tmp_path)), model="m", sandbox="none")


def test_read_set_by_layer_keys_the_present_layers_only():
    from wikitrace.harness.terminal.turns import read_set_by_layer

    rs = ["knowledge/document-0003.md", "raw/step-0001/", "knowledge/document-0011.md", "x.md"]
    assert read_set_by_layer(rs, ("K",)) == {"K": ["document-0003", "document-0011"]}
    assert read_set_by_layer(rs, ("R", "K")) == {"R": ["step-0001"], "K": ["document-0003", "document-0011"]}


def test_write_turns_multiline(tmp_path: Path):
    path = tmp_path / "selector.full.jsonl"
    turns = [{"turn": 0, "type": "tool_call"}, {"turn": 1, "type": "sentinel"}]
    write_turns(path, turns)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2




# ---------------------------------------------------------------------------
# Terminal cost accounting
# ---------------------------------------------------------------------------

# These use a REAL pipe and the real ``select``: the bug this capture is most likely to
# hit is a buffering interaction between the two, which a fake stdout cannot express.


class _PipeProc:
    """A Pi RPC stand-in over a real pipe: `feed()` writes bytes Pi would emit."""

    def __init__(self, alive: bool = True):
        r, self._w = os.pipe()
        self.stdout = os.fdopen(r, "rb")
        self.stdin = self
        self.sent: list[bytes] = []
        self._alive = alive

    def feed(self, data: bytes) -> None:
        """One os.write == one pipe chunk, however many lines it holds."""
        os.write(self._w, data)

    # stdin protocol
    def write(self, data: bytes) -> None:
        self.sent.append(data)

    def flush(self) -> None:
        pass

    def poll(self):
        return None if self._alive else 0

    def close_write(self) -> None:
        """EOF for the reader, WITHOUT closing the read end it is still using."""
        try:
            os.close(self._w)
        except OSError:
            pass

    def close(self) -> None:
        self.close_write()
        self.stdout.close()


@pytest.fixture()
def pi_proc():
    proc = _PipeProc()
    yield proc
    proc.close()


def _stats_line(**data) -> str:
    return json.dumps({
        "type": "response", "command": "get_session_stats", "success": True, "data": data,
    }) + "\n"


def _messages_line(*assistant: dict) -> str:
    """A `get_usage` response: one slim entry per provider round-trip, no bodies."""
    return json.dumps({
        "type": "response", "command": "get_usage", "success": True,
        "data": {"calls": list(assistant)},
    }) + "\n"


def _assistant(input_, output, cost, **kw) -> dict:
    return {
        "model": kw.pop("model", "deepseek/deepseek-v4-flash"),
        "responseId": kw.pop("responseId", "gen-x"),
        "stopReason": kw.pop("stopReason", "toolUse"),
        "usage": {
            "input": input_, "output": output, "cacheRead": kw.pop("cacheRead", 0),
            "cacheWrite": 0, "totalTokens": input_ + output,
            "cost": {"total": cost},
        },
    }


_TOKENS = {"input": 120, "output": 30, "cacheRead": 900, "cacheWrite": 50, "total": 1100}


def test_session_stats_normalized_to_llm_call_vocabulary(pi_proc):
    """Session stats normalise onto the llm-call vocabulary, whose cached tokens are a subset."""
    pi_proc.feed(_stats_line(tokens=_TOKENS, cost=0.0042).encode())
    usage = runner_mod.collect_session_stats(pi_proc, timeout=5.0)
    assert usage == {
        "prompt_tokens": 120 + 900 + 50, "completion_tokens": 30,
        "reasoning_tokens": 0,
        "cached_tokens": 900, "cache_write_tokens": 50,
        "total_tokens": 1100, "cost": 0.0042,
    }
    assert usage["cached_tokens"] < usage["prompt_tokens"]        # subset invariant
    assert usage["prompt_tokens"] + usage["completion_tokens"] == _TOKENS["total"]
    assert json.loads(pi_proc.sent[0].decode())["type"] == "get_session_stats"


def test_reasoning_tokens_are_a_subset_of_completion_not_an_addition(pi_proc):
    """`reasoning_tokens` is a SUBSET of `completion_tokens`, as on the chat path."""

    # pi-patch 0003 reports `reasoning` beside `output` rather than folded into it, so
    # one vocabulary (`harness/llm/usage.py`) covers both harnesses; reading it as an
    # addition would inflate a terminal run's completion count.
    tokens = {"input": 100, "output": 4096, "reasoning": 4096,
              "cacheRead": 0, "cacheWrite": 0, "total": 4196}
    pi_proc.feed(_stats_line(tokens=tokens, cost=0.01).encode())
    usage = runner_mod.collect_session_stats(pi_proc, timeout=5.0)
    assert usage["completion_tokens"] == 4096       # NOT 8192
    assert usage["reasoning_tokens"] == 4096
    assert usage["reasoning_tokens"] <= usage["completion_tokens"]


def test_session_stats_found_when_a_trailing_event_shares_the_pipe_read(pi_proc):
    """Pi's last event and this response routinely arrive in ONE pipe read."""
    pi_proc.feed(
        (json.dumps({"type": "agent_end"}) + "\n" + _stats_line(tokens=_TOKENS, cost=1.5)).encode()
    )
    started = time.monotonic()
    usage = runner_mod.collect_session_stats(pi_proc, timeout=5.0)
    assert usage["cost"] == 1.5                     # not dropped
    assert time.monotonic() - started < 1.0         # and not waited out


def test_session_stats_gives_up_within_its_timeout_when_pi_never_answers(pi_proc):
    """The load-bearing guarantee: accounting must never hang a run."""
    started = time.monotonic()
    assert runner_mod.collect_session_stats(pi_proc, timeout=0.3) == {}
    assert 0.25 < time.monotonic() - started < 3.0


@pytest.mark.parametrize(
    "payload",
    [
        b"",                                                        # EOF, nothing said
        _stats_line(cost=1.0).encode(),                             # no tokens block
        _stats_line(tokens={"cacheRead": 5}, cost=1.0).encode(),    # tokens, but not the real ones
        b'{"type":"response","command":"get_session_stats","success":false}\n',
        b"not json at all\n",
    ],
)
def test_session_stats_degrades_to_empty_never_raises(payload, pi_proc):
    pi_proc.feed(payload)
    pi_proc.close_write()  # EOF: a non-answer ends by exhaustion, not by timeout
    assert runner_mod.collect_session_stats(pi_proc, timeout=5.0) == {}


def test_session_stats_skipped_when_process_already_exited():
    proc = _PipeProc(alive=False)
    try:
        proc.feed(_stats_line(tokens=_TOKENS, cost=1.0).encode())
        assert runner_mod.collect_session_stats(proc) == {}
        assert proc.sent == []  # never written to a dead session's stdin
    finally:
        proc.close()


def test_call_usage_is_one_entry_per_provider_round_trip(pi_proc):
    """A terminal phase makes MANY provider calls (one per tool call)."""
    pi_proc.feed(_messages_line(
        _assistant(100, 20, 0.01, responseId="gen-1"),
        _assistant(300, 40, 0.03, responseId="gen-2", cacheRead=200, stopReason="stop"),
    ).encode())
    calls = runner_mod.collect_call_usage(pi_proc, timeout=5.0)
    assert [c["cost"] for c in calls] == [0.01, 0.03]
    assert [c["generation_id"] for c in calls] == ["gen-1", "gen-2"]
    assert calls[1]["prompt_tokens"] == 300 + 200      # cacheRead folded in
    assert calls[1]["cached_tokens"] == 200            # ...and still the subset
    assert calls[1]["finish_reason"] == "stop"
    assert calls[0]["model_returned"] == "deepseek/deepseek-v4-flash"
    assert runner_mod._sum_usage(calls)["cost"] == pytest.approx(0.04)


def test_call_usage_skips_the_synthetic_message_an_abort_leaves_behind(pi_proc):
    """Call usage skips the synthetic message an abort leaves behind."""
    aborted = _assistant(0, 0, 0.0, responseId="", stopReason="aborted")
    pi_proc.feed(_messages_line(_assistant(100, 20, 0.01), aborted).encode())
    calls = runner_mod.collect_call_usage(pi_proc, timeout=5.0)
    assert len(calls) == 1 and calls[0]["cost"] == 0.01
    assert "aborted" not in [c["finish_reason"] for c in calls]


@pytest.mark.parametrize(
    "payload",
    [
        b'{"type":"response","command":"get_usage","success":true,"data":{}}\n',
        # an unpatched Pi does not know the command
        b'{"type":"response","command":"get_usage","success":false}\n',
        b"",
    ],
)
def test_call_usage_degrades_to_empty_so_the_caller_can_fall_back(payload, pi_proc):
    pi_proc.feed(payload)
    pi_proc.close_write()
    assert runner_mod.collect_call_usage(pi_proc, timeout=5.0) == []


def test_rpc_read_gives_up_on_a_line_that_never_finishes(pi_proc):
    """An RPC read gives up on a line that never finishes, rather than wedging the phase."""
    pi_proc.feed(b'{"type":"response","command":"get_usage","success":true,"data":{"cal')
    started = time.monotonic()
    assert runner_mod.collect_call_usage(pi_proc, timeout=0.3) == []
    elapsed = time.monotonic() - started
    assert 0.25 < elapsed < 3.0, f"read was not bounded by the timeout ({elapsed:.2f}s)"


def test_rpc_read_reassembles_a_line_split_across_pipe_reads(pi_proc):
    """An RPC read reassembles a line split across pipe reads."""
    line = _messages_line(_assistant(100, 20, 0.01, responseId="gen-split"))
    half = len(line) // 2
    pi_proc.feed(line[:half].encode())
    pi_proc.feed(line[half:].encode())
    calls = runner_mod.collect_call_usage(pi_proc, timeout=5.0)
    assert [c["generation_id"] for c in calls] == ["gen-split"]


# ---------------------------------------------------------------------------
# The capture + abort wiring inside run_session, against a scripted fake process
# ---------------------------------------------------------------------------


class _ScriptedProc(_PipeProc):
    """A Popen stand-in: scripted stdout, recorded stdin, no real child."""

    def __init__(self, script: bytes, stderr: bytes = b""):
        super().__init__()
        r, w = os.pipe()          # stderr, immediately closed so the drain thread ends
        self.stderr = os.fdopen(r, "rb")
        os.write(w, stderr)
        os.close(w)
        self.pid = -1             # os.getpgid(-1) raises -> the finally falls to kill()
        self.killed = False
        self.feed(script)

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout=None):
        return 0

    def close(self) -> None:
        super().close()
        self.stderr.close()


def _events(*lines: str) -> bytes:
    return "".join(json.dumps(e) + "\n" if isinstance(e, dict) else e for e in lines).encode()


def _run_scripted(
    tmp_path: Path, monkeypatch, script: bytes, max_turns: int,
    proc: _ScriptedProc | None = None, session_kw: dict | None = None, **runner_kw,
):
    proc = proc or _ScriptedProc(script)
    # Shim the MODULE reference, not `subprocess.Popen` itself: setattr on the real
    # module would hand this fake to every other Popen caller in the process,
    # pytest's own machinery included.
    import types as _types
    shim = _types.SimpleNamespace(
        Popen=lambda *a, **kw: proc,
        PIPE=runner_mod.subprocess.PIPE,
        run=runner_mod.subprocess.run,
    )
    monkeypatch.setattr(runner_mod, "subprocess", shim)
    runner = PiTraverseRunner(
        pi_cli=str(_fake_pi_cli(tmp_path)), model="m", sandbox="none", **runner_kw
    )
    out = tmp_path / "out"
    res = runner.run_session(
        system_prompt="sys", user_msg="u", corpus_dir=None, out_dir=out,
        max_turns=max_turns, label="selector", **(session_kw or {}),
    )
    return res, proc


@pytest.fixture()
def _quiet_host_mode():
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)   # sandbox="none" warns by design
        yield


def test_run_session_aborts_during_tool_execution_not_at_turn_start(
    tmp_path: Path, monkeypatch, _quiet_host_mode
):
    """Where the abort lands decides whether a generation is billed-and-lost."""
    script = _events(
        {"type": "turn_start"},                                   # turn 1
        {"type": "tool_execution_start", "toolName": "bash", "args": {}},
        {"type": "turn_start"},                                   # turn 2 — over cap
        # No abort may have been written yet: the request for this turn is in flight.
        {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "hi"}},
        {"type": "tool_execution_start", "toolName": "bash", "args": {}},   # <- abort here
        {"type": "agent_end"},
        _messages_line(_assistant(100, 20, 0.01, responseId="gen-1")),
    )
    res, proc = _run_scripted(tmp_path, monkeypatch, script, max_turns=1)

    sent = [json.loads(b.decode()) for b in proc.sent]
    kinds = [m["type"] for m in sent]
    assert "abort" in kinds, "the turn cap was never enforced"
    # The abort must follow the prompt and precede get_usage, and — the point of
    # this test — must not have been the reaction to `turn_start`.
    assert kinds.index("abort") > kinds.index("prompt")
    # usage still captured after an aborted session
    assert res.usage_calls and res.usage_calls[0]["generation_id"] == "gen-1"
    assert res.usage["cost"] == pytest.approx(0.01)
    assert proc.killed  # the finally still reaped the process


def test_run_session_records_usage_on_a_clean_finish(
    tmp_path: Path, monkeypatch, _quiet_host_mode
):
    """No cap hit: no abort, and the per-call usage still reaches TerminalResult."""
    script = _events(
        {"type": "turn_start"},
        {"type": "agent_end"},
        _messages_line(
            _assistant(100, 20, 0.01, responseId="gen-a"),
            _assistant(300, 40, 0.03, responseId="gen-b"),
        ),
    )
    res, proc = _run_scripted(tmp_path, monkeypatch, script, max_turns=50)
    assert "abort" not in [json.loads(b.decode())["type"] for b in proc.sent]
    assert [c["generation_id"] for c in res.usage_calls] == ["gen-a", "gen-b"]
    assert res.usage["cost"] == pytest.approx(0.04)
    assert res.model_returned == "deepseek/deepseek-v4-flash"


def test_run_session_declares_a_session_it_had_to_kill(
    tmp_path: Path, monkeypatch, _quiet_host_mode
):
    """A session the runner had to kill is declared, since its cost cannot be recovered."""
    script = _events(
        {"type": "turn_start"},
        # ...and then nothing: the session goes quiet and the idle guard fires.
    )
    # Slow (~9s) on purpose: a fake that answers NOTHING makes the post-kill usage
    # capture wait out both of its leashes. That is the behaviour under test — the
    # capture must give up on its own rather than hang — so it is not stubbed away.
    proc = _ScriptedProc(script)
    import types as _types
    monkeypatch.setattr(runner_mod, "subprocess", _types.SimpleNamespace(
        Popen=lambda *a, **kw: proc, PIPE=runner_mod.subprocess.PIPE,
        run=runner_mod.subprocess.run,
    ))
    runner = PiTraverseRunner(
        pi_cli=str(_fake_pi_cli(tmp_path)), model="m", sandbox="none", idle_timeout=0.2,
    )
    res = runner.run_session(
        system_prompt="s", user_msg="u", corpus_dir=None, out_dir=tmp_path / "o",
        max_turns=50, label="selector",
    )
    assert res.stopped_by == "idle"
    assert [t["type"] for t in res.turns if t["type"] == "timeout"] == ["timeout"]


def test_run_session_does_not_abort_a_turn_that_never_reaches_a_tool(
    tmp_path: Path, monkeypatch, _quiet_host_mode
):
    """The discriminating half of the abort-position pair."""
    script = _events(
        {"type": "turn_start"},                              # turn 1
        {"type": "tool_execution_start", "toolName": "bash", "args": {}},
        {"type": "turn_start"},                              # turn 2 — over cap...
        {"type": "agent_end"},                               # ...but it just answers
        _messages_line(_assistant(100, 20, 0.01, responseId="gen-1")),
    )
    res, proc = _run_scripted(tmp_path, monkeypatch, script, max_turns=1)
    kinds = [json.loads(b.decode())["type"] for b in proc.sent]
    assert "abort" not in kinds, "aborted a turn with no tool window — a live stream"
    assert res.usage_calls  # ...and the usage still came back


def test_run_session_re_arms_a_dropped_abort_and_stops_hard(
    tmp_path: Path, monkeypatch, _quiet_host_mode
):
    """A dropped abort is re-armed, and the session stops hard once several go unanswered."""
    over = [
        {"type": "turn_start"},
        {"type": "tool_execution_start", "toolName": "bash", "args": {}},
    ]
    script = _events(*(over * 8))    # Pi ignores every abort
    res, proc = _run_scripted(tmp_path, monkeypatch, script, max_turns=1)
    aborts = [m for m in (json.loads(b.decode()) for b in proc.sent) if m["type"] == "abort"]
    assert len(aborts) > 1, "the abort was sent once and never re-armed"
    assert res.stopped_by == "turn_cap"
    assert [t for t in res.turns if t.get("reason") == "turn_cap"]


def test_run_session_declares_an_eof_it_did_not_expect(
    tmp_path: Path, monkeypatch, _quiet_host_mode
):
    """An unexpected EOF is declared rather than read as a clean finish."""
    proc = _ScriptedProc(_events({"type": "turn_start"}))
    proc.close_write()               # EOF, with no agent_end
    import types as _types
    monkeypatch.setattr(runner_mod, "subprocess", _types.SimpleNamespace(
        Popen=lambda *a, **kw: proc, PIPE=runner_mod.subprocess.PIPE,
        run=runner_mod.subprocess.run,
    ))
    res = PiTraverseRunner(
        pi_cli=str(_fake_pi_cli(tmp_path)), model="m", sandbox="none",
    ).run_session(
        system_prompt="s", user_msg="u", corpus_dir=None, out_dir=tmp_path / "o",
        max_turns=50, label="selector",
    )
    assert res.stopped_by == "eof"


# ---------------------------------------------------------------------------
# The per-command force-kill (pi-patch 0004) and its session budget
# ---------------------------------------------------------------------------

_KILL_TEXT = (
    "[harness] Command force-killed after 90s — a hard per-command limit, not a "
    "sandbox error."
)


def _kill_end() -> dict:
    """A `tool_execution_end` shaped like the one pi-patch 0004 produces."""
    return {
        "type": "tool_execution_end", "toolName": "bash", "isError": True,
        "result": {"content": [{"type": "text", "text": _KILL_TEXT}], "details": {}},
    }


def test_a_force_killed_command_no_longer_ends_the_session(
    tmp_path: Path, monkeypatch, _quiet_host_mode
):
    """A force-killed command ends the COMMAND, not the session, which goes on to answer."""
    script = _events(
        {"type": "turn_start"},
        {"type": "tool_execution_start", "toolName": "bash", "args": {"command": "python3 -c ..."}},
        _kill_end(),
        {"type": "turn_start"},
        {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "ok"}},
        {"type": "agent_end"},
    )
    res, proc = _run_scripted(tmp_path, monkeypatch, script, max_turns=50)

    assert res.stopped_by == "", "a force-killed command must not stop the session"
    assert res.final_text == "ok", "the session ran on and produced its answer"
    kills = [t for t in res.turns if t.get("type") == "cmd_kill"]
    assert [k["n"] for k in kills] == [1], "the kill is recorded for later analysis"
    # Nothing was steered: the budget is 5 and only one was spent.
    assert "steer" not in [json.loads(b.decode())["type"] for b in proc.sent]


def test_the_kill_budget_warns_once_and_then_ends_the_session(
    tmp_path: Path, monkeypatch, _quiet_host_mode
):
    """The kill budget warns once and then ends the session, leaving a turn to answer in."""
    call = [
        {"type": "tool_execution_start", "toolName": "bash", "args": {}},
        _kill_end(),
    ]
    script = _events(*(call * 3), {"type": "agent_end"})
    res, proc = _run_scripted(tmp_path, monkeypatch, script, max_turns=50, max_cmd_kills=2)

    sent = [json.loads(b.decode()) for b in proc.sent]
    steers = [m for m in sent if m["type"] == "steer"]
    assert len(steers) == 1, "warned exactly once, on the last kill of the budget"
    assert "2 of 2" in steers[0]["message"] and "__out.json" in steers[0]["message"]
    assert res.stopped_by == "cmd_kill_budget"
    assert [t["n"] for t in res.turns if t.get("type") == "cmd_kill"] == [1, 2, 3]


def test_an_ordinary_tool_error_is_not_a_force_kill(
    tmp_path: Path, monkeypatch, _quiet_host_mode
):
    """A command that merely FAILED spends no budget."""
    script = _events(
        {"type": "turn_start"},
        {"type": "tool_execution_start", "toolName": "bash", "args": {}},
        {"type": "tool_execution_end", "toolName": "bash", "isError": True,
         "result": {"content": [{"type": "text", "text": "Command exited with code 1"}]}},
        {"type": "agent_end"},
    )
    res, _ = _run_scripted(tmp_path, monkeypatch, script, max_turns=50, max_cmd_kills=1)
    assert res.stopped_by == ""
    assert not [t for t in res.turns if t.get("type") == "cmd_kill"]


def test_the_runner_guard_sits_above_the_limit_pi_enforces(tmp_path: Path, _quiet_host_mode):
    """Two limits, and the order between them is what makes the fix work."""
    r = PiTraverseRunner(pi_cli=str(_fake_pi_cli(tmp_path)), model="m", sandbox="none",
                         cmd_timeout=90.0)
    assert r.cmd_backstop > r.cmd_timeout
    # ...and the value Pi enforces is handed to it, in both sandbox modes.
    assert r._env()["PI_BASH_MAX_TIMEOUT_SEC"] == "90"
    argv = PiTraverseRunner(
        pi_cli=str(_fake_pi_cli(tmp_path / "b")), model="m", sandbox="none",
        cmd_timeout=300.0,
    )._docker_cmd(
        container_name="c", pi_cli=str(_fake_pi_cli(tmp_path / "c")), work=tmp_path,
        corpus_host=None, files_host=None, pi_args=[], env={},
    )
    assert "PI_BASH_MAX_TIMEOUT_SEC=300" in argv


def test_is_harness_kill_reads_the_marker_not_the_error_flag():
    from wikitrace.harness.terminal.runner import _is_harness_kill

    assert _is_harness_kill(_kill_end())
    # right text, but the call succeeded — the agent quoting the notice back at us
    assert not _is_harness_kill({**_kill_end(), "isError": False})
    assert not _is_harness_kill({"isError": True, "result": {"content": [{"text": "boom"}]}})
    assert not _is_harness_kill({"isError": True, "result": None})


# ---------------------------------------------------------------------------
# A failed provider request: Pi's own retry, and the error when retries do not clear it
# ---------------------------------------------------------------------------

_ERR = "429 Provider returned error"


def _assistant_end(stop: str, **kw) -> dict:
    """A `message_end` for an assistant message, shaped as Pi's agent loop emits it."""
    return {"type": "message_end", "message": {"role": "assistant", "stopReason": stop, "content": [], **kw}}


class _UsageOnAskProc(_ScriptedProc):
    """Answers `get_usage` only once asked, so a runner that reads ahead cannot eat the reply."""

    def write(self, data: bytes) -> None:
        super().write(data)
        if b'"get_usage"' in data:
            self.feed(_messages_line(_assistant(100, 20, 0.01, responseId="gen-1")).encode())


def test_a_failed_request_that_pi_retries_keeps_the_session_running(
    tmp_path: Path, monkeypatch, _quiet_host_mode
):
    """(i) agent_end on an error, then Pi's retry: the session carries on to its real end."""
    script = _events(
        {"type": "turn_start"},
        _assistant_end("error", errorMessage=_ERR),
        {"type": "agent_end"},
        {"type": "auto_retry_start", "attempt": 1, "maxAttempts": 3, "delayMs": 2000, "errorMessage": _ERR},
        {"type": "turn_start"},
        {"type": "tool_execution_start", "toolName": "bash", "args": {"command": "ls"}},
        {"type": "tool_execution_end", "toolName": "bash", "isError": False, "result": {"content": []}},
        {"type": "turn_start"},
        {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "done"}},
        _assistant_end("stop"),
        {"type": "auto_retry_end", "success": True, "attempt": 1},
        {"type": "agent_end"},
        _messages_line(_assistant(100, 20, 0.01, responseId="gen-1")),
    )
    res, _ = _run_scripted(tmp_path, monkeypatch, script, max_turns=2)
    assert res.provider_error == "" and res.provider_retries == 1
    assert res.final_text == "done" and res.last_stop_reason == "stop" and res.stopped_by == ""
    retry = [t for t in res.turns if t["type"] == "provider_retry"]
    assert [(t["attempt"], t["error"]) for t in retry] == [(1, _ERR)]
    assert [t["type"] for t in res.turns] == ["provider_retry", "function_call"]
    assert res.usage_calls  # the usage reply after the second agent_end was not consumed


def test_a_failed_request_pi_does_not_retry_ends_on_a_provider_error(
    tmp_path: Path, monkeypatch, _quiet_host_mode
):
    """(ii) No auto_retry_start within the grace window: stop, carrying the error."""
    monkeypatch.setattr(runner_mod, "_PROVIDER_RETRY_GRACE", 0.3)
    script = _events({"type": "turn_start"}, _assistant_end("error", errorMessage=_ERR), {"type": "agent_end"})
    started = time.monotonic()
    res, _ = _run_scripted(
        tmp_path, monkeypatch, script, max_turns=50, proc=_UsageOnAskProc(script),
    )
    assert 0.3 <= time.monotonic() - started < 5.0
    assert res.provider_error == _ERR and res.provider_retries == 0
    assert res.stopped_by == "" and not [t for t in res.turns if t["type"] == "timeout"]
    assert res.usage_calls


def test_retries_that_run_out_end_on_the_final_error(tmp_path: Path, monkeypatch, _quiet_host_mode):
    """Pi's `auto_retry_end` failure ends the session at once, with its final error."""
    script = _events(
        {"type": "turn_start"}, _assistant_end("error", errorMessage=_ERR), {"type": "agent_end"},
        {"type": "auto_retry_end", "success": False, "attempt": 3, "finalError": "503 upstream"},
        _messages_line(_assistant(100, 20, 0.01, responseId="gen-1")),
    )
    res, _ = _run_scripted(tmp_path, monkeypatch, script, max_turns=50)
    assert res.provider_error == "503 upstream" and res.usage_calls


def test_a_normal_end_stops_at_once_with_no_provider_fields(
    tmp_path: Path, monkeypatch, _quiet_host_mode
):
    """(iii) A clean agent_end waits for nothing and sets nothing new."""
    script = _events({"type": "turn_start"}, _assistant_end("stop"), {"type": "agent_end"})
    started = time.monotonic()
    res, _ = _run_scripted(
        tmp_path, monkeypatch, script, max_turns=50, proc=_UsageOnAskProc(script),
    )
    assert time.monotonic() - started < runner_mod._PROVIDER_RETRY_GRACE
    assert (res.provider_error, res.provider_retries, res.stopped_by) == ("", 0, "")
    assert res.turns == [] and res.usage_calls
    assert not (tmp_path / "out" / "_pi_stderr.log").exists()


def test_pi_stderr_is_saved_to_the_session_dir(tmp_path: Path, monkeypatch, _quiet_host_mode):
    """(iv) Pi's stderr lands next to the session's other artifacts."""
    script = _events({"type": "turn_start"}, {"type": "agent_end"}, _messages_line())
    _run_scripted(
        tmp_path, monkeypatch, script, max_turns=50,
        proc=_ScriptedProc(script, stderr=b"Error: 429 Too Many Requests\n"),
    )
    assert (tmp_path / "out" / "_pi_stderr.log").read_text() == "Error: 429 Too Many Requests\n"


# ---------------------------------------------------------------------------
# The final message row: the last reply's own text and reasoning
# ---------------------------------------------------------------------------


def _delta(text: str) -> dict:
    return {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": text}}


def _call(cid: str) -> list[dict]:
    return [
        {"type": "tool_execution_start", "toolCallId": cid, "toolName": "bash", "args": {"command": "ls"}},
        {"type": "tool_execution_end", "toolName": "bash", "isError": False, "result": {"content": []}},
    ]


def test_the_final_text_is_the_last_reply_alone_with_its_reasoning(
    tmp_path: Path, monkeypatch, _quiet_host_mode
):
    """Prose written beside a tool call stays out of the final reply."""
    script = _events(
        {"type": "turn_start"},
        _delta("Let me look."),
        _assistant_end("toolUse", content=[
            {"type": "thinking", "thinking": "start by listing"},
            {"type": "text", "text": "Let me look."}, {"type": "toolCall", "id": "c1"},
        ]),
        *_call("c1"),
        {"type": "turn_start"},
        _delta("One more."),
        _assistant_end("toolUse", content=[{"type": "text", "text": "One more."}, {"type": "toolCall", "id": "c2"}]),
        *_call("c2"),
        {"type": "turn_start"},
        _delta("The answer is 42."),
        _assistant_end("stop", content=[
            {"type": "thinking", "thinking": "both outputs agree"},
            {"type": "text", "text": "The answer is 42."},
        ]),
        {"type": "agent_end"},
    )
    res, _ = _run_scripted(tmp_path, monkeypatch, script, max_turns=50)
    assert res.final_text == "The answer is 42."
    assert res.final_reasoning == "both outputs agree"
    assert res.session_text == "Let me look.One more.The answer is 42."


def test_the_citation_turn_leaves_the_final_text_to_the_solvers_last_reply(
    tmp_path: Path, monkeypatch, _quiet_host_mode
):
    from wikitrace.harness.environment import EnvTool
    from wikitrace.memory.citation import citation_message

    steer = citation_message(("R",))
    script = _events(
        {"type": "turn_start"},
        _delta("Putting it in the box."),
        _assistant_end("toolUse", content=[
            {"type": "thinking", "thinking": "the box is open"},
            {"type": "text", "text": "Putting it in the box."}, {"type": "toolCall", "id": "c1"},
        ]),
        {"type": "tool_execution_start", "toolCallId": "c1", "toolName": "act", "args": {"command": "put key in box"}},
        {"type": "tool_execution_end", "toolName": "act", "isError": False,
         "result": {"content": [{"type": "text", "text": "Done.\n\nEpisode over"}]}},
        {"type": "message_end", "message": {"role": "toolResult", "toolCallId": "c1"}},
        {"type": "message_end", "message": {"role": "user", "content": steer}},
        {"type": "turn_start"},
        _delta('{"R": ["step-0001"]}'),
        _assistant_end("stop", content=[{"type": "text", "text": '{"R": ["step-0001"]}'}]),
        {"type": "agent_end"},
    )
    res, _ = _run_scripted(
        tmp_path, monkeypatch, script, max_turns=50,
        session_kw=dict(env_tool=EnvTool("act", tmp_path / "sock"), citation_prompt=steer, citation_layers=("R",)),
    )
    assert res.citation == {"R": ["step-0001"]}
    assert res.final_text == "Putting it in the box."
    assert res.final_reasoning == "the box is open"


def test_a_message_end_without_content_falls_back_to_that_messages_deltas(
    tmp_path: Path, monkeypatch, _quiet_host_mode
):
    script = _events(
        {"type": "turn_start"},
        _delta("beside the call"),
        {"type": "message_end", "message": {"role": "assistant", "stopReason": "toolUse"}},
        *_call("c1"),
        {"type": "turn_start"},
        _delta("the "), _delta("answer"),
        {"type": "message_end", "message": {"role": "assistant", "stopReason": "stop"}},
        {"type": "agent_end"},
    )
    res, _ = _run_scripted(tmp_path, monkeypatch, script, max_turns=50)
    assert res.final_text == "the answer" and res.final_reasoning == ""


# ---------------------------------------------------------------------------
# The message channel: a reply without a tool call goes to the user, whose reply continues
# ---------------------------------------------------------------------------


class _ConversationProc(_ScriptedProc):
    """Feeds one scripted segment per continuing prompt, a refusal where scripted, and the usage reply."""

    def __init__(self, first: list, *later: list | str):
        super().__init__(_events(*first))
        self.later = list(later)

    def write(self, data: bytes) -> None:
        super().write(data)
        msg = json.loads(data.decode())
        if msg["type"] == "prompt" and msg["id"] != "py-1" and self.later:
            nxt = self.later.pop(0)
            if nxt == "refuse":
                self.feed(_events({"id": msg["id"], "type": "response", "command": "prompt", "success": True},
                                  {"id": msg["id"], "type": "response", "command": "prompt", "success": False,
                                   "error": "Agent is already processing."}))
            else:
                self.feed(_events(*nxt))
        elif msg["type"] == "get_usage":
            self.feed(_messages_line(_assistant(100, 20, 0.01, responseId="gen-1")).encode())


def _user(replies: list):
    """A unix-socket stand-in for the environment server: answers `message` ops from ``replies``."""
    import socket
    import tempfile
    import threading

    from wikitrace.harness.environment.protocol import recv_line, send_line

    sock_dir = Path(tempfile.mkdtemp(prefix="msg-"))
    received: list[str] = []
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_dir / runner_mod.SOCKET_NAME))
    srv.listen(8)

    def serve() -> None:
        for reply in replies:
            conn, _ = srv.accept()
            with conn:
                req = recv_line(conn)
                received.append(req["text"] if req.get("op") == "message" else f"<{req.get('op')}>")
                send_line(conn, reply)
        srv.close()

    threading.Thread(target=serve, daemon=True).start()
    return sock_dir, received


def _reply_end(text: str, thinking: str = "", tool: bool = False) -> dict:
    content = ([{"type": "thinking", "thinking": thinking}] if thinking else []) + [{"type": "text", "text": text}]
    if tool:
        content.append({"type": "toolCall", "id": "c1"})
    return _assistant_end("toolUse" if tool else "stop", content=content)


def _take_action(obs: str) -> list[dict]:
    return [
        {"type": "tool_execution_start", "toolCallId": "c1", "toolName": "take_action",
         "args": {"action": "get_user_details", "argument": {"user_id": "u1"}}},
        {"type": "tool_execution_end", "toolName": "take_action", "isError": False,
         "result": {"content": [{"type": "text", "text": obs}]}},
    ]


def _tau3_tool(sock_dir: Path, messages: bool = True):
    from wikitrace.harness.environment import EnvTool

    return EnvTool("take_action", sock_dir, "take_action_tau3", messages=messages)


def test_a_conversation_sends_each_reply_without_a_tool_call_to_the_user_and_goes_on(
    tmp_path: Path, monkeypatch, _quiet_host_mode
):
    sock_dir, received = _user([
        {"observation": "a@example.com", "done": False, "over": False},
        {"observation": "The shoes, please.", "done": False, "over": False},
        {"observation": "Thanks! ###STOP###", "done": True, "over": False},
    ])
    proc = _ConversationProc(
        [{"type": "turn_start"}, _delta("May I have your email?"),
         _reply_end("May I have your email?", thinking="authenticate first"), {"type": "agent_end"}],
        # Text beside a tool call is not a message: only the call runs.
        [{"type": "turn_start"}, _delta("Looking you up."), _reply_end("Looking you up.", tool=True),
         *_take_action('{"user_id": "u1"}'),
         {"type": "turn_start"}, _reply_end("Found you. What should I return?", thinking="user found"),
         {"type": "agent_end"}],
        [{"type": "turn_start"}, _reply_end("Your return is filed."), {"type": "agent_end"}],
    )
    res, _ = _run_scripted(tmp_path, monkeypatch, b"", max_turns=50, proc=proc,
                           session_kw=dict(env_tool=_tau3_tool(sock_dir)))
    assert received == ["May I have your email?", "Found you. What should I return?", "Your return is filed."]
    prompts = [json.loads(b.decode()) for b in proc.sent]
    assert [m["message"] for m in prompts if m["type"] == "prompt"] == ["u", "a@example.com", "The shoes, please."]
    assert "abort" not in [m["type"] for m in prompts]
    rows = [(t["type"], t.get("role"), t.get("content"), t.get("reasoning")) for t in res.turns]
    assert rows == [
        ("message", "assistant", "May I have your email?", "authenticate first"),
        ("message", "user", "a@example.com", None),
        ("function_call", "assistant", None, None),
        ("message", "assistant", "Found you. What should I return?", "user found"),
        ("message", "user", "The shoes, please.", None),
        ("message", "assistant", "Your return is filed.", ""),
        ("message", "user", "Thanks! ###STOP###", None),
        ("episode_over", None, None, None),
    ]
    assert [t["turn"] for t in res.turns] == list(range(8))
    assert res.final_text == "Your return is filed." and res.stopped_by == "" and res.provider_error == ""


def test_a_prompt_pi_refuses_while_settling_is_sent_again(tmp_path: Path, monkeypatch, _quiet_host_mode):
    monkeypatch.setattr(runner_mod, "_PROMPT_RESEND_DELAY", 0.0)
    sock_dir, received = _user([
        {"observation": "yes", "done": False, "over": False},
        {"observation": "###STOP###", "done": True, "over": False},
    ])
    proc = _ConversationProc(
        [{"type": "turn_start"}, _reply_end("Shall I?"), {"type": "agent_end"}],
        "refuse",
        [{"type": "turn_start"}, _reply_end("Done."), {"type": "agent_end"}],
    )
    res, _ = _run_scripted(tmp_path, monkeypatch, b"", max_turns=50, proc=proc,
                           session_kw=dict(env_tool=_tau3_tool(sock_dir)))
    prompts = [json.loads(b.decode()) for b in proc.sent if json.loads(b.decode())["type"] == "prompt"]
    assert [m["message"] for m in prompts] == ["u", "yes", "yes"] and prompts[1] == prompts[2]
    assert received == ["Shall I?", "Done."] and res.turns[-1]["type"] == "episode_over"


def test_the_message_that_ends_the_episode_is_followed_by_the_citation_prompt_alone(
    tmp_path: Path, monkeypatch, _quiet_host_mode
):
    from wikitrace.memory.citation import citation_message

    steer = citation_message(("I",))
    sock_dir, received = _user([{"observation": "###STOP###", "done": True, "over": False}])
    proc = _ConversationProc(
        [{"type": "turn_start"}, _reply_end("All set.", thinking="wrap up"), {"type": "agent_end"}],
        [{"type": "turn_start"}, _delta('{"I": ["insight-0001"]}'),
         _reply_end('{"I": ["insight-0001"]}'), {"type": "agent_end"}],
    )
    res, _ = _run_scripted(tmp_path, monkeypatch, b"", max_turns=50, proc=proc, session_kw=dict(
        env_tool=_tau3_tool(sock_dir), citation_prompt=steer, citation_layers=("I",)))
    # The user's closing reply is recorded, never shown to the agent; the citation answer is not a message.
    assert received == ["All set."]
    assert [json.loads(b.decode()).get("message") for b in proc.sent if b'"prompt"' in b] == ["u", steer]
    assert [t["type"] for t in res.turns] == ["message", "message", "episode_over", "citation"]
    assert res.citation == {"I": ["insight-0001"]}
    assert res.final_text == "All set." and res.final_reasoning == "wrap up"


def test_a_reply_after_episode_over_is_the_citation_answer_and_never_reaches_the_user(
    tmp_path: Path, monkeypatch, _quiet_host_mode
):
    from wikitrace.memory.citation import citation_message

    steer = citation_message(("I",))
    sock_dir, received = _user([])
    proc = _ConversationProc([
        {"type": "turn_start"}, _reply_end("Checking.", tool=True),
        *_take_action("Error: Tool 'x' not found.\n\nEpisode over"),
        {"type": "message_end", "message": {"role": "user", "content": steer}},
        {"type": "turn_start"}, _delta('{"I": []}'), _reply_end('{"I": []}'), {"type": "agent_end"},
    ])
    res, _ = _run_scripted(tmp_path, monkeypatch, b"", max_turns=50, proc=proc, session_kw=dict(
        env_tool=_tau3_tool(sock_dir), citation_prompt=steer, citation_layers=("I",)))
    assert received == []
    assert [t["type"] for t in res.turns] == ["function_call", "episode_over", "citation"]
    assert res.citation == {"I": []} and res.final_text == "Checking."


def test_a_simulator_failure_on_a_message_ends_the_session(tmp_path: Path, monkeypatch, _quiet_host_mode):
    sock_dir, received = _user([{"error": "IndexError: pop from empty list"}])
    proc = _ConversationProc([{"type": "turn_start"}, _reply_end("Hello?"), {"type": "agent_end"}], [])
    res, _ = _run_scripted(tmp_path, monkeypatch, b"", max_turns=50, proc=proc,
                           session_kw=dict(env_tool=_tau3_tool(sock_dir)))
    assert received == ["Hello?"]
    assert [(t["type"], t.get("role")) for t in res.turns] == [("message", "assistant")]
    assert [json.loads(b.decode())["type"] for b in proc.sent].count("prompt") == 1


@pytest.mark.parametrize("reply", [
    _reply_end("   "),                                                      # nothing to send
    _assistant_end("length", content=[{"type": "text", "text": "I will now"}]),  # truncated
])
def test_an_empty_or_truncated_reply_is_never_sent_and_ends_the_session(
    tmp_path: Path, monkeypatch, _quiet_host_mode, reply
):
    sock_dir, received = _user([])
    proc = _ConversationProc([{"type": "turn_start"}, reply, {"type": "agent_end"}], [])
    res, _ = _run_scripted(tmp_path, monkeypatch, b"", max_turns=50, proc=proc,
                           session_kw=dict(env_tool=_tau3_tool(sock_dir)))
    assert received == [] and res.turns == []


# A benchmark that does not declare the channel: the same stream as at the head of
# feat/shared-final-message, so its TerminalResult and turn log are pinned field by field.
_NO_CHANNEL = [
    {"type": "turn_start"}, _delta("Opening it."),
    _assistant_end("toolUse", content=[{"type": "thinking", "thinking": "open first"},
                                       {"type": "text", "text": "Opening it."}, {"type": "toolCall", "id": "c1"}]),
    {"type": "tool_execution_start", "toolCallId": "c1", "toolName": "act", "args": {"command": "open box"}},
    {"type": "tool_execution_end", "toolName": "act", "isError": False,
     "result": {"content": [{"type": "text", "text": "The box is open."}]}},
    {"type": "turn_start"}, _delta("I think the task is done."),
    _assistant_end("stop", content=[{"type": "thinking", "thinking": "nothing left"},
                                    {"type": "text", "text": "I think the task is done."}]),
    {"type": "agent_end"},
]


@pytest.mark.parametrize("with_env_tool", [True, False])
def test_without_the_message_channel_a_reply_without_a_tool_call_still_ends_the_session(
    tmp_path: Path, monkeypatch, _quiet_host_mode, with_env_tool
):
    from wikitrace.harness.environment import EnvTool

    proc = _UsageOnAskProc(_events(*_NO_CHANNEL))
    kw = dict(env_tool=EnvTool("act", tmp_path / "sock")) if with_env_tool else {}
    res, proc = _run_scripted(tmp_path, monkeypatch, b"", max_turns=50, proc=proc, session_kw=kw)
    assert [json.loads(b.decode())["type"] for b in proc.sent] == ["prompt", "get_usage"]
    assert res.turns == [{"turn": 0, "type": "function_call", "role": "assistant", "call_id": "c1", "name": "act",
                          "arguments": {"command": "open box"}}]
    assert (res.final_text, res.final_reasoning, res.session_text) == (
        "I think the task is done.", "nothing left", "Opening it.I think the task is done.")
    assert (res.stopped_by, res.last_stop_reason, res.last_had_tool_call, res.truncated_calls) == ("", "stop", False, 0)
    assert (res.provider_error, res.provider_retries, res.citation, res.citation_text) == ("", 0, None, "")
    assert [c["generation_id"] for c in res.usage_calls] == ["gen-1"] and res.read_set == []
