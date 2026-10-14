"""Shared pytest fixtures: a network-free transport, an offline ALFWorld over the four-command
fake game, the resolved base config, and a tmp_path config writer that keeps artifacts out of the repo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

import pytest
import yaml

from wikitrace.config import load_config
from wikitrace.harness.llm.openrouter import ChatResponse

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end runs over the fake game")


# ---------------------------------------------------------------------------
# Fake LLM transport
# ---------------------------------------------------------------------------


class FakeTransport:
    """A network-free stand-in for OpenRouterClient, canned schema-valid JSON per role."""
    # ``doc_evals`` drives the compiler's votes, ``n_add`` the writer's ADDs per pass, and
    # ``validate_ok`` whether the model-catalog check raises.

    def __init__(
        self,
        *,
        doc_evals: Optional[list[tuple[str, str]]] = None,
        n_add: int = 1,
        validate_ok: bool = True,
    ) -> None:
        self.doc_evals = doc_evals or []
        self.n_add = n_add
        self.validate_ok = validate_ok
        self.calls: list[dict[str, Any]] = []

    # -- catalog validation -------------------------------------------------

    def validate_models(self, model_ids: list[str]) -> dict[str, Any]:
        if not self.validate_ok:
            from wikitrace.harness.llm.openrouter import ModelNotInCatalogError

            raise ModelNotInCatalogError(f"unknown model(s): {model_ids}")
        return {"validated": True, "requested_models": list(model_ids)}

    # -- chat ---------------------------------------------------------------

    def _payload_for(self, schema_name: str) -> dict[str, Any]:
        if schema_name == "EnvCompilerSuccessOutput":
            return {
                "title": "fake success analysis",
                "description": "fake success description",
                "key_decisions": [{"decision": "took the key first", "support": "step 2"}],
                "candidate_insights": [self._env_insight()],
            }
        if schema_name in ("EnvCompilerFailureOutput", "EnvCompilerFailureOutputWithConfidence"):
            cause = {"decision": "put the key in the wrong place", "support": "step 3"}
            out = {
                "title": "fake failure analysis",
                "description": "fake failure description",
                "failure_causes": [cause],
                "candidate_insights": [self._env_insight()],
            }
            if schema_name.endswith("WithConfidence"):
                cause["confidence"] = "high"
                out["uncertainties"] = []
            return out
        if schema_name == "CompilerSuccessOutput":
            return {
                "title": "fake success analysis",
                "description": "fake success description",
                "outcome_interpretation": "ok",
                "key_decisions": [
                    {"agent": "Selector", "decision": "got the right tables",
                     "observation": ""}
                ],
                "candidate_insights": [self._insight()],
                "document_evaluations": self._evals(),
            }
        if schema_name == "CompilerFailureOutput":
            return {
                "title": "fake failure analysis",
                "description": "fake failure description",
                "outcome_interpretation": "bad",
                "failure_causes": [
                    {"description": "wrong table", "support": "",
                     "confidence": "high"}
                ],
                "uncertainties": [],
                "correct_parts": [],
                "candidate_insights": [self._failure_insight()],
                "document_evaluations": self._evals(),
            }
        if schema_name == "CompilerSuccessOutputNoMem":
            return {
                "title": "fake success analysis",
                "description": "fake success description",
                "outcome_interpretation": "ok",
                "key_decisions": [
                    {"agent": "Selector", "decision": "got the right tables",
                     "observation": ""}
                ],
                "candidate_insights": [self._insight()],
            }
        if schema_name == "CompilerFailureOutputNoMem":
            return {
                "title": "fake failure analysis",
                "description": "fake failure description",
                "outcome_interpretation": "bad",
                "failure_causes": [
                    {"description": "wrong table", "support": "",
                     "confidence": "high"}
                ],
                "uncertainties": [],
                "correct_parts": [],
                "candidate_insights": [self._failure_insight()],
            }
        if schema_name == "WriterOutput":
            ops = [
                {
                    "type": "ADD",
                    "description": f"insight {i}",
                    "title": f"title {i}",
                    "guideline": f"guideline {i}",
                    "applicability": f"applicability {i}",
                    "rationale": f"rationale {i}",
                    "source_step_ids": [],
                }
                for i in range(self.n_add)
            ]
            return {"batch_patterns": [{"label": "join-grain", "count": 2, "description": "same-grain join"}],
                    "operations": ops, "reasoning": "fake writer"}
        if schema_name in ("WriterRewriteOutput", "WriterOnceOutput"):
            docs = [
                {
                    "title": f"title {i}",
                    "description": f"insight {i}",
                    "guideline": f"guideline {i}",
                    "applicability": f"applicability {i}",
                    "rationale": f"rationale {i}",
                    "source_step_ids": ["step-0000", "step-0001"],
                }
                for i in range(self.n_add)
            ]
            return {
                "batch_patterns": [{"label": "join-grain", "count": 2, "description": "same-grain join"}],
                "documents": docs,
                "reasoning": "fake rewrite",
            }
        # Unknown schema: return an empty object (will fail validation -> retry).
        return {}

    def _env_insight(self) -> dict[str, str]:
        return {"title": "take before placing", "guideline": "take the object first",
                "applicability": "when a task moves an object"}

    def _insight(self) -> dict[str, str]:
        return {
            "guideline": "always check the join key",
            "applicability": "when joining two tables",
        }

    def _failure_insight(self) -> dict[str, str]:
        """Failure compilations carry a confidence on every insight."""
        return {**self._insight(), "confidence": "high"}

    def _evals(self) -> list[dict[str, str]]:
        return [
            {"document_id": doc_id, "tag": tag, "causal_note": "note"}
            for doc_id, tag in self.doc_evals
        ]

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 8192,
        response_format: Optional[dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> ChatResponse:
        schema_name = ""
        if response_format is not None:
            schema_name = (
                response_format.get("json_schema", {}).get("name", "")
            )
        # ``messages`` is recorded so a test can assert on the REAL rendered prompt a
        # code path produced, not on a re-render of it.
        self.calls.append({"model": model, "schema": schema_name, "messages": messages})
        content = json.dumps(self._payload_for(schema_name))
        raw = {
            "id": "fake-gen-id",
            "model": model,
            "choices": [
                {"message": {"content": content}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
        }
        return ChatResponse(
            content=content,
            raw=raw,
            generation_id="fake-gen-id",
            model_returned=model,
            finish_reason="stop",
        )


# ---------------------------------------------------------------------------
# Offline ALFWorld: the real adapter and prompts over the harness's fake game
# ---------------------------------------------------------------------------

# A train/test pool for the fake game: the "box" tasks are winnable, the "drawer" ones not,
# so a corpus carries both outcomes.
FAKE_TASKS = ("put the key in the box", "put the key in the drawer") * 4


def fake_game_samples(suite_id: str) -> list[Any]:
    from wikitrace.envs.textgame.models import TextGameSample

    return [
        TextGameSample(
            id=f"alfworld_{n:04d}_fake", db_id=suite_id, task_type="pick_and_place_simple",
            variant=f"fake-{n}", task_text=task, spec={"task": task},
        )
        for n, task in enumerate(FAKE_TASKS)
    ]


def play_fake_game(*, user_msg: str, env_tool: Any, citation_layers: tuple = (), **_: Any):
    """A FakeTraverseRunner handler: win a "box" task, miss a "drawer" one, read one document."""
    from wikitrace.harness.environment.protocol import request
    from wikitrace.harness.terminal import TerminalResult

    target = "box" if "in the box" in user_msg else "drawer"
    commands = ["examine table", "take key", f"put key in {target}"]
    sock = str(env_tool.socket_dir / f"{env_tool.name}.sock")
    turns: list[dict] = []
    ctx: list[dict] = [{"role": "system", "content": "sys"}, {"role": "user", "content": user_msg}]
    for i, cmd in enumerate(commands):
        reply = request(sock, {"op": "step", "input": cmd})
        turns.append({"type": "function_call", "name": env_tool.name,
                      "arguments": {"command": cmd}, "call_id": f"c{i}"})
        ctx.append({"role": "assistant", "content": [{"type": "thinking", "thinking": f"try {cmd}"},
                                                     {"type": "toolCall", "id": f"c{i}"}]})
        ctx.append({"role": "toolResult", "toolCallId": f"c{i}",
                    "content": [{"type": "text", "text": reply["observation"]}]})
    return TerminalResult(final_text="done", turns=turns, final_context=ctx,
                          usage={"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.001})


@pytest.fixture
def fake_game(monkeypatch):
    """ALFWorld's adapter, prompts and vocabulary over the four-command fake game, offline."""
    import sys

    from wikitrace.envs.textgame.benchmarks.alfworld import AlfworldBenchmarkAdapter
    from wikitrace.harness.environment import fake
    from wikitrace.harness.terminal import FakeTraverseRunner
    from wikitrace.run import build, cell

    monkeypatch.setattr(AlfworldBenchmarkAdapter, "load_suite",
                        lambda self, dataset_cfg, suite_id: fake_game_samples(suite_id))
    monkeypatch.setattr(AlfworldBenchmarkAdapter, "env_factory", f"{fake.__file__}:make_env")
    monkeypatch.setattr(AlfworldBenchmarkAdapter, "python", sys.executable)
    monkeypatch.setattr(AlfworldBenchmarkAdapter, "verify_install", lambda self, cfg: None)
    runner = FakeTraverseRunner(play_fake_game)
    for module in (build, cell):
        monkeypatch.setattr(module, "_default_traverse_runner", lambda *a, **kw: runner)
    return runner


@pytest.fixture
def fake_transport_factory() -> Callable[..., FakeTransport]:
    """Return a factory that builds tunable FakeTransport instances."""

    def _make(**kwargs: Any) -> FakeTransport:
        return FakeTransport(**kwargs)

    return _make


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_config_path() -> Path:
    return BASE_CONFIG


@pytest.fixture
def base_config():
    return load_config(BASE_CONFIG)


@pytest.fixture
def temp_config_writer(tmp_path: Path) -> Callable[[dict], Path]:
    """Return a helper that writes a tmp config extending base.yaml."""

    def _write(overrides: dict, name: str = "tmp_config.yaml") -> Path:
        output_root = tmp_path / "outputs"
        raw: dict[str, Any] = {"extends": str(BASE_CONFIG)}
        merged = dict(overrides)
        exp = dict(merged.get("experiment", {}))
        exp.setdefault("output_root", str(output_root))
        merged["experiment"] = exp
        raw.update(merged)

        path = tmp_path / name
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=True)
        return path

    return _write


@pytest.fixture
def local_sockets_available(tmp_path):
    """Skip socket integration only when the environment forbids local binds."""
    import errno
    import socket
    for family, address in (
        (socket.AF_INET, ("127.0.0.1", 0)),
        (socket.AF_UNIX, str(tmp_path / "probe.sock")),
    ):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as probe:
                probe.bind(address)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EPERM, errno.EAFNOSUPPORT):
                pytest.skip(f"Local socket integration unavailable: {exc}")
            raise
        finally:
            if family == socket.AF_UNIX:
                Path(address).unlink(missing_ok=True)
