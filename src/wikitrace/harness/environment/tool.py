"""The environment tool a terminal session exposes, and the socket its extension dials."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

EXTENSIONS_DIR = Path(__file__).resolve().parents[1] / "terminal" / "extensions"
SOCKET_ENV_VAR = "AGENT_WIKI_ENV_SOCKET"
SOCKET_NAME = "act.sock"
# Queued from inside Pi, since a steer the runner writes in reaction to an event lands one turn late.
CITATION_ENV_VAR = "AGENT_WIKI_CITATION_PROMPT"


@dataclass(frozen=True)
class EnvTool:
    """One environment tool: `act` or `take_action` for a game, `exec(code)` for a task shell."""

    name: Literal["act", "exec", "take_action"]
    socket_dir: Path
    extension_name: str = ""
    # A reply without a tool call goes to the environment as a message to the user, whose reply continues the session.
    messages: bool = False

    @property
    def extension(self) -> Path:
        """The self-contained Pi extension that registers this tool."""
        return EXTENSIONS_DIR / f"{self.extension_name or self.name}.ts"


JSON_STEP_EXTENSIONS = ("take_action_tau3",)


def step_input(tool: str, args: dict, extension: str = "") -> str:
    """The one step string a game tool's extension sends for these call arguments."""
    if extension in JSON_STEP_EXTENSIONS:
        return json.dumps({"action": args.get("action", ""), "argument": args.get("argument", "")},
                          ensure_ascii=False, separators=(",", ":"))
    if tool == "take_action":
        return f"{args.get('action', '')}: {args.get('argument', '')}"
    return str(args.get("command", ""))


def close_reason(stopped_by: str) -> str:
    """The `close` reason for a session the runner reports as ``stopped_by``."""
    return "harness_stopped" if stopped_by else "agent_stopped"


def blocked_by_hook(result: str, tool: str) -> bool:
    """Is this tool result the extension's own one-call-per-turn refusal, never served?"""
    return result.startswith(f"one {tool} per turn")
