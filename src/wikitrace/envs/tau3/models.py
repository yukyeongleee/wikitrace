"""The tau3 result contract: textgame's, with each observation marked by who produced it."""

from __future__ import annotations

from pydantic import BaseModel, Field

from wikitrace.envs.textgame.models import Action, TextGameResult


class TauAction(Action):
    """One `take_action` step; `source` is the server's mark, `user` for a simulator reply, `env` otherwise."""

    source: str = ""


class TauMessage(BaseModel):
    """One message the agent sent the user and the reply; `after` counts the tool calls before it."""

    after: int = 0
    text: str = ""
    reasoning: str = ""
    # None when no reply came back: the simulator failed on this message.
    reply: str | None = None


class TauBenchResult(TextGameResult):
    """A tau3 episode's result; the renders label observations from `actions[*].source`."""

    actions: list[TauAction] = Field(default_factory=list)
    messages: list[TauMessage] = Field(default_factory=list)
