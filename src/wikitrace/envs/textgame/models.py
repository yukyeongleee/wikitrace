"""The textgame domain's sample, result and outcome contract (ALFWorld, ScienceWorld)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from wikitrace.config import CitationRecord, PhaseLog, ToolStep


class TextGameSample(BaseModel):
    """One episode to play: a task type, a variant of it, and the task text the agent reads."""

    id: str
    db_id: str
    task_type: str
    variant: str
    task_text: str
    split_role: str = ""
    stratum: str = ""
    spec: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def question(self) -> str:
        """The task text under the name the shared layers read a sample's task by."""
        return self.task_text


class Action(BaseModel):
    """One command the agent sent to the environment and what came back."""

    step: int
    command: str
    observation: str
    score: float = 0.0
    invalid: bool = False


class TextGameResult(BaseModel):
    """Outcome of one episode: the environment's final state plus the agent's trajectory."""

    sample_id: str
    db_id: str
    question: str
    task_type: str = ""
    variant: str = ""
    initial_observation: str = ""
    success: bool = False
    score: float = 0.0
    failed: bool = False
    steps: int = 0
    step_limit: int = 0
    step_limit_hit: bool = False
    invalid_commands: int = 0
    ended_by: str = ""
    ended_on_end_command: bool = False
    generation_error: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)
    actions: list[Action] = Field(default_factory=list)
    citations: list[CitationRecord] = Field(default_factory=list)
    tool_steps: list[ToolStep] = Field(default_factory=list)


@dataclass
class TextGameOutcome:
    """A completed episode: the scored result plus the one Pi phase log."""

    result: TextGameResult
    phase_logs: list[PhaseLog] = field(default_factory=list)
    generation_failed: bool = False
