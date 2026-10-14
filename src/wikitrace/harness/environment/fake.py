"""A four-command text environment for offline tests and gate builds."""

from __future__ import annotations

from typing import Any

TASK = "put the key in the box"
ROOM = "You are in a small room. You see a table and a box."


class FakeEnvironment:
    """Deterministic: examine the table, take the key, put it in the box."""

    def __init__(self, task: str = TASK) -> None:
        self.task = task
        self.key_seen = False
        self.key_taken = False
        self.done = False

    def _info(self, invalid: bool = False) -> dict[str, Any]:
        score = 1.0 if self.done else (0.5 if self.key_taken else 0.0)
        return {"success": self.done, "score": score, "failed": False,
                "invalid_command": invalid, "task": self.task}

    def reset(self) -> tuple[str, dict[str, Any]]:
        return f"{ROOM}\nYour task is to: {self.task}.", self._info()

    def step(self, command: str) -> tuple[str, bool, dict[str, Any]]:
        cmd = " ".join(command.strip().lower().split())
        if cmd == "look":
            return ROOM, False, self._info()
        if cmd == "examine table":
            self.key_seen = True
            return "On the table you see a key.", False, self._info()
        if cmd == "take key" and not self.key_taken:
            self.key_taken = True
            return "You take the key from the table.", False, self._info()
        if cmd == "put key in box" and self.key_taken:
            self.done = True
            return "You put the key in the box. Task complete.", True, self._info()
        return "Nothing happens.", False, self._info(invalid=True)

    def close(self) -> None:
        return None

    def evaluation(self) -> dict[str, Any]:
        return {}


def make_env(spec: dict[str, Any]) -> FakeEnvironment:
    """Server-side factory; ``spec`` may carry a ``task`` string."""
    return FakeEnvironment(task=str(spec.get("task", TASK)))
