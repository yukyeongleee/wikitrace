"""ScienceWorld: one (task, variation) episode on its own JVM behind the Environment protocol."""
# Runs under data/scienceworld/.venv/bin/python: standard library plus the simulator only.

from __future__ import annotations

import os
from typing import Any

INVALID = "No known action matches that input."
_TASK_PREFIX_LEN = 18


def clean(s: str) -> str:
    """The SwiftSage harness's observation filter, verbatim: one line per observation."""
    for tok in ("\n", "\t"):
        s = s.replace(tok, " ")
    return s


_SUBGOAL_SECTIONS = {"Sequential Subgoals:": "sequential", "Unordered and Optional Subgoals:": "optional"}


def parse_goal_progress(text: str) -> list[tuple[str, str, str, bool]]:
    """``(kind, class, name, achieved)`` per subgoal, in the simulator's order."""
    # The class is kept because some goal classes carry an empty description.
    kind = ""
    out: list[tuple[str, str, str, bool]] = []
    for line in text.splitlines():
        if line.strip() in _SUBGOAL_SECTIONS:
            kind = _SUBGOAL_SECTIONS[line.strip()]
            continue
        parts = line.split("\t")
        if kind and len(parts) == 4 and parts[0].isdigit():
            out.append((kind, parts[2].strip(), parts[3].strip(), parts[1] == "true"))
    return out


class ScienceworldEnvironment:
    """One episode; the score is the simulator's progress, frozen at the last non-negative value."""
    # The `--no_stop` rule of the reference loop: a wrong `focus on` makes the simulator
    # report a negative score and end the task, and the episode keeps its last good score.

    def __init__(self, task: str, variation: int, simplification: str, env_step_limit: int) -> None:
        from scienceworld import ScienceWorldEnv

        self.env = ScienceWorldEnv("", envStepLimit=env_step_limit)
        self.env.load(task, variation, simplification)
        self.env_step_limit = env_step_limit
        self.score = 0.0
        self.failed = False
        self.moves = 0
        self.actions = 0
        self.subgoals: list[tuple[str, str, str]] = []
        self.achieved_at: list[int | None] = []

    def _track_subgoals(self) -> None:
        progress = parse_goal_progress(self.env.get_goal_progress())
        if not self.subgoals:
            self.subgoals = [(kind, cls, name) for kind, cls, name, _ in progress]
            self.achieved_at = [None] * len(progress)
        for i, (_kind, _cls, _name, achieved) in enumerate(progress[:len(self.achieved_at)]):
            if achieved and self.achieved_at[i] is None:
                self.achieved_at[i] = self.actions

    def _info(self, invalid: bool = False) -> dict[str, Any]:
        success = bool(self.env.server.getCompleted())
        return {
            "success": success,
            "score": self.score, "failed": self.failed, "invalid_command": invalid,
            "env_steps": self.moves, "env_step_limit": self.env_step_limit,
            "step_limit_hit": self.moves > self.env_step_limit and not success,
        }

    def reset(self) -> tuple[str, dict[str, Any]]:
        obs, _info = self.env.reset()
        task = self.env.taskdescription()[_TASK_PREFIX_LEN:]
        self._track_subgoals()
        return clean(obs), {**self._info(), "task": task}

    def step(self, command: str) -> tuple[str, bool, dict[str, Any]]:
        obs, _reward, done, info = self.env.step(command)
        self.actions += 1
        self.moves = int(info["moves"])
        raw = int(info["score"])
        if raw < 0:
            self.failed = True
            done = True
        else:
            self.score = raw / 100.0
        self._track_subgoals()
        return clean(obs), bool(done), self._info(invalid=(obs.strip() == INVALID))

    def close(self) -> None:
        self.env.close()

    def evaluation(self) -> dict[str, Any]:
        """Partial credit and the subgoal timeline, from the progress read at each action."""
        timeline = [
            {"kind": kind, "class": cls, "name": name, "achieved_at_step": at}
            for (kind, cls, name), at in zip(self.subgoals, self.achieved_at)
        ]
        subgoals: dict[str, Any] = {}
        for kind in ("sequential", "optional"):
            of_kind = [t for t in timeline if t["kind"] == kind]
            subgoals[kind] = {
                "achieved": sum(t["achieved_at_step"] is not None for t in of_kind),
                "total": len(of_kind),
            }
        subgoals["timeline"] = timeline
        return {"score": self.score, "subgoals": subgoals}


def make_env(spec: dict[str, Any]) -> ScienceworldEnvironment:
    """``spec``: ``task``, ``variation``, ``simplification``, ``env_step_limit``, ``java_home``."""
    # py4j launches whichever `java` is first on PATH; the installed JRE goes there.
    java_home = spec.get("java_home")
    if java_home:
        os.environ["PATH"] = f"{java_home}/bin{os.pathsep}{os.environ.get('PATH', '')}"
    return ScienceworldEnvironment(
        str(spec["task"]), int(spec["variation"]), str(spec.get("simplification", "easy")),
        int(spec.get("env_step_limit", 100)),
    )
