"""AppWorld as an Environment: one task's Python shell behind `exec(code)`, graded apart."""
# Loaded by file path in the episode container and run as a script on the host to grade a finished episode, so it
# imports only the standard library and the simulator; APPWORLD_ROOT is set BEFORE `appworld` is imported.

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

_EXECUTION_FAILED = "Execution failed."
EXPERIMENT_NAME = "wikitrace"


def _use_root(root: str) -> None:
    os.environ["APPWORLD_ROOT"] = root
    os.environ.setdefault("APPWORLD_CACHE", os.path.join(root, ".cache"))


class AppWorldEnvironment:

    """One episode: `AppWorld.execute` per step, run without the task's ground truth."""
    def __init__(self, spec: dict[str, Any]) -> None:
        _use_root(str(Path(spec["appworld_root"]).resolve()))
        from appworld import AppWorld

        self.task_id = str(spec["task_id"])
        self.world = AppWorld(
            task_id=self.task_id, experiment_name=EXPERIMENT_NAME, load_ground_truth=False,
        )

    def _info(self, invalid: bool = False) -> dict[str, Any]:
        return {
            "success": False, "score": 0.0, "failed": False, "invalid_command": invalid,
            "task": self.world.task.instruction,
        }

    def reset(self) -> tuple[str, dict[str, Any]]:
        task = self.world.task
        info = self._info()
        info["supervisor"] = dict(task.supervisor)
        info["world_datetime"] = task.datetime.isoformat()
        info["app_descriptions"] = json.dumps(
            [{"name": k, "description": v} for k, v in task.app_descriptions.items()],
            indent=1,
        )
        return "", info

    def step(self, command: str) -> tuple[str, bool, dict[str, Any]]:
        output = self.world.execute(command)
        done = bool(self.world.task_completed())
        return output, done, self._info(invalid=output.startswith(_EXECUTION_FAILED))

    def close(self) -> None:
        try:
            self.world.save()
        finally:
            self.world.close()


def make_env(spec: dict[str, Any]) -> AppWorldEnvironment:
    """Server-side factory: ``spec`` names ``appworld_root`` and ``task_id``."""
    return AppWorldEnvironment(spec)


def _refuse_attach() -> None:
    """Make every SQLite connection this process opens refuse ATTACH, and with it VACUUM INTO."""
    # Replaying the saved change log on the host must reach no file but the databases the grader loads.
    import sqlite3

    connect = sqlite3.connect

    def guarded(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection = connect(*args, **kwargs)
        connection.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
        return connection

    sqlite3.connect = sqlite3.dbapi2.connect = guarded


def evaluate(root: str, task_id: str) -> dict[str, Any]:
    """Grade the state an episode saved under ``root`` against the task's gold: the evaluation block."""
    _use_root(root)
    _refuse_attach()
    from appworld.evaluator import evaluate_task

    # No report on disk: the outputs dir is the episode's, so a path there may be its link.
    tracker = evaluate_task(
        task_id=task_id, experiment_name=EXPERIMENT_NAME, suppress_errors=True, save_report=False,
    )
    return {"success": bool(tracker.success), "tests": {
        "passed": tracker.pass_count, "total": tracker.num_tests,
        "passes": [dict(p) for p in tracker.passes],
        "failures": [dict(f) for f in tracker.failures],
    }}


if __name__ == "__main__":
    print(json.dumps(evaluate(sys.argv[1], sys.argv[2]), ensure_ascii=False))
