"""ALFWorld: one TextWorld game file behind the Environment protocol."""
# Runs under data/alfworld/.venv/bin/python: standard library plus the simulator only.

from __future__ import annotations

import os
from typing import Any

INVALID = "Nothing happens."
# Room for the server's own 49-action limit: TextWorld must not end the episode first.
_TEXTWORLD_EPISODE_STEPS = 1000


def process_ob(ob: str) -> str:
    """The ReAct notebook's observation filter (cell 1), verbatim."""
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2:]
    return ob


class AlfworldEnvironment:
    """One ALFWorld game; ``won`` from the game's own PDDL goal is the whole score."""

    def __init__(self, game_file: str) -> None:
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        # Mirrors AlfredTWEnv.init_env for an eval split over this one file, without domain randomization.
        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        env_id = textworld.gym.register_game(
            game_file, request_infos, max_episode_steps=_TEXTWORLD_EPISODE_STEPS,
            wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
        )
        self.env = textworld.gym.make(env_id)
        self.won = False

    def _info(self, invalid: bool = False) -> dict[str, Any]:
        return {"success": self.won, "score": float(self.won), "failed": False,
                "invalid_command": invalid}

    def reset(self) -> tuple[str, dict[str, Any]]:
        ob, _info = self.env.reset()
        ob = "\n".join(ob.split("\n\n")[1:])
        task = next((ln for ln in ob.split("\n") if ln.startswith("Your task is to:")), "")
        return ob, {**self._info(), "task": task}

    def step(self, command: str) -> tuple[str, bool, dict[str, Any]]:
        ob, _score, done, info = self.env.step(command)
        ob = process_ob(ob)
        self.won = bool(info["won"])
        return ob, bool(done) or self.won, self._info(invalid=(ob.strip() == INVALID))

    def close(self) -> None:
        self.env.close()

    def evaluation(self) -> dict[str, Any]:
        """Nothing beyond the head: the simulator reports ``won`` only."""
        return {}


def make_env(spec: dict[str, Any]) -> AlfworldEnvironment:
    """``spec``: ``game_file`` (absolute) and ``alfworld_data`` (the package reads it at import)."""
    os.environ["ALFWORLD_DATA"] = str(spec["alfworld_data"])
    return AlfworldEnvironment(str(spec["game_file"]))
