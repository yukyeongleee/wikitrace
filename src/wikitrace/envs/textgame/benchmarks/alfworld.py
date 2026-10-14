"""ALFWorld benchmark adapter: household tasks in TextWorld, scored by the game's own goal."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wikitrace.envs.base import register_benchmark
from wikitrace.envs.textgame.benchmarks.base import TextGameBenchmark
from wikitrace.envs.textgame.exemplars import load_exemplars, strip_think
from wikitrace.envs.textgame.manifest import read_manifest, roled
from wikitrace.envs.textgame.models import TextGameSample

SUITES = {"valid-seen": "valid_seen", "valid-unseen": "valid_unseen"}

PREFIXES = {
    "pick_and_place": "put", "pick_clean_then_place": "clean", "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool", "look_at_obj": "examine", "pick_two_obj": "puttwo",
}


class AlfworldBenchmarkAdapter(TextGameBenchmark):
    """ALFWorld (TextWorld): `won` at some step within the 49-command budget."""

    name = "alfworld"
    config_dataset_keys = ("alfworld_root", "limit_explicit_split", "train_size", "test_size")
    step_limit = 49
    solver_intro = "Interact with a household to solve a task. Here are two examples."
    task_kind = "household text games"

    @staticmethod
    def train_group(suite_id: str) -> str:
        """Both suites share one train set, hence one wiki."""
        return "alfworld" if suite_id in SUITES else suite_id

    def exemplars_for(self, sample: TextGameSample) -> str:
        """The two ReAct exemplars of this task type, in the notebook's order, minus `think:`."""
        d = load_exemplars("alfworld")
        v = next(v for k, v in PREFIXES.items() if sample.task_type.startswith(k))
        return strip_think(d[f"react_{v}_1"] + d[f"react_{v}_0"])

    @staticmethod
    def task_block(observation: str, info: dict[str, Any]) -> str:
        """The reset observation already carries the `Your task is to:` line."""
        return observation

    @staticmethod
    def initial_observation(observation: str, info: dict[str, Any]) -> str:
        """The room description alone: the task line is already the sample's question."""
        return "\n".join(ln for ln in observation.splitlines() if not ln.startswith("Your task is to:"))

    def load_suite(self, dataset_cfg: Any, suite_id: str) -> list[TextGameSample]:
        if suite_id not in SUITES:
            raise ValueError(f"unknown ALFWorld suite id {suite_id!r}; expected one of {sorted(SUITES)}")
        self.root = Path(getattr(dataset_cfg, "alfworld_root", "data/alfworld")).resolve()
        games = self.root / "json_2.1.1"
        if not games.is_dir():
            raise FileNotFoundError(f"ALFWorld needs {games} — run scripts/setup/alfworld.sh")
        manifest = read_manifest("alfworld")
        samples: list[TextGameSample] = []
        for n, (entry, role) in enumerate(roled(manifest, "train", SUITES[suite_id])):
            game_file = self.root / entry["game"] / "game.tw-pddl"
            if not game_file.exists():
                raise FileNotFoundError(
                    f"ALFWorld manifest names {entry['game']}, absent under {self.root} — "
                    "the manifest and the installed data have diverged"
                )
            samples.append(TextGameSample(
                id=f"alfworld_{n:04d}_{Path(entry['game']).parent.name}",
                db_id=suite_id, task_type=entry["task_type"], variant=entry["game"],
                task_text=entry["task_text"], split_role=role, stratum=entry["task_type"],
                spec={"game_file": str(game_file), "alfworld_data": str(self.root)},
                meta={"scene_id": entry["scene_id"], "split": entry["split"]},
            ))
        return samples


register_benchmark("alfworld", AlfworldBenchmarkAdapter)
