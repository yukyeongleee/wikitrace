"""Plancraft benchmark adapter: Minecraft crafting in text-only mode, scored by the authors' `check_done`."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from wikitrace.envs.base import register_benchmark
from wikitrace.envs.textgame.benchmarks.base import RAW_METRIC_VOCABULARY, TextGameBenchmark
from wikitrace.envs.textgame.exemplars import load_exemplars
from wikitrace.envs.textgame.manifest import read_manifest, roled
from wikitrace.envs.textgame.models import TextGameSample

SUITES = {"plancraft-seen": "test_seen", "plancraft-unseen": "test_unseen"}


class PlancraftBenchmarkAdapter(TextGameBenchmark):
    """Plancraft @ 7617e9ff: the target in an inventory slot within 30 `take_action` calls."""

    name = "plancraft"
    config_dataset_keys = ("plancraft_root", "limit_explicit_split", "train_size", "test_size")
    step_limit = 30
    env_tool = "take_action"
    end_command = "impossible"
    end_phrase = "ended by the agent: it gave up"
    # The seeded search prints a set of item names, whose order follows the hash seed.
    server_env = (("PYTHONHASHSEED", "0"),)
    multiline_observations = True
    solver_intro = load_exemplars("plancraft")["system"]
    task_kind = "Minecraft crafting text games"
    raw_metric_vocabulary = replace(
        RAW_METRIC_VOCABULARY,
        step_contents=RAW_METRIC_VOCABULARY.step_contents.replace(
            "(an `act` call is a command", "(a `take_action` call is an action"
        ),
    )

    @staticmethod
    def train_group(suite_id: str) -> str:
        """Both suites share one train set, hence one wiki."""
        return "plancraft" if suite_id in SUITES else suite_id

    def exemplars_for(self, sample: TextGameSample) -> str:
        """The authors' one few-shot episode, a blank line below their system prompt."""
        return "\n" + load_exemplars("plancraft")["fewshot"]

    @staticmethod
    def task_block(observation: str, info: dict[str, Any]) -> str:
        """The reset observation already opens with the `Craft an item of type:` line."""
        return observation

    @staticmethod
    def initial_observation(observation: str, info: dict[str, Any]) -> str:
        """The inventory alone: the objective line is already the sample's question."""
        return "\n".join(ln for ln in observation.splitlines() if ln != info.get("task"))

    def load_suite(self, dataset_cfg: Any, suite_id: str) -> list[TextGameSample]:
        if suite_id not in SUITES:
            raise ValueError(f"unknown Plancraft suite id {suite_id!r}; expected one of {sorted(SUITES)}")
        self.root = Path(getattr(dataset_cfg, "plancraft_root", "data/plancraft")).resolve()
        manifest = read_manifest("plancraft")
        samples: list[TextGameSample] = []
        for n, (entry, role) in enumerate(roled(manifest, "train", SUITES[suite_id])):
            split = "train" if role == "train" else "test"
            samples.append(TextGameSample(
                id=f"plancraft_{n:04d}_{entry['id']}",
                db_id=suite_id, task_type="craft", variant=entry["id"],
                task_text=f"Craft an item of type: {entry['target']}", split_role=role,
                stratum=entry["complexity_split"],
                spec={"split": split, "example_id": entry["id"]},
                meta={"complexity_split": entry["complexity_split"], "split": split,
                      "optimal_path_length": entry["optimal_path_length"],
                      "oracle_steps": entry["oracle_steps"]},
            ))
        return samples


register_benchmark("plancraft", PlancraftBenchmarkAdapter)
