"""ScienceWorld benchmark adapter: elementary-science tasks scored by the simulator's goal progress."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wikitrace.envs.base import Metrics, register_benchmark
from wikitrace.envs.textgame.benchmarks.base import TextGameBenchmark
from wikitrace.envs.textgame.exemplars import load_exemplars, strip_think
from wikitrace.envs.textgame.manifest import read_manifest, roled
from wikitrace.envs.textgame.models import TextGameResult, TextGameSample

SUITE = "scienceworld"
# The reference loop's settings (`--simplification_str easy --env_step_limit 100`); the server enforces no action limit.
SIMPLIFICATION = "easy"
ENV_STEP_LIMIT = 100


class ScienceworldBenchmarkAdapter(TextGameBenchmark):
    """ScienceWorld 1.3.0 under `easy`: `success` is the simulator's completion flag, `score` its progress."""

    name = "scienceworld"
    config_dataset_keys = ("scienceworld_root", "limit_explicit_split", "train_size", "test_size")
    secondary_metric_name = "score"
    max_turns = 400
    solver_intro = "Interact with a household to solve a task. Here is an example."
    task_kind = "elementary science text games"

    def exemplars_for(self, sample: TextGameSample) -> str:
        """The harness's one exemplar for this task type, keyed by the simulator's task index."""
        return strip_think(load_exemplars("scienceworld")[str(sample.meta["task_index"])])

    @staticmethod
    def task_block(observation: str, info: dict[str, Any]) -> str:
        """The harness's task block: the reset observation, then the task description."""
        return f"{observation}\n{info.get('task', '')}"

    def load_suite(self, dataset_cfg: Any, suite_id: str) -> list[TextGameSample]:
        if suite_id != SUITE:
            raise ValueError(f"unknown ScienceWorld suite id {suite_id!r}; expected {SUITE!r}")
        self.root = Path(getattr(dataset_cfg, "scienceworld_root", "data/scienceworld")).resolve()
        java_home = self.root / "jdk"
        if not (java_home / "bin" / "java").exists():
            raise FileNotFoundError(
                f"ScienceWorld needs {java_home} — run scripts/setup/scienceworld.sh"
            )
        manifest = read_manifest("scienceworld")
        samples: list[TextGameSample] = []
        for n, (entry, role) in enumerate(roled(manifest, "train", "test")):
            samples.append(TextGameSample(
                id=f"scienceworld_{n:04d}_{entry['task']}-var{entry['variation']}",
                db_id=suite_id, task_type=entry["task"], variant=str(entry["variation"]),
                task_text=entry["description"], split_role=role, stratum=entry["task"],
                spec={"task": entry["task"], "variation": entry["variation"],
                      "simplification": SIMPLIFICATION, "env_step_limit": ENV_STEP_LIMIT,
                      "java_home": str(java_home)},
                meta={"task_index": entry["task_index"], "variation": entry["variation"],
                      "description": entry["description"], "split": entry["split"]},
            ))
        return samples

    def metrics_from_result(self, result: TextGameResult) -> Metrics:
        m = super().metrics_from_result(result)
        m.extra["failed"] = result.failed
        return m

    def summarize(self, metrics: list[Metrics]) -> dict[str, Any]:
        n = len(metrics)
        return {
            **super().summarize(metrics),
            "mean_score": (sum(m.secondary or 0.0 for m in metrics) / n) if n else 0.0,
            "failed_rate": (sum(1 for m in metrics if m.extra.get("failed")) / n) if n else 0.0,
        }


register_benchmark("scienceworld", ScienceworldBenchmarkAdapter)
