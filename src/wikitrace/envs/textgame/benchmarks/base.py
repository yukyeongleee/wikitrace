"""What the two textgame benchmarks share: the environment harness, the manifest split, the metric."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from wikitrace.envs.base import WIKI_ROLES, HarnessSpec, Metrics, neutral_summary, steps_text
from wikitrace.envs.textgame.exemplars import data_file_hashes
from wikitrace.envs.textgame.manifest import interpreter
from wikitrace.envs.textgame.models import TextGameResult, TextGameSample
from wikitrace.memory.store import RawMetricVocabulary

ENVS_DIR = Path(__file__).resolve().parents[1] / "envs"

# The wiki sees the binary outcome alone; a simulator's progress score stays in the run metrics.
RAW_METRIC_VOCABULARY = RawMetricVocabulary(
    step_contents=(
        "Each `/work/corpus/raw/step-NNNN/` holds: `metadata.md` (sample id, task_type,\n"
        "instruction, initial_observation, success, steps); the agent turn log\n"
        "`solver.full.jsonl` (an `act` call is a command and its `result` the observation)."
    ),
    step_contents_outcome_only=(
        "Each `/work/corpus/raw/step-NNNN/` holds: `metadata.md` (sample id, task_type,\n"
        "instruction, initial_observation, success, steps)."
    ),
    metadata_keys=("task_type", "initial_observation", "success", "steps"),
    index_score="",
    index_score_pattern="",
    index_score_columns="",
    index_metric_names="success",
    compilation_layer_desc="(outcome, one-line summary)",
    compilation_task_hint="task_type, instruction, initial_observation",
    compilation_outcome_hint="what success does/doesn't establish (no partial credit)",
    compilation_correct_parts_hint=(
        "(failure only) subgoals the commands did reach (no partial-credit score — "
        "judge from the observations)"
    ),
    compilation_description="{outcome} analysis.",
    compilation_metric_keys=(),
    compilation_fm_metric_lines=(),
    writer_metric_header=(),
)


class TextGameBenchmark:
    """Base of the ALFWorld and ScienceWorld adapters; a subclass states its own data and metric."""

    name = ""
    domain = "textgame"
    primary_metric_name = "success"
    secondary_metric_name = ""
    has_evidence = False
    # The Pi agent runs on `models.default`, so no generator role is read.
    config_roles = WIKI_ROLES
    # Enforced by the environment server in every method; 0 leaves the cutoff to the simulator's clock.
    step_limit = 0
    env_tool = "act"
    # A harness addition: the reference ReAct loops have no give-up path.
    end_command = "done"
    # The steps line's phrase for an episode the end command ended; empty keeps "ended by the agent".
    end_phrase = ""
    server_env: tuple[tuple[str, str], ...] = ()
    multiline_observations = False
    # A turn is one provider request and may carry read/bash calls, so it is a session guard, not the budget.
    max_turns = 150
    root: Path | None = None
    solver_intro = ""
    task_kind = ""
    raw_metric_vocabulary = RAW_METRIC_VOCABULARY

    def harness_spec(self, suite_id: str) -> HarnessSpec:
        return HarnessSpec(kind="environment", input_on_disk=False, max_turns=self.max_turns)

    @staticmethod
    def initial_observation(observation: str, info: dict[str, Any]) -> str:
        """What the reset observation says about the world, the task sentence excluded."""
        return observation

    @property
    def env_factory(self) -> str:
        return f"{ENVS_DIR / self.name}.py:make_env"

    @property
    def python(self) -> str:
        """The benchmark venv's interpreter, which runs the environment server."""
        if self.root is None:
            raise RuntimeError(f"{self.name}: load_suite must run before an episode starts")
        return interpreter(self.root, f"scripts/setup/{self.name}.sh")

    def verify_install(self, dataset_cfg: Any) -> None:
        """Refuse to run without the benchmark venv, which ``load_suite`` located."""
        _ = self.python

    def data_file_hashes(self) -> dict[str, str]:
        """The exemplar and manifest hashes: outside the prompt fingerprint, recorded in run.json."""
        return data_file_hashes(self.name)


    def carries_task_context(self, suite_id: str) -> bool:
        return False


    def steps_text(self, result: TextGameResult) -> str:
        """The `steps` bullet: tool calls, and what ended an episode the task did not."""
        return steps_text(
            result.steps, result.ended_by, tool=self.env_tool,
            end_phrase=self.end_phrase if result.ended_on_end_command else "",
        )

    def score(self, sample: TextGameSample, prediction: str) -> Any:  # pragma: no cover
        raise NotImplementedError("a textgame episode is scored by its environment")


    def grade(self, sample: TextGameSample, result: TextGameResult, llm: Any = None) -> None:
        return None

    def metrics_from_result(self, result: TextGameResult) -> Metrics:
        return Metrics(
            primary=float(result.success),
            secondary=result.score if self.secondary_metric_name else None,
            success=result.success,
            extra={"task_type": result.task_type},
        )

    def summarize(self, metrics: list[Metrics]) -> dict[str, Any]:
        n_by_type = Counter(m.extra.get("task_type", "") for m in metrics)
        won_by_type = Counter(m.extra.get("task_type", "") for m in metrics if m.success)
        return {
            **neutral_summary(metrics),
            "success_rate_by_task_type": {
                t: won_by_type[t] / n for t, n in sorted(n_by_type.items()) if n
            },
        }
