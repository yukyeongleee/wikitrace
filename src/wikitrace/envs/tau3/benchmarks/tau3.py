"""tau3 retail benchmark adapter: the authors' 114 retail tasks under our frozen split, scored by their evaluator."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from wikitrace.envs.base import HarnessSpec, register_benchmark, steps_text
from wikitrace.envs.textgame.benchmarks.base import RAW_METRIC_VOCABULARY, TextGameBenchmark
from wikitrace.envs.textgame.manifest import interpreter, roled
from wikitrace.envs.textgame.models import TextGameResult, TextGameSample

SUITE = "retail"
DEFAULT_ROOT = "data/tau3"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MANIFEST_PATH = DATA_DIR / "tau3.manifest.json"
CLARIFIED_GUIDELINES_PATH = DATA_DIR / "user_guidelines_clarified.md"
PROMPT_DATA_PATH = DATA_DIR / "tau3_tools.json"
ENVS_DIR = Path(__file__).resolve().parents[1] / "envs"
TAU2_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"

# Fixed whatever model the solver or the memory agents run on, so a solver change leaves the other side alone.
SIMULATOR_MODEL = "openrouter/openai/gpt-5.4-mini"
# An `openrouter/` call is one httpx attempt plus `num_retries` from litellm, so a hung call ends after 2 x 35 s,
# inside the runner's 120 s idle kill (which counts while `take_action` waits): a simulator error, not a stop.
SIMULATOR_LLM_ARGS: dict[str, Any] = {"reasoning_effort": "medium", "max_tokens": 30000, "timeout": 35, "num_retries": 1}
JUDGE_MODEL = SIMULATOR_MODEL
JUDGE_LLM_ARGS = dict(SIMULATOR_LLM_ARGS)

_METADATA_KEYS = ("task_type", "success", "steps")
_METADATA_LINE = "`metadata.md` (sample id, task_type,\nuser_request, success, steps)"
_STEP_LOG = ("(a `take_action` call is a retail tool call and its `result` the tool output; an\n"
             "assistant `message` row is a message to the user, and the user `message` row after\n"
             "it the user's reply)")
_END_PHRASE = {
    "task_failed": "ended by the environment: too many tool errors",
    "game_time_exhausted": "ended by the environment: the conversation's step budget ran out",
}


def read_manifest() -> dict[str, Any]:
    """The committed ``data/tau3.manifest.json`` (scripts/setup/build_tau3_manifest.py writes it)."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} is missing — run scripts/setup/build_tau3_manifest.py")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def read_prompt_data() -> dict[str, Any]:
    """The authors' instruction, policy and tool schemas the solver prompt is rendered from."""
    if not PROMPT_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{PROMPT_DATA_PATH} is missing — run scripts/setup/build_tau3_prompt_data.py")
    return json.loads(PROMPT_DATA_PATH.read_text(encoding="utf-8"))


def data_file_hashes() -> dict[str, str]:
    """sha256 per data file this benchmark reads at run time, keyed by its name."""
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (MANIFEST_PATH, PROMPT_DATA_PATH) if p.exists()}


class Tau3BenchmarkAdapter(TextGameBenchmark):
    """tau3 retail @ fc0055dc: one customer conversation, the reward the product over the task's basis."""

    name = "tau3"
    domain = "tau3"
    config_dataset_keys = ("tau3_root", "limit_explicit_split", "train_size", "test_size",
                           "tau3_user_model", "tau3_user_guidelines")
    config_omit_at_default = ("tau3_user_model", "tau3_user_guidelines")
    step_limit = 0
    env_tool = "take_action"
    env_extension = "take_action_tau3"
    end_command = ""
    message_channel = True
    multiline_observations = True
    task_kind = "retail customer service with tools"
    raw_metric_vocabulary = replace(
        RAW_METRIC_VOCABULARY,
        step_contents=RAW_METRIC_VOCABULARY.step_contents.replace(
            "`metadata.md` (sample id, task_type,\ninstruction, initial_observation, success, steps)", _METADATA_LINE,
        ).replace("(an `act` call is a command and its `result` the observation)", _STEP_LOG),
        step_contents_outcome_only=RAW_METRIC_VOCABULARY.step_contents_outcome_only.replace(
            "`metadata.md` (sample id, task_type,\ninstruction, initial_observation, success, steps)", _METADATA_LINE,
        ),
        metadata_keys=_METADATA_KEYS,
        compilation_task_hint="task_type, user_request",
    )

    @property
    def env_factory(self) -> str:
        return f"{ENVS_DIR / self.name}.py:make_env"

    @property
    def python(self) -> str:
        """The benchmark venv's interpreter, which runs the environment server."""
        if self.root is None:
            raise RuntimeError(f"{self.name}: load_suite must run before an episode starts")
        return interpreter(self.root, "scripts/setup/tau3.sh")

    data_file_hashes = staticmethod(data_file_hashes)

    def verify_install(self, dataset_cfg: Any) -> None:
        """Refuse to run on a clone off the pinned commit, a venv not synced from its lock, or no venv."""
        root = Path(getattr(dataset_cfg, "tau3_root", self.root or DEFAULT_ROOT)).resolve()
        hint = "run scripts/setup/tau3.sh"
        head_file = root / "repo" / ".git" / "HEAD"
        head = head_file.read_text(encoding="utf-8").strip() if head_file.exists() else ""
        if head != TAU2_COMMIT:
            raise RuntimeError(f"{root / 'repo'} is at {head or 'no commit'!r}, not the pinned {TAU2_COMMIT}; {hint}")
        lock, mark = root / "repo" / "uv.lock", root / ".venv" / ".uv-lock.sha256"
        synced = mark.read_text(encoding="utf-8").strip() if mark.exists() else ""
        if not lock.exists() or synced != hashlib.sha256(lock.read_bytes()).hexdigest():
            raise RuntimeError(f"{root / '.venv'} was not synced from {lock}; {hint}")
        interpreter(root, "scripts/setup/tau3.sh")

    def harness_spec(self, suite_id: str) -> HarnessSpec:
        return replace(super().harness_spec(suite_id), task_text_at_reset=True)

    def steps_text(self, result: TextGameResult) -> str:
        """The `steps` bullet: `take_action` calls, and what ended a conversation the user did not."""
        return steps_text(result.steps, result.ended_by, tool=self.env_tool,
                          end_phrase=_END_PHRASE.get(result.ended_by, ""))

    def exemplars_for(self, sample: TextGameSample) -> str:
        """No few-shot: the authors' agent has none."""
        return ""

    @staticmethod
    def task_block(observation: str, info: dict[str, Any]) -> str:
        """The user's first message is the whole task."""
        return observation

    @staticmethod
    def initial_observation(observation: str, info: dict[str, Any]) -> str:
        """Nothing beside the task: the first observation IS the user's first message."""
        return ""

    @staticmethod
    def spec_for(task_id: str, user_model: str | None = None, guidelines: str = "official") -> dict[str, Any]:
        """What `make_env` needs: the task and the user and judge model settings."""
        user_args = dict(SIMULATOR_LLM_ARGS)
        if user_model:
            user_args.pop("reasoning_effort")
            user_args["temperature"] = 0.0
        spec: dict[str, Any] = {
            "task_id": task_id, "user_model": user_model or SIMULATOR_MODEL, "user_llm_args": user_args,
            "judge_model": JUDGE_MODEL, "judge_llm_args": dict(JUDGE_LLM_ARGS)}
        if guidelines == "clarified":
            spec["user_guidelines_path"] = str(CLARIFIED_GUIDELINES_PATH)
        return spec

    def load_suite(self, dataset_cfg: Any, suite_id: str) -> list[TextGameSample]:
        if suite_id != SUITE:
            raise ValueError(f"unknown tau3 suite id {suite_id!r}; expected {SUITE!r}")
        self.root = Path(getattr(dataset_cfg, "tau3_root", DEFAULT_ROOT)).resolve()
        samples: list[TextGameSample] = []
        for n, (entry, role) in enumerate(roled(read_manifest(), "train", "test")):
            task_id = str(entry["task_id"])
            samples.append(TextGameSample(
                id=f"tau3_{n:04d}_{task_id}", db_id=suite_id, task_type=SUITE, variant=task_id,
                task_text="", split_role=role, stratum=SUITE, spec=self.spec_for(
                    task_id, getattr(dataset_cfg, "tau3_user_model", None),
                    getattr(dataset_cfg, "tau3_user_guidelines", "official")),
                meta={"task_id": task_id, "user_id": entry["user_id"], "split": role},
            ))
        return samples


register_benchmark("tau3", Tau3BenchmarkAdapter)
