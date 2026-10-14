"""AppWorld benchmark adapter: frozen scenario manifests and TGC/SGC scoring."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from wikitrace.config import CitationRecord, ToolStep
from wikitrace.envs.base import WIKI_ROLES, HarnessSpec, Metrics, neutral_summary, register_benchmark

MAX_STEPS = 50
# A turn is one provider request and may carry read/bash calls, so it is a session guard, not the budget.
MAX_TURNS = 150
DEFAULT_ROOT = "data/appworld"
SUITES = {"test-normal": "seen_test", "test-challenge": "unseen_test"}

DATA_DIGEST = "427b1caff03bc0c3db7bee5c89e2bf5d81f3920af8ed4ff3558a8d314e496c5a"
INTEGRITY_FILE = "integrity.json"

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SPLITS_PATH = DATA_DIR / "appworld_splits.json"
EXEMPLAR_PATH = DATA_DIR / "react_code_agent_instructions.txt"
LICENSE_PATH = DATA_DIR / "LICENSE"


class AppWorldSample(BaseModel):
    """One AppWorld task: the supervisor's instruction, plus the split facts analysis needs."""

    id: str
    db_id: str
    question: str
    scenario_id: str
    difficulty: int
    split_role: str
    supervisor: dict[str, str] = Field(default_factory=dict)
    world_datetime: str = ""
    root: str = ""
    lenient_answer: bool = False


class AppWorldResult(BaseModel):
    """One episode's outcome: the evaluation numbers, the episode facts, the agent's calls."""
    # ``report`` describes the target state, so it never reaches ``raw/``.

    sample_id: str
    db_id: str
    question: str
    scenario_id: str = ""
    difficulty: int = 0
    supervisor: dict[str, str] = Field(default_factory=dict)
    world_datetime: str = ""
    tgc: bool = False
    tgc_lenient: bool | None = Field(default=None, exclude_if=lambda v: v is None)
    tests_passed: int = 0
    tests_total: int = 0
    steps: int = 0
    step_limit_hit: bool = False
    invalid_commands: int = 0
    ended_by: str = ""
    # The server's whole output per executed chunk; Pi's context may truncate or clear its copy.
    outputs: list[str] = Field(default_factory=list)
    report: dict[str, Any] = Field(default_factory=dict)
    generation_error: str = ""
    citations: list[CitationRecord] = Field(default_factory=list)
    tool_steps: list[ToolStep] = Field(default_factory=list)

    @property
    def success(self) -> bool:
        """The cohort's success signal: the lenient judgement when made, else official TGC."""
        return self.tgc if self.tgc_lenient is None else self.tgc_lenient


def lenient_tgc(tgc: bool, failures: list[dict], gold_answer_is_null: bool) -> bool:
    """TGC, or a task whose only failing test is the answer check against a null gold answer."""
    return tgc or (
        gold_answer_is_null and len(failures) == 1
        and str(failures[0].get("requirement", "")).startswith("assert answers match")
    )


def gold_answer_is_null(appworld_root: str | Path, task_id: str) -> bool:
    """Whether the task's gold answer (`ground_truth/answer.json`) is JSON null."""
    path = Path(appworld_root) / "data" / "tasks" / task_id / "ground_truth" / "answer.json"
    return json.loads(path.read_text(encoding="utf-8")) is None


def data_file_hashes() -> dict[str, str]:
    """sha256 per data file this benchmark reads at run time, keyed by its name."""
    # The exemplar sits outside the prompt fingerprint, so this is its only record.
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (EXEMPLAR_PATH, SPLITS_PATH) if p.exists()
    }


def _checked(rel: tuple[str, ...]) -> bool:
    """Whether a path under the install's ``data/`` is one an episode or its grading reads."""
    if "__pycache__" in rel or rel[-1].endswith(".pyc"):
        return False
    if rel == ("version.txt",) or rel[0] in ("api_docs", "base_dbs"):
        return True
    return rel[0] == "tasks" and len(rel) > 2 and (
        rel[2:] == ("specs.json",) or (rel[2] in ("dbs", "ground_truth") and len(rel) > 3))


def dataset_digests(install: str | Path) -> dict[str, str]:
    """sha256 per checked dataset file, keyed by its path under ``install``."""
    data = Path(install) / "data"
    files = [p for p in sorted(data.rglob("*")) if _checked(p.relative_to(data).parts) and p.is_file()]
    # Reading dominates on a network file system, and parallel reads overlap its latency.
    with ThreadPoolExecutor(16) as pool:
        sums = pool.map(lambda p: hashlib.sha256(p.read_bytes()).hexdigest(), files)
        return {p.relative_to(install).as_posix(): h for p, h in zip(files, sums)}


def aggregate_digest(digests: dict[str, str]) -> str:
    """One sha256 over every (path, sha256) pair, in path order."""
    return hashlib.sha256(
        "".join(f"{k}\0{v}\n" for k, v in sorted(digests.items())).encode("utf-8")).hexdigest()


def verify_dataset(install: str | Path) -> None:
    """Raise, naming the files that differ from ``integrity.json``, unless the install is the pinned one."""
    digests = dataset_digests(install)
    if aggregate_digest(digests) == DATA_DIGEST:
        return
    manifest = Path(install) / INTEGRITY_FILE
    recorded = json.loads(manifest.read_text(encoding="utf-8"))["files"] if manifest.exists() else {}
    differing = [
        f"{k} ({'missing' if k not in digests else 'added' if k not in recorded else 'changed'})"
        for k in sorted(digests.keys() | recorded.keys()) if digests.get(k) != recorded.get(k)
    ]
    raise RuntimeError(
        f"the AppWorld dataset under {install} does not match its pinned digest {DATA_DIGEST}; "
        + (f"{len(differing)} file(s) differ from {manifest}: {', '.join(differing[:20])}"
           + (" ..." if len(differing) > 20 else "") if manifest.exists()
           else f"{manifest} is absent, so the differing files cannot be named")
        + (" (none: the manifest itself is not the pinned one)" if manifest.exists() and not differing else "")
    )


def load_suite_samples(cfg: Any, suite_id: str) -> list[AppWorldSample]:
    """Load one suite from the frozen manifest: the shared train list, then its test list."""
    if suite_id not in SUITES:
        raise ValueError(f"unknown AppWorld suite id {suite_id!r}; expected one of {list(SUITES)}")
    install = Path(getattr(cfg, "appworld_root", DEFAULT_ROOT))
    root = install / "data"
    if not SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"AppWorld suite needs {SPLITS_PATH} — run "
            "scripts/setup/build_appworld_manifest.py"
        )
    if not (root / "tasks").is_dir():
        raise FileNotFoundError(
            f"AppWorld suite needs {root / 'tasks'} — run scripts/setup/appworld.sh"
        )
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    ordered = [(t, "train") for t in splits["train"]] + [(t, "test") for t in splits[SUITES[suite_id]]]

    def build(entry: dict[str, Any], role: str) -> AppWorldSample:
        specs = json.loads((root / "tasks" / entry["id"] / "specs.json").read_text(encoding="utf-8"))
        return AppWorldSample(
            id=entry["id"], db_id=suite_id, question=specs["instruction"],
            scenario_id=entry["scenario_id"], difficulty=int(entry["difficulty"]),
            split_role=role, supervisor={k: str(v) for k, v in specs["supervisor"].items()},
            world_datetime=str(specs.get("datetime", "")), root=str(install),
            lenient_answer=bool(getattr(cfg, "appworld_lenient_answer", False)),
        )

    return [build(entry, role) for entry, role in ordered]


class AppWorldBenchmarkAdapter:
    """AppWorld: TGC per task (all unit tests pass), SGC per scenario (all three tasks)."""

    name = "appworld"
    domain = "appworld"
    primary_metric_name = "tgc"
    secondary_metric_name = "tests"
    has_evidence = False
    # The Pi agent runs on `models.default`, so no generator role is read.
    config_dataset_keys = (
        "appworld_root", "limit_explicit_split", "train_size", "test_size",
        "appworld_lenient_answer",
    )
    config_roles = WIKI_ROLES
    task_kind = "app automation through Python APIs"

    @staticmethod
    def train_group(suite_id: str) -> str:
        """Both suites probe ONE train set, so they share a corpus and a wiki."""
        return "appworld" if suite_id in SUITES else suite_id

    def harness_spec(self, suite_id: str) -> HarnessSpec:
        return HarnessSpec(kind="environment", input_on_disk=False, max_turns=MAX_TURNS)

    data_file_hashes = staticmethod(data_file_hashes)

    def data_version(self, dataset_cfg: Any) -> str:
        """The installed dataset's `version.txt`, recorded once per run; empty when absent."""
        path = Path(getattr(dataset_cfg, "appworld_root", DEFAULT_ROOT)) / "data" / "version.txt"
        return path.read_text(encoding="utf-8").strip() if path.exists() else ""

    def verify_install(self, dataset_cfg: Any) -> None:
        """Refuse to run on a dataset that is not the pinned one or without the episode image."""
        install = Path(getattr(dataset_cfg, "appworld_root", DEFAULT_ROOT))
        verify_dataset(install)
        from wikitrace.envs.appworld.session import image_tag, require_image

        require_image(image_tag(install))


    def carries_task_context(self, suite_id: str) -> bool:
        return False


    def load_suite(self, dataset_cfg: Any, suite_id: str) -> list[AppWorldSample]:
        return load_suite_samples(dataset_cfg, suite_id)

    def score(self, sample: AppWorldSample, prediction: str) -> Any:  # pragma: no cover
        raise NotImplementedError("AppWorld scores the environment's final state, not a prediction")


    def grade(self, sample: AppWorldSample, result: AppWorldResult, llm: Any = None) -> None:
        return None

    def metrics_from_result(self, result: AppWorldResult) -> Metrics:
        extra: dict[str, Any] = {
            "scenario_id": result.scenario_id, "difficulty": result.difficulty,
            "tests_passed": result.tests_passed, "tests_total": result.tests_total,
        }
        if result.tgc_lenient is not None:
            extra["tgc_lenient"] = result.tgc_lenient
        return Metrics(
            primary=float(result.tgc),
            secondary=(result.tests_passed / result.tests_total) if result.tests_total else None,
            success=result.success,
            # SGC needs the scenario beside the sample metrics; a resumed run reads it back from there.
            extra=extra,
        )

    def summarize(self, metrics: list[Metrics]) -> dict[str, Any]:
        def rates(ok: list[bool]) -> tuple[float, float]:
            by_scenario: dict[str, list[bool]] = {}
            for m, o in zip(metrics, ok):
                by_scenario.setdefault(str(m.extra.get("scenario_id", "")), []).append(o)
            scenarios = [all(v) for v in by_scenario.values()]
            return (
                (sum(ok) / len(ok)) if ok else 0.0,
                (sum(scenarios) / len(scenarios)) if scenarios else 0.0,
            )

        tgc, sgc = rates([m.primary == 1.0 for m in metrics])
        out = {
            **neutral_summary(metrics), "tgc": tgc, "sgc": sgc,
            "n_scenarios": len({str(m.extra.get("scenario_id", "")) for m in metrics}),
        }
        if any("tgc_lenient" in m.extra for m in metrics):
            out["tgc_lenient"], out["sgc_lenient"] = rates([m.success for m in metrics])
        return out


register_benchmark("appworld", AppWorldBenchmarkAdapter)
