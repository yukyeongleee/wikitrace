"""Output paths, run.json, the cohort fingerprint and the run summary."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from wikitrace.config import (
    AppConfig, ConfigView, ModelsConfig, aggregate_llm_calls, config_hash, provider_counts,
)
from wikitrace.harness.prompts import prompts_source_hash
from wikitrace.memory.citation import CITATION_ROLE


def corpus_dir_for(output_root: str | Path, benchmark: str, train_group: str, trial: int) -> Path:
    """``<output_root>/corpus/<benchmark>/<train_group>/rep-<NN>``."""
    return Path(output_root) / "corpus" / benchmark / train_group / f"rep-{trial:02d}"


def run_dir_for(output_root: str | Path, benchmark: str, suite: str, method: str, trial: int) -> Path:
    """``<output_root>/<benchmark>/<suite>/<method>/trial-<NN>``."""
    return Path(output_root) / benchmark / suite / method / f"trial-{trial:02d}"


def run_fingerprint(config: AppConfig, git: dict[str, Any], view: ConfigView) -> dict[str, str]:
    """The code, config and prompts a run or a corpus was produced under."""
    cfg_h = config_hash(config, view)
    prompts_h = prompts_source_hash()
    commit = str(git.get("commit", ""))
    dirty = str(bool(git.get("dirty", False)))
    full = hashlib.sha256("\0".join([cfg_h, prompts_h, commit, dirty]).encode("utf-8")).hexdigest()
    return {"fingerprint": full, "fp8": full[:8], "config_hash": cfg_h, "prompts_hash": prompts_h,
            "git_commit": commit, "git_dirty": dirty}


def git_info(repo_dir: Path | str = Path(__file__).resolve().parent) -> dict[str, str | bool]:
    """The running code's commit and dirty flag; empty when not a git checkout."""
    def _run(args: list[str]) -> str:
        return subprocess.check_output(args, cwd=str(repo_dir), stderr=subprocess.DEVNULL,
                                       text=True).strip()

    try:
        commit = _run(["git", "rev-parse", "HEAD"])
    except Exception:
        commit = ""
    try:
        dirty = bool(_run(["git", "status", "--porcelain"]))
    except Exception:
        dirty = False
    return {"commit": commit, "dirty": dirty}


def gather_model_ids(models: ModelsConfig) -> list[str]:
    """Every configured model id, de-duplicated."""
    ids = [models.default.model]
    ids.extend(o["model"] for o in models.roles.values() if o.get("model"))
    return list(dict.fromkeys(ids))


def write_run_json(run_dir: Path | str, data: dict[str, Any]) -> None:
    path = Path(run_dir) / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def read_run_json(run_dir: Path | str) -> dict[str, Any]:
    return json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))


def update_run_json(run_dir: Path | str, **fields: Any) -> dict[str, Any]:
    """Merge ``fields`` into ``run.json`` and rewrite it."""
    data = read_run_json(run_dir)
    data.update(fields)
    write_run_json(run_dir, data)
    return data


def record_provider_calls(run_dir: Path) -> None:
    """Stamp run.json with the calls each provider served."""
    try:
        update_run_json(run_dir, provider_calls=provider_counts(run_dir / "llm_calls.jsonl"))
    except Exception:  # noqa: BLE001
        pass


def refuse_if_all_failed(run_dir: Path, test_metrics: list[Any], n_test_failed: int) -> None:
    """Refuse to score a cell whose every test sample failed to generate."""
    if test_metrics and n_test_failed == len(test_metrics):
        raise RuntimeError(
            f"test: all {n_test_failed} samples ended in a generation failure, so this cell "
            f"measured the harness rather than the model. See the `generation_failed` events "
            f"in {run_dir}/events.jsonl."
        )


def _harness_stats(run_dir: Path) -> dict[str, Any]:
    """How often the harness, rather than the task, cut a session short."""
    kills: dict[str, int] = {}
    truncated_calls = truncated_sessions = fatal_truncations = 0
    path = run_dir / "events.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for line in lines:
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("event") != "session_cut_short":
            continue
        if reason := ev.get("stopped_by"):
            kills[reason] = kills.get(reason, 0) + 1
        if n := int(ev.get("truncated_calls") or 0):
            truncated_calls += n
            truncated_sessions += 1
            if ev.get("last_stop_reason") == "length" and not ev.get("last_had_tool_call"):
                fatal_truncations += 1
    return {"sessions_killed": dict(sorted(kills.items())), "truncated_calls": truncated_calls,
            "truncated_sessions": truncated_sessions, "fatal_truncations": fatal_truncations}


def write_summary(run_dir: Path, test_summary: dict[str, Any], test_metrics: list[Any],
                  n_failed: int) -> None:
    """Write ``metrics/summary.json``: the benchmark's scores, API totals and harness stats."""
    refuse_if_all_failed(run_dir, test_metrics, n_failed)
    api = aggregate_llm_calls(run_dir / "llm_calls.jsonl")
    citation = api["by_role"].get(CITATION_ROLE)
    for key, value in (citation or {}).items():
        api["totals"][key] -= value
    api["citation_cost"] = float((citation or {}).get("cost", 0.0))
    summary = {
        "test": test_summary,
        "documents": {"active": 0, "total": 0},
        "votes": {"likes": 0, "dislikes": 0, "helpful": 0, "harmful": 0, "neutral": 0},
        "api": api,
        "harness": _harness_stats(run_dir),
    }
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics" / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
