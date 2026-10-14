"""Continue a killed run in its own directory, skipping the samples it already scored."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from wikitrace.envs import Metrics


@dataclass(frozen=True)
class ResumePlan:
    """The samples a killed run already scored, read back off its own records."""

    done: tuple[str, ...] = ()
    seeded: tuple[Metrics, ...] = ()
    n_failed: int = 0


def plan_resume(run_dir: Path, *, fp8: str, test_ids: list[str]) -> ResumePlan:
    """An empty plan for a fresh directory; the scored prefix of a killed run of the same cohort."""
    meta_path = run_dir / "run.json"
    if not meta_path.exists():
        return ResumePlan()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") == "completed" or (run_dir / "metrics" / "summary.json").exists():
        raise RuntimeError(f"{run_dir} is a completed run; delete it to run this trial again")
    if meta.get("fp8") and meta["fp8"] != fp8:
        raise RuntimeError(f"{run_dir} was started under cohort {meta['fp8']}, this run is {fp8}; "
                           f"delete it to run this trial again")
    samples = run_dir / "metrics" / "samples.jsonl"
    rows = [json.loads(x) for x in samples.read_text(encoding="utf-8").splitlines() if x.strip()] \
        if samples.exists() else []
    done = tuple(str(r.get("sample_id", "")) for r in rows)
    if done != tuple(test_ids[: len(done)]):
        raise RuntimeError(f"the samples recorded in {run_dir} are not a prefix of this test split; "
                           f"delete it to run this trial again")
    return ResumePlan(done=done, seeded=tuple(Metrics(**(r.get("metrics") or {})) for r in rows),
                      n_failed=sum(1 for r in rows if r.get("generation_failed")))
