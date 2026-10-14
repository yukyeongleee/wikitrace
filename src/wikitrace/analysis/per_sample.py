"""Export per-sample outcome rows from finished run directories."""
from __future__ import annotations

import csv
import json
from pathlib import Path

COLUMNS = ["benchmark", "suite", "method", "trial", "sample_id", "outcome", "value"]


def run_rows(run_dir: Path) -> list[dict]:
    """Rows of one completed run, or none when it has not completed."""
    meta = json.loads((run_dir / "run.json").read_text())
    if meta.get("status") != "completed" or not (run_dir / "metrics/summary.json").exists():
        return []
    rows = []
    for line in (run_dir / "metrics/samples.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        if s.get("generation_failed"):
            continue
        success = bool(s.get("metrics", {}).get("success"))
        value = float(s["score"]) if meta["benchmark"] == "scienceworld" else float(success)
        rows.append(dict(benchmark=meta["benchmark"], suite=meta["db_id"], method=meta["label"],
                         trial=int(meta["trial"]), sample_id=s["sample_id"], outcome=int(success), value=value))
    return rows


def export(run_dirs: list[str | Path], out_csv: str | Path) -> int:
    """Write every completed run found under run_dirs to out_csv and return the row count."""
    runs = sorted({p.parent for d in run_dirs for p in Path(d).rglob("run.json")})
    rows = [r for run in runs for r in run_rows(run)]
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)
