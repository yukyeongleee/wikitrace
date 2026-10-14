#!/usr/bin/env python3
"""Record the installed AppWorld dataset's sha256 per file in <install>/integrity.json.
Prints the aggregate, which `DATA_DIGEST` in the AppWorld benchmark module pins; an AppWorld
run refuses any other install and names the files that differ from this record."""
# uv run python scripts/setup/appworld_integrity.py [--install data/appworld]

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wikitrace.envs.appworld.benchmarks.appworld import (
    DATA_DIGEST, DEFAULT_ROOT, INTEGRITY_FILE, aggregate_digest, dataset_digests,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--install", type=Path, default=Path(DEFAULT_ROOT))
    args = ap.parse_args()
    out = args.install / INTEGRITY_FILE
    # The record is what a mismatch is diffed against, so it is never rewritten in place.
    if out.exists():
        raise SystemExit(f"{out} exists; move it aside to record the dataset again")
    digests = dataset_digests(args.install)
    aggregate = aggregate_digest(digests)
    out.write_text(json.dumps({"aggregate": aggregate, "files": digests}, indent=1) + "\n",
                   encoding="utf-8")
    print(f"{len(digests)} files, aggregate {aggregate} -> {out}")
    if aggregate != DATA_DIGEST:
        print(f"differs from the pinned DATA_DIGEST {DATA_DIGEST}")


if __name__ == "__main__":
    main()
