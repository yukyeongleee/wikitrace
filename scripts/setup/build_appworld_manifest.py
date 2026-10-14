#!/usr/bin/env python3
"""Freeze the AppWorld suites to exact task ids, committed beside the benchmark module.
The split must be a COMPLETELY FIXED SET, so the ids are chosen here (offline, seeded) and
the loader just reads them. Needs only the installed data, not the simulator."""
# Check without writing: uv run python scripts/setup/build_appworld_manifest.py --check

from __future__ import annotations

import argparse
import json
import random
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "appworld" / "data"
OUT = REPO / "src" / "wikitrace" / "envs" / "appworld" / "data" / "appworld_splits.json"
SEED = 0
# train: every difficulty-2 and -3 scenario of the official `train` split, whole (54 tasks),
# shared by both suites. seen_test: the same difficulties of `test_normal` (111 tasks).
TRAIN_DIFFICULTIES = (2, 3)
SEEN_DIFFICULTIES = (2, 3)
# unseen_test: a seeded 33 of the 65 difficulty-3 scenarios of `test_challenge` (99 tasks);
# the scenarios left out are recorded under `unseen_excluded`.
UNSEEN_DIFFICULTY = 3
UNSEEN_SCENARIOS = 33


def scenarios(split: str) -> "OrderedDict[str, list[str]]":
    """{scenario_id: [task_id, ...]} in the official file order, tasks sorted by number."""
    by: OrderedDict[str, list[str]] = OrderedDict()
    for task_id in (DATA / "datasets" / f"{split}.txt").read_text().split():
        by.setdefault(task_id.rsplit("_", 1)[0], []).append(task_id)
    for tasks in by.values():
        tasks.sort(key=lambda t: int(t.rsplit("_", 1)[1]))
    return by


def difficulty(task_id: str) -> int:
    meta = json.loads((DATA / "tasks" / task_id / "ground_truth" / "metadata.json").read_text())
    return int(meta["difficulty"])


def by_difficulty(split: str) -> dict[int, list[tuple[str, list[str]]]]:
    out: dict[int, list[tuple[str, list[str]]]] = {}
    for scenario, tasks in scenarios(split).items():
        levels = {difficulty(t) for t in tasks}
        assert len(levels) == 1, f"{scenario}: tasks disagree on difficulty {levels}"
        out.setdefault(levels.pop(), []).append((scenario, tasks))
    return out


def interleave(groups: list[list[tuple[str, list[str]]]]) -> list[tuple[str, list[str]]]:
    """Round-robin across difficulties until every group is drained."""
    # So a head slice spans both difficulties; a scenario's three tasks stay adjacent, so
    # the slice cuts on scenario boundaries and SGC is defined on it.
    out: list[tuple[str, list[str]]] = []
    queues = [list(g) for g in groups]
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def entries(picked: list[tuple[str, list[str]]]) -> list[dict]:
    return [
        {"id": t, "scenario_id": s, "difficulty": difficulty(t)}
        for s, tasks in picked for t in tasks
    ]


def build() -> dict:
    rng = random.Random(SEED)

    def shuffled(split: str, levels: tuple[int, ...]) -> list[list[tuple[str, list[str]]]]:
        groups = by_difficulty(split)
        out = []
        for level in levels:
            g = list(groups.get(level, []))
            rng.shuffle(g)
            out.append(g)
        return out

    train = interleave(shuffled("train", TRAIN_DIFFICULTIES))
    seen = interleave(shuffled("test_normal", SEEN_DIFFICULTIES))
    challenge = shuffled("test_challenge", (UNSEEN_DIFFICULTY,))[0]
    unseen, excluded = challenge[:UNSEEN_SCENARIOS], challenge[UNSEEN_SCENARIOS:]
    return {
        "seed": SEED,
        "data_version": (DATA / "version.txt").read_text().strip(),
        "train": entries(train),
        "seen_test": entries(seen),
        "unseen_test": entries(unseen),
        "unseen_excluded": sorted(s for s, _ in excluded),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="build and compare with the committed file")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    manifest = build()
    text = json.dumps(manifest, indent=1) + "\n"
    counts = {k: len(manifest[k]) for k in ("train", "seen_test", "unseen_test", "unseen_excluded")}
    if args.check:
        same = args.out.exists() and args.out.read_text() == text
        print(f"{counts}; committed file {'matches' if same else 'DIFFERS or is missing'}")
        raise SystemExit(0 if same else 1)
    args.out.write_text(text)
    print(f"wrote {args.out}: {counts}")


if __name__ == "__main__":
    main()
