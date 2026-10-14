#!/usr/bin/env python3
"""Freeze the ALFWorld suites to exact game files, committed beside the benchmark module.
The split is a COMPLETELY FIXED SET, so the games are chosen here (offline, seeded) and the
loader just reads them. Standard library only: the task text is read off the game grammar."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import OrderedDict
from pathlib import Path

_DESCRIPTION = """Freeze the ALFWorld suites to exact game files.

The manifest is a COMPLETELY FIXED SET, committed to
``src/wikitrace/envs/textgame/data/alfworld.manifest.json``; the loader just reads it.

  train        50 games from `train`, the six task types balanced (9/9/8/8/8/8), drawn
               with a seeded shuffle that prefers, per type, a scene occurring in
               `valid_seen` and a scene not yet in the train set.
  valid_seen   the games of `valid_seen` set in a scene one of the 50 train games visits
               (101 of 140), so "seen" is seen relative to our train set.
  valid_unseen every game of `valid_unseen` (134): four rooms absent from `train`.

Every list interleaves the task types round-robin, so a head slice spans them. Games are
collected by the same rules as `AlfredTWEnv.collect_game_files` (six types, solvable, no
movable-receptacle or sliced variants). Standard library only: the task text is read off
the game's own grammar, so no simulator is needed.

  uv run python scripts/setup/build_alfworld_manifest.py            # write
  uv run python scripts/setup/build_alfworld_manifest.py --check    # compare
"""

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "src" / "wikitrace" / "envs" / "textgame" / "data" / "alfworld.manifest.json"
GAMES = "json_2.1.1"
# `AlfredTWEnv.TASK_TYPES` in its numbering, which is the round-robin order here.
TASK_TYPES = (
    "pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place",
)
TRAIN_SIZE = 50
_TASK_RE = re.compile(r"Your task is to: (.*?)\"")


def collect(root: Path, split: str) -> list[dict]:
    """Every solvable game of ``split`` as a manifest entry, in path order."""
    out = []
    for game in sorted((root / GAMES / split).glob("*/*/game.tw-pddl")):
        rel = game.parent.relative_to(root).as_posix()
        if "movable" in rel or "Sliced" in rel:
            continue
        traj = json.loads((game.parent / "traj_data.json").read_text())
        if traj["task_type"] not in TASK_TYPES:
            continue
        data = json.loads(game.read_text())
        if not data.get("solvable"):
            continue
        m = _TASK_RE.search(data["grammar"])
        if m is None:
            raise ValueError(f"{game}: no task line in the grammar")
        out.append({
            "game": rel, "split": split, "task_type": traj["task_type"],
            "scene_id": int(traj["scene"]["scene_num"]),
            "task_text": f"Your task is to: {m.group(1)}",
        })
    return out


def by_type(entries: list[dict]) -> OrderedDict[str, list[dict]]:
    groups: OrderedDict[str, list[dict]] = OrderedDict((t, []) for t in TASK_TYPES)
    for e in entries:
        groups[e["task_type"]].append(e)
    return groups


def interleave(groups: OrderedDict[str, list[dict]]) -> list[dict]:
    """Round-robin across the task types, so any head slice spans them."""
    out: list[dict] = []
    queues = [list(g) for g in groups.values()]
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def pick_train(train: list[dict], seen_scenes: set[int], seed: int) -> OrderedDict[str, list[dict]]:
    """50 games over the six types, preferring `valid_seen` scenes and distinct scenes."""
    rng = random.Random(seed)
    per_type = [TRAIN_SIZE // len(TASK_TYPES) + (1 if i < TRAIN_SIZE % len(TASK_TYPES) else 0)
                for i in range(len(TASK_TYPES))]
    used_scenes: set[int] = set()
    used_texts: set[str] = set()
    chosen: OrderedDict[str, list[dict]] = OrderedDict()
    for task_type, want in zip(TASK_TYPES, per_type):
        pool = [e for e in train if e["task_type"] == task_type]
        rng.shuffle(pool)
        picks: list[dict] = []
        while len(picks) < want and pool:
            # The ordering is re-evaluated after every pick, since it depends on which
            # task descriptions and scenes the train set already holds.
            best = min(pool, key=lambda e: (
                e["task_text"] in used_texts,
                e["scene_id"] not in seen_scenes, e["scene_id"] in used_scenes,
            ))
            pool.remove(best)
            picks.append(best)
            used_scenes.add(best["scene_id"])
            used_texts.add(best["task_text"])
        if len(picks) < want:
            raise ValueError(f"{task_type}: only {len(picks)} train games for {want}")
        chosen[task_type] = picks
    return chosen


def build(root: Path, seed: int) -> dict:
    splits = {s: collect(root, s) for s in ("train", "valid_seen", "valid_unseen")}
    seen_scenes = {e["scene_id"] for e in splits["valid_seen"]}
    train = pick_train(splits["train"], seen_scenes, seed)
    train_scenes = {e["scene_id"] for g in train.values() for e in g}
    kept = [e for e in splits["valid_seen"] if e["scene_id"] in train_scenes]
    kept_scenes = {e["scene_id"] for e in kept}
    manifest = {
        "_meta": {
            "generator": "scripts/experiments/setup/build_alfworld_manifest.py",
            "seed": seed,
            "train_split": "train",
            "sizes": {"train": TRAIN_SIZE, "valid_seen": len(kept),
                      "valid_unseen": len(splits["valid_unseen"])},
            "pool_sizes": {s: len(v) for s, v in splits.items()},
            "train_scenes": sorted(train_scenes),
            "valid_seen_dropped": len(splits["valid_seen"]) - len(kept),
            "valid_seen_scenes_covered": len(kept_scenes & train_scenes),
            "valid_seen_scenes": len(kept_scenes),
            "valid_seen_split_scenes": len(seen_scenes),
            "order": "task types interleaved round-robin; tests in path order within a type",
        },
        "train": interleave(train),
        "valid_seen": interleave(by_type(kept)),
        "valid_unseen": interleave(by_type(splits["valid_unseen"])),
    }
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=_DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO / "data" / "alfworld")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", action="store_true", help="compare with the committed manifest, write nothing")
    args = ap.parse_args()
    manifest = build(args.root.resolve(), args.seed)
    text = json.dumps(manifest, indent=1, ensure_ascii=False) + "\n"
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        print("unchanged" if current == text else "DIFFERS from", args.out)
        raise SystemExit(0 if current == text else 1)
    args.out.write_text(text, encoding="utf-8")
    meta = manifest["_meta"]
    print(f"wrote {args.out}: train {meta['sizes']['train']}, valid_seen {meta['sizes']['valid_seen']} "
          f"({meta['valid_seen_dropped']} dropped), valid_unseen {meta['sizes']['valid_unseen']}; "
          f"train covers {meta['valid_seen_scenes_covered']}/{meta['valid_seen_scenes']} retained "
          f"scenes of the split's {meta['valid_seen_split_scenes']}")


if __name__ == "__main__":
    main()
