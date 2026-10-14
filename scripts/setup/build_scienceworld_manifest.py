#!/usr/bin/env python3
"""Freeze the ScienceWorld suite to exact (task, variation) pairs, committed beside the module.
The split is a COMPLETELY FIXED SET, so the pairs are chosen here (offline, seeded) and the
loader just reads them. Needs the simulator: run under data/scienceworld/.venv/bin/python."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import OrderedDict
from pathlib import Path

_DESCRIPTION = """Freeze the ScienceWorld suite to exact (task, variation) pairs.

The manifest is a COMPLETELY FIXED SET, committed to
``src/wikitrace/envs/textgame/data/scienceworld.manifest.json``; the loader just reads it.

A candidate variation is selectable when its gold path fits the 100-step limit, and a test
candidate also when its task description occurs in no train pick. The counts follow from
those pools: a small pool is taken whole and the remainder of the total is split as evenly
as possible over the larger types, the earlier type in TASK_TYPES taking the odd one.

  train  50 variations from the official `train` split over the 10 task types, 5 each
         unless a pool holds fewer: within a type, one variation per distinct task
         description before a second of any, in seeded order.
  test   100 variations from the official `test` split, the Matter types and chemistry-mix
         whole and the rest over the Measurement and conductivity types.

Both lists interleave the task types round-robin, so a head slice spans them. The
generator asserts the train/test description disjointness, records why every task type
outside the suite is excluded, and records the gold-path length of every candidate
variation (a number only; the gold path itself is withheld everywhere). Everything runs
under `simplification="easy"`, the reference harness's default.

Needs the simulator, so run it under the benchmark's venv (Java is taken from
`<root>/jdk`, where scripts/setup/scienceworld.sh installs it):

  data/scienceworld/.venv/bin/python scripts/setup/build_scienceworld_manifest.py
  data/scienceworld/.venv/bin/python scripts/setup/build_scienceworld_manifest.py --check
"""

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "src" / "wikitrace" / "envs" / "textgame" / "data" / "scienceworld.manifest.json"
SIMPLIFICATION = "easy"
ENV_STEP_LIMIT = 100
# Where the simulator's task description starts after its "Task Description:\n" header.
_TASK_PREFIX_LEN = 18

TASK_TYPES = (
    "boil", "change-the-state-of-matter-of", "freeze", "melt",
    "measure-melting-point-known-substance", "measure-melting-point-unknown-substance",
    "use-thermometer", "test-conductivity", "test-conductivity-of-unknown-substances",
    "chemistry-mix",
)
# The types whose test pool is taken whole; the larger types absorb the remainder.
WHOLE_TEST_POOL_TYPES = ("boil", "change-the-state-of-matter-of", "freeze", "melt",
                         "chemistry-mix")
TRAIN_TOTAL = 50
TRAIN_PER_TYPE = 5
TEST_TOTAL = 100

# Why a task type is not in the suite. Every task type outside TASK_TYPES matches a prefix
# here, which build() asserts, so the fallback reason names no type.
EXCLUSION_REASONS = {
    "find-": "classification: only the answer box colour varies between variations",
    "lifespan-": "one task description per type",
    "identify-life-stages-": "too few variations",
    "chemistry-mix-paint-": "3-6 distinct descriptions per type",
    "mendelian-genetics-": "gold paths over the 100-step limit on the variations measured",
    "inclined-plane-friction-unnamed-surfaces": "gold paths over the 100-step limit on the variations measured",
    "inclined-plane-determine-angle": "train and test task descriptions largely overlap",
    "inclined-plane-friction-named-surfaces": "train and test task descriptions largely overlap",
    "power-component": "ten variations in each split, too few to split further",
    "grow-": "a later addition to the benchmark",
}

_FALLBACK_REASON = "outside the selected task types"


def exclusion_reason(task: str) -> str:
    for prefix, reason in EXCLUSION_REASONS.items():
        if task.startswith(prefix):
            return reason
    return _FALLBACK_REASON


def describe(env, task: str, variation: int) -> str:
    env.load(task, variation, SIMPLIFICATION)
    return env.taskdescription()[_TASK_PREFIX_LEN:]


def gold_length(env, task: str, variation: int) -> int:
    env.load(task, variation, SIMPLIFICATION, generateGoldPath=True)
    return len(env.get_gold_action_sequence())


def split_evenly(total: int, tasks: list[str]) -> dict[str, int]:
    """``total`` over ``tasks`` as evenly as possible, the earlier tasks taking the odd ones."""
    base, extra = divmod(total, len(tasks))
    return {t: base + (1 if i < extra else 0) for i, t in enumerate(tasks)}


def train_counts(sizes: dict[str, int]) -> dict[str, int]:
    """``TRAIN_PER_TYPE`` each, a short pool taken whole with its shortfall spread over the rest."""
    counts = {t: min(TRAIN_PER_TYPE, sizes[t]) for t in TASK_TYPES}
    rest = [t for t in TASK_TYPES if counts[t] == TRAIN_PER_TYPE]
    for task, extra in split_evenly(TRAIN_TOTAL - sum(counts.values()), rest).items():
        counts[task] += extra
    return counts


def test_counts(sizes: dict[str, int]) -> dict[str, int]:
    """The small pools whole, the remainder of ``TEST_TOTAL`` split over the larger types."""
    counts = {t: sizes[t] for t in WHOLE_TEST_POOL_TYPES}
    rest = [t for t in TASK_TYPES if t not in counts]
    counts.update(split_evenly(TEST_TOTAL - sum(counts.values()), rest))
    return counts


def pick(pool: list[tuple[int, str]], want: int, rng: random.Random) -> list[tuple[int, str]]:
    """``want`` of ``pool``'s (variation, description): one per description before a second."""
    by_desc: OrderedDict[str, list[int]] = OrderedDict()
    for variation, desc in pool:
        by_desc.setdefault(desc, []).append(variation)
    descs = list(by_desc)
    rng.shuffle(descs)
    for variations in by_desc.values():
        rng.shuffle(variations)
    picks: list[tuple[int, str]] = []
    while len(picks) < want and any(by_desc.values()):
        for desc in descs:
            if by_desc[desc] and len(picks) < want:
                picks.append((by_desc[desc].pop(0), desc))
    if len(picks) < want:
        raise ValueError(f"only {len(picks)} variations for {want}")
    return picks


def interleave(groups: OrderedDict[str, list[dict]]) -> list[dict]:
    out: list[dict] = []
    queues = [list(g) for g in groups.values()]
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def build(root: Path, seed: int) -> dict:
    os.environ["PATH"] = f"{root / 'jdk' / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
    from scienceworld import ScienceWorldEnv

    env = ScienceWorldEnv("", envStepLimit=ENV_STEP_LIMIT)
    task_names = env.get_task_names()
    rng = random.Random(seed)
    gold: dict[str, dict[str, int]] = {}
    indexes: dict[str, int] = {}
    raw: dict[str, dict[str, list[int]]] = {}
    train_pool: dict[str, list[tuple[int, str]]] = {}
    for task in TASK_TYPES:
        indexes[task] = task_names.index(task)
        env.load(task, 0, SIMPLIFICATION)
        raw[task] = {"train": list(env.get_variations_train()),
                     "test": list(env.get_variations_test())}
        gold[task] = {str(v): gold_length(env, task, v)
                      for v in raw[task]["train"] + raw[task]["test"]}
        train_pool[task] = [(v, describe(env, task, v)) for v in raw[task]["train"]
                            if gold[task][str(v)] <= ENV_STEP_LIMIT]

    n_train = train_counts({t: len(p) for t, p in train_pool.items()})
    train: OrderedDict[str, list[dict]] = OrderedDict()
    test_pool: dict[str, list[tuple[int, str]]] = {}
    for task in TASK_TYPES:
        picks = pick(train_pool[task], n_train[task], rng)
        descs = {d for _, d in picks}
        train[task] = picks
        test_pool[task] = [(v, describe(env, task, v)) for v in raw[task]["test"]
                           if gold[task][str(v)] <= ENV_STEP_LIMIT]
        test_pool[task] = [(v, d) for v, d in test_pool[task] if d not in descs]

    n_test = test_counts({t: len(p) for t, p in test_pool.items()})
    test: OrderedDict[str, list[dict]] = OrderedDict()
    pools: dict[str, dict[str, int]] = {}
    for task in TASK_TYPES:
        test[task] = pick(test_pool[task], n_test[task], rng)
        removed = sum(1 for n in gold[task].values() if n > ENV_STEP_LIMIT)
        pools[task] = {"train": len(raw[task]["train"]), "test": len(raw[task]["test"]),
                       "gold_over_limit": removed, "train_selectable": len(train_pool[task]),
                       "test_selectable": len(test_pool[task])}
        assert {d for _, d in train[task]}.isdisjoint(d for _, d in test[task]), task
        print(f"{task}: train {n_train[task]} of {pools[task]['train_selectable']}, "
              f"test {n_test[task]} of {pools[task]['test_selectable']}, "
              f"gold over limit {removed}")
    env.close()

    def entries(picks: list[tuple[int, str]], task: str, split: str) -> list[dict]:
        return [{"task": task, "task_index": indexes[task], "variation": v, "split": split,
                 "description": d, "gold_length": gold[task][str(v)]} for v, d in picks]

    for task in TASK_TYPES:
        train[task] = entries(train[task], task, "train")
        test[task] = entries(test[task], task, "test")
    excluded = {t: exclusion_reason(t) for t in task_names if t not in TASK_TYPES}
    assert _FALLBACK_REASON not in excluded.values(), excluded
    over_limit = sorted(
        f"{t}/var{v}" for t, g in gold.items() for v, n in g.items() if n > ENV_STEP_LIMIT
    )
    return {
        "_meta": {
            "generator": "scripts/experiments/setup/build_scienceworld_manifest.py",
            "seed": seed,
            "simplification": SIMPLIFICATION,
            "env_step_limit": ENV_STEP_LIMIT,
            "sizes": {"train": sum(n_train.values()), "test": sum(n_test.values())},
            "counts": {"train": n_train, "test": n_test},
            "pools": pools,
            "train_test_descriptions_disjoint": True,
            "excluded": excluded,
            "excluded_gold_over_limit": over_limit,
            "order": "task types interleaved round-robin",
        },
        "train": interleave(train),
        "test": interleave(test),
        "candidate_gold_lengths": gold,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=_DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO / "data" / "scienceworld")
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
    print(f"wrote {args.out}: train {meta['sizes']['train']}, test {meta['sizes']['test']}; "
          f"candidates with gold over {ENV_STEP_LIMIT}: {len(meta['excluded_gold_over_limit'])}")


if __name__ == "__main__":
    main()
