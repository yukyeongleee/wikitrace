#!/usr/bin/env python3
"""Freeze the Plancraft suites (train group, test-seen, test-unseen) to exact example ids.
Needs the simulator and the authors' OracleModel: run under data/plancraft/.venv/bin/python,
which scripts/setup/plancraft.sh installs. `--check` compares with the committed file."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import plancraft

from plancraft.config import EvalConfig
from plancraft.environment import ImpossibleActionHandler, MoveActionHandler, SmeltActionHandler
from plancraft.evaluator import Evaluator
from plancraft.models import get_model

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "src" / "wikitrace" / "envs" / "textgame" / "data" / "plancraft.manifest.json"
SEED = 0
STEP_LIMIT = 30  # the suite's server-side act budget; an example whose oracle needs more is unsolvable
BINS = ("easy", "medium", "hard")
HARDER = ("medium", "hard")
TRAIN_SIZE = 50
TEST_SIZE = 100


def oracle_steps(split: str) -> dict[str, tuple[bool, int]]:
    """Run the authors' OracleModel on every solvable example of a split: (solved, steps)."""
    actions = [MoveActionHandler(), SmeltActionHandler(), ImpossibleActionHandler()]
    cfg = EvalConfig(plancraft=dict(model="oracle", mode="oracle", split=split, max_steps=200,
                                    max_message_window=40, few_shot=False, use_text_inventory=True,
                                    use_images=False, valid_actions=["move", "smelt", "impossible"],
                                    output_dir="/tmp/plancraft-oracle"),
                     wandb=dict(project="p", entity="n", mode="disabled"))
    ev = Evaluator(run_name="oracle", actions=actions, output_dir="/tmp/plancraft-oracle", split=split,
                   max_steps=200, resume=False)
    model = get_model(cfg)
    out = {}
    for e in ev.examples:
        if e.impossible:
            continue
        model.reset()
        r = ev.eval_example(e, model)
        out[e.id] = (bool(r["success"]), int(r["number_of_steps"]))
    return out


def load(split: str) -> list[dict]:
    rows = json.loads((Path(plancraft.__file__).parent / "data" / f"{split}.json").read_text())
    return [r for r in rows if r["impossible"] in (False, "False")]


def proportional(rows: list[dict]) -> list[dict]:
    """Spread each bin evenly over the list, so every head slice mirrors the list's bin mix."""
    rows = sorted(rows, key=lambda r: r["id"])
    size = Counter(r["complexity_split"] for r in rows)
    pos: Counter = Counter()
    keyed = []
    for r in rows:
        b = r["complexity_split"]
        keyed.append(((pos[b] + 0.5) / size[b], BINS.index(b), r["id"], r))
        pos[b] += 1
    return [k[-1] for k in sorted(keyed, key=lambda k: k[:3])]


def harder_first(pool: list[dict], rng: random.Random) -> list[dict]:
    """TEST_SIZE examples: every medium and hard one (sampled down if over), then easy to fill."""
    by_id = sorted(pool, key=lambda r: r["id"])
    harder = [r for r in by_id if r["complexity_split"] in HARDER]
    easy = [r for r in by_id if r["complexity_split"] == "easy"]
    if len(harder) >= TEST_SIZE:
        return rng.sample(harder, TEST_SIZE)
    if len(harder) + len(easy) < TEST_SIZE:
        raise SystemExit(f"pool holds {len(harder)} medium/hard + {len(easy)} easy < {TEST_SIZE}")
    return harder + rng.sample(easy, TEST_SIZE - len(harder))


def entry(r: dict, steps: dict) -> dict:
    return {"id": r["id"], "target": r["target"], "complexity_split": r["complexity_split"],
            "optimal_path_length": int(float(r["optimal_path_length"])), "oracle_steps": steps[r["id"]][1]}


def build() -> dict:
    train, test = load("train"), load("test")
    steps = {**oracle_steps("train"), **oracle_steps("test")}

    def fits(r: dict) -> bool:
        return steps[r["id"]][0] and steps[r["id"]][1] <= STEP_LIMIT

    over = {s: sum(not fits(r) for r in rows) for s, rows in (("train", train), ("test", test))}
    train, test = [r for r in train if fits(r)], [r for r in test if fits(r)]

    fitting = defaultdict(lambda: defaultdict(list))
    for r in train:
        fitting[r["target"]][r["complexity_split"]].append(r)
    harder_n = Counter(r["target"] for r in test if r["complexity_split"] in HARDER)
    easy_n = Counter(r["target"] for r in test if r["complexity_split"] == "easy")
    ranked = sorted((t for t in fitting if harder_n[t] or easy_n[t]), key=lambda t: (-harder_n[t], -easy_n[t], t))
    group = []
    for t in ranked[:TRAIN_SIZE]:
        cand = fitting[t]["medium"] + fitting[t]["hard"] or fitting[t]["easy"]
        group.append(random.Random(t).choice(sorted(cand, key=lambda r: r["id"])))

    group_targets = {r["target"] for r in group}
    train_targets = {r["target"] for r in load("train")}
    pools = {"seen": [r for r in test if r["target"] in group_targets],
             "unseen": [r for r in test if r["target"] not in train_targets]}
    suites = {name: proportional(harder_first(pool, random.Random(SEED))) for name, pool in pools.items()}

    def bins(rows: list[dict]) -> dict[str, int]:
        c = Counter(r["complexity_split"] for r in rows)
        return {b: c[b] for b in BINS}

    manifest = {
        "_meta": {
            "generator": "scripts/experiments/setup/build_plancraft_manifest.py",
            "source": "gautierdag/plancraft@7617e9ff plancraft/data/{train,test}.json, impossible examples excluded",
            "seed": SEED, "step_limit": STEP_LIMIT,
            "oracle_over_limit_or_unsolved": over,
            "train_group": {
                "size": TRAIN_SIZE, "bins": bins(group),
                "rule": "targets with a train example within step_limit that occur in test, ranked by "
                        "(-test medium/hard examples, -test easy examples, target); first size taken; one "
                        "train example each, Random(target).choice over its medium/hard examples sorted by "
                        "id, else over its easy ones",
            },
            "test": {
                "size": TEST_SIZE,
                "rule": "every medium/hard example of the pool (Random(seed).sample down to size if over), "
                        "then easy examples sampled with the same rng from the pool's easy ones sorted by id",
                "seen": {"pool": bins(pools["seen"]), "picked": bins(suites["seen"])},
                "unseen": {"pool": bins(pools["unseen"]), "picked": bins(suites["unseen"])},
            },
            "seen": "test example whose target is a train_group target",
            "unseen": "test example whose target is in no solvable train example",
            "order": "each list sorted by ((position within its bin + 0.5) / bin size, bin, id), so a head "
                     "slice mirrors the list's bin mix",
        },
        "train": [entry(r, steps) for r in proportional(group)],
        "test_seen": [entry(r, steps) for r in suites["seen"]],
        "test_unseen": [entry(r, steps) for r in suites["unseen"]],
    }
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", action="store_true", help="compare with the committed manifest, write nothing")
    args = ap.parse_args()
    manifest = build()
    text = json.dumps(manifest, indent=1)
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        raise SystemExit(0 if current == text else f"{args.out} differs from a fresh build")
    args.out.write_text(text, encoding="utf-8")
    print(json.dumps(manifest["_meta"], indent=1))


if __name__ == "__main__":
    main()
