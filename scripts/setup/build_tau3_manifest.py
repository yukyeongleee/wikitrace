#!/usr/bin/env python3
"""Freeze the tau3 retail suite: train 38 / test 76 over all 114 tasks, each multi-task user on
both sides, the smoke slice (4 train, then 2 test of the same users, one with NL assertions) at the
head of each list. Reads the clone tau3.sh installs; `--check` compares with the file."""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "src" / "wikitrace" / "envs" / "tau3" / "data" / "tau3.manifest.json"
DEFAULT_ROOT = REPO / "data" / "tau3"
SOURCE = {"repo": "sierra-research/tau2-bench", "tag": "v1.0.1",
          "commit": "fc0055dc4e0a316c3f83133267fbd6faaa770992"}
DOMAIN, SEED, TRAIN_N = "retail", 0, 38
SMOKE_SEED, SMOKE_TRAIN_N, SMOKE_TEST_N = 0, 4, 2


def load(root: Path) -> tuple[list[dict], dict]:
    clone = root / "repo"
    head = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    if head != SOURCE["commit"]:
        raise SystemExit(f"{clone} is at {head}, not the pinned {SOURCE['commit']}")
    data = clone / "data" / "tau2" / "domains" / DOMAIN
    return (json.loads((data / "tasks.json").read_text(encoding="utf-8")),
            json.loads((data / "db.json").read_text(encoding="utf-8")))


def user_of(task: dict, db: dict) -> str:
    """The DB user a task is about: gold arguments first, then ids, emails, orders or a name (and zip) in the text."""
    # Gold lookups by email or name/zip may be deliberately wrong, so only ids are trusted there.
    users, orders = db["users"], db["orders"]
    for action in task["evaluation_criteria"].get("actions") or []:
        args = action.get("arguments") or {}
        if args.get("user_id") in users:
            return args["user_id"]
        if args.get("order_id") in orders:
            return orders[args["order_id"]]["user_id"]
    text = " ".join(str(v) for v in task["user_scenario"]["instructions"].values() if v)
    for uid in re.findall(r"\b[a-z]+_[a-z]+_\d{4}\b", text):
        if uid in users:
            return uid
    by_email = {u["email"]: uid for uid, u in users.items()}
    for email in re.findall(r"[\w.]+@example\.com", text):
        if email in by_email:
            return by_email[email]
    for order in re.findall(r"#?W\d{7}", text):
        if "#" + order.lstrip("#") in orders:
            return orders["#" + order.lstrip("#")]["user_id"]
    by_name: dict[tuple[str, str], set[str]] = defaultdict(set)
    for uid, u in users.items():
        by_name[(u["name"]["first_name"].lower(), u["name"]["last_name"].lower())].add(uid)
    zips = set(re.findall(r"\b\d{5}\b", text))
    for first, last in re.findall(r"\b([A-Z][a-z]+) ([A-Z][a-z]+)\b", text):
        match = by_name.get((first.lower(), last.lower()), set())
        if len(match) > 1:
            match = {uid for uid in match if users[uid]["address"]["zip"] in zips}
        if len(match) == 1:
            return next(iter(match))
    raise ValueError(f"task {task['id']}: no user found")


def split(tasks: list[dict], db: dict) -> tuple[list[str], list[str], dict[str, str]]:
    """The train and test ids, and each task's user."""
    groups: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        groups[user_of(task, db)].append(task["id"])
    rng = random.Random(SEED)
    train: list[str] = []
    singles: list[str] = []
    for uid in sorted(groups):
        ids = groups[uid]
        if len(ids) == 1:
            singles += ids
            continue
        # A third of each user's tasks go to train, at least one, so every such user is on both sides.
        train += rng.sample(ids, max(1, round(len(ids) / 3)))
    train += rng.sample(sorted(singles, key=int), TRAIN_N - len(train))
    train_set = set(train)
    test = [t["id"] for t in tasks if t["id"] not in train_set]
    return sorted(train_set, key=int), sorted(test, key=int), {
        tid: uid for uid, ids in groups.items() for tid in ids}


def smoke(train: list[str], test: list[str], user: dict[str, str], nl: set[str]) -> tuple[list[str], list[str]]:
    """4 train tasks of users that also have a test task, then 2 test tasks of those users."""
    # Drawn again from the same generator until one of the six has NL assertions, so the
    # smoke runs the judge.
    rng = random.Random(SMOKE_SEED)
    test_users = {user[t] for t in test}
    while True:
        picked = sorted(rng.sample([t for t in train if user[t] in test_users], SMOKE_TRAIN_N), key=int)
        users = {user[t] for t in picked}
        held = sorted(rng.sample([t for t in test if user[t] in users], SMOKE_TEST_N), key=int)
        if nl & {*picked, *held}:
            return picked, held


def build(root: Path) -> dict:
    tasks, db = load(root)
    train, test, user = split(tasks, db)
    nl = {t["id"] for t in tasks if t["evaluation_criteria"].get("nl_assertions")}
    smoke_train, smoke_test = smoke(train, test, user, nl)

    def rows(head: list[str], ids: list[str]) -> list[dict]:
        # The smoke slice first, so a head slice of each role is the smoke suite.
        return [{"task_id": i, "user_id": user[i]} for i in head + [i for i in ids if i not in head]]

    return {
        "source": SOURCE, "seed": SEED, "pool": len(tasks), "users": len(set(user.values())),
        "smoke": {"seed": SMOKE_SEED, "train": len(smoke_train), "test": len(smoke_test),
                  "nl_assertions": [t for t in smoke_train + smoke_test if t in nl]},
        "train": rows(smoke_train, train), "test": rows(smoke_test, test),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="the tau3 install (holds repo/)")
    ap.add_argument("--check", action="store_true", help="compare with the committed manifest")
    args = ap.parse_args()
    manifest = build(args.root)
    text = json.dumps(manifest, indent=1) + "\n"
    if args.check:
        if OUT.read_text(encoding="utf-8") != text:
            raise SystemExit(f"{OUT} differs from a fresh build")
        print(f"{OUT} is up to date")
        return
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}: train {len(manifest['train'])}, test {len(manifest['test'])}, "
          f"smoke {manifest['smoke']['train']}/{manifest['smoke']['test']}")


if __name__ == "__main__":
    main()
