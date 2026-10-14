#!/usr/bin/env python3
"""Build the Plancraft solver's intro and few-shot transcript from the authors' prompt builders.
Run under data/plancraft/.venv/bin/python. The handlers are the environment's own (move, smelt,
seeded search, impossible), so the few-shot's search replies equal what the environment returns."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEXTGAME = REPO / "src" / "wikitrace" / "envs" / "textgame"
OUT = TEXTGAME / "data" / "plancraft_react_prompts.json"


def load_env_module():
    """``envs/plancraft.py`` by path: it imports nothing from the package."""
    spec = importlib.util.spec_from_file_location("plancraft_env", TEXTGAME / "envs" / "plancraft.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transcript(dialogue: list[dict]) -> str:
    """The authors' dialogue as ``> command`` lines between observations, as the ALFWorld file holds it."""
    out = ""
    for message in dialogue:
        text = message["content"] if message["role"] == "user" else f"> {message['content']}"
        out += text if text.endswith("\n") else text + "\n"
    return out


def build() -> dict[str, str]:
    from plancraft.environment import prompts
    from plancraft.environment.search import gold_search_recipe

    env = load_env_module()
    handlers = env.handlers()

    def seeded_search(recipe_name: str) -> str:
        random.seed(env.SEARCH_SEED)
        return gold_search_recipe(recipe_name)

    prompts.gold_search_recipe = seeded_search
    return {
        "system": prompts.get_system_prompt(handlers)["content"],
        "fewshot": transcript(prompts.get_prompt_example(handlers, use_text_inventory=True)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", action="store_true", help="compare with the committed file, write nothing")
    args = ap.parse_args()
    text = json.dumps(build(), indent=1, ensure_ascii=False) + "\n"
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        raise SystemExit(0 if current == text else f"{args.out} differs from a fresh build")
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
