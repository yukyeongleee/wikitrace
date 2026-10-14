#!/usr/bin/env python3
"""Copy the tau3 retail instruction, policy and tool schemas out of the pinned clone and render the
solver prompt's `<tool>` blocks and signature list from them. Runs under data/tau3/.venv/bin/python
(scripts/setup/tau3.sh installs it); `--check` compares with the committed file."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "src" / "wikitrace" / "envs" / "tau3" / "data" / "tau3_tools.json"
DEFAULT_ROOT = REPO / "data" / "tau3"
SOURCE = {"repo": "sierra-research/tau2-bench", "tag": "v1.0.1",
          "commit": "fc0055dc4e0a316c3f83133267fbd6faaa770992"}
DOMAIN = "retail"


def flat(text: str) -> str:
    """A description's wrapped lines joined, its `- ` bullets kept one per line."""
    text = re.sub(r"\s*\n\s*(?=-\s)", "\n", text.strip())
    text = re.sub(r"\s*\n\s*(?!-\s)", " ", text)
    return re.sub(r"(?m)^-\s+", "- ", text)


def type_name(schema: dict) -> str:
    kind = schema.get("type", "value")
    if kind == "array":
        return f"array of {schema.get('items', {}).get('type', 'value')}"
    return kind


def placeholder(schema: dict):
    if schema.get("type") == "array":
        return [f"<{schema.get('items', {}).get('type', 'value')}>"]
    return f"<{schema.get('type', 'value')}>"


def tool_block(fn: dict) -> str:
    """One `<tool>` block: the description, one line per parameter, the `take_action` call shape."""
    props = fn["parameters"].get("properties", {})
    required = set(fn["parameters"].get("required", []))
    lines = [
        f"- {k} ({type_name(v)}{', required' if k in required else ', optional'}): {flat(v.get('description', ''))}"
        for k, v in props.items()
    ]
    params = "parameters:\n" + "\n".join(lines) if lines else "parameters: none"
    call = json.dumps({"action": fn["name"], "argument": {k: placeholder(v) for k, v in props.items()}})
    return f'<tool name="{fn["name"]}">\n{flat(fn["description"])}\n{params}\ncall: {call}\n</tool>'


def signature(fn: dict) -> str:
    return f"- {fn['name']}({', '.join(fn['parameters'].get('properties', {}))})"


def joined_instruction(text: str) -> str:
    """The authors' instruction with its wrapped lines joined: the sentence, the two bullets, the rest."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    return "\n".join([lines[0] + " " + lines[1], *lines[2:4], " ".join(lines[4:])])


def build(root: Path) -> dict:
    head = subprocess.run(["git", "-C", str(root / "repo"), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    if head != SOURCE["commit"]:
        raise SystemExit(f"{root / 'repo'} is at {head}, not the pinned {SOURCE['commit']}")
    from tau2.agent.llm_agent import AGENT_INSTRUCTION
    from tau2.registry import registry

    env = registry.get_env_constructor(DOMAIN)()
    functions = [t.openai_schema["function"] for t in env.get_tools()]
    return {
        "source": SOURCE,
        "domain": DOMAIN,
        "instruction": AGENT_INSTRUCTION,
        "instruction_joined": joined_instruction(AGENT_INSTRUCTION),
        "policy": env.get_policy().strip(),
        "tools": functions,
        "tool_blocks": [tool_block(fn) for fn in functions],
        "signatures": [signature(fn) for fn in functions],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="the tau3 install (clone + venv)")
    ap.add_argument("--check", action="store_true", help="compare with the committed file instead of writing")
    args = ap.parse_args()
    text = json.dumps(build(args.root), ensure_ascii=False, indent=1) + "\n"
    if args.check:
        same = OUT.exists() and OUT.read_text(encoding="utf-8") == text
        print(f"{OUT}: {'identical' if same else 'DIFFERS'}")
        return 0 if same else 1
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
