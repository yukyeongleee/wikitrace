"""The tau3 solver prompt: the record's sections in order, built from the generated tools data."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from wikitrace.envs import resolve
from wikitrace.envs.tau3.benchmarks.tau3 import PROMPT_DATA_PATH, read_prompt_data
from wikitrace.harness.prompts import render_system, render_user

REPO = Path(__file__).resolve().parents[2]
GENERATOR = REPO / "scripts" / "setup" / "build_tau3_prompt_data.py"
VENV_PYTHON = REPO / "data" / "tau3" / ".venv" / "bin" / "python"
HEADER = ("Every action goes through the `take_action` tool, which takes `action` and `argument`. "
          "`action` is the name of one tool below, and `argument` is that tool's parameters as one JSON object.")
NOTES = "Runtime notes:\n- Act on the store through the `take_action` tool."
# The design record's System prompt decision, verbatim.
RUNTIME_NOTES = """Runtime notes:
- Act on the store through the `take_action` tool. Its `action` is the name of one tool in
  <available_tools>, and its `argument` is what that tool's `call:` line shows. A call
  returns the tool's output or an `Error:` line. Send one action per turn and read its
  result before you send the next.
- The user's first message is in the task below. To talk to the user, reply with a message
  and no tool call. The user's answer comes back as the next message.
- The sandbox is offline and has no pip. You cannot install or download anything.
- The episode ends when the user ends the conversation, after ten rejected or failed tool
  calls, or when the step budget runs out. After that, `take_action` only answers
  `Episode over`."""


def _system(**memory) -> str:
    slots = dict(access="none", prompt_strategy="top-down", substrate="", keep_index=True, keep_links=True,
                 memory_template="", items="", layers="", strategy="", by_task_type=True, memory_documents="")
    slots.update(memory)
    return render_system("tau3_solver", prompt_data=read_prompt_data(), env_tool="take_action", **slots)


def test_the_tools_data_holds_the_sixteen_schemas_and_no_respond() -> None:
    data = read_prompt_data()
    assert data["source"]["commit"] == "fc0055dc4e0a316c3f83133267fbd6faaa770992"
    assert [fn["name"] for fn in data["tools"]] == [
        "calculate", "cancel_pending_order", "exchange_delivered_order_items", "find_user_id_by_name_zip",
        "find_user_id_by_email", "get_order_details", "get_product_details", "get_item_details",
        "get_user_details", "list_all_product_types", "modify_pending_order_address",
        "modify_pending_order_items", "modify_pending_order_payment", "modify_user_address",
        "return_delivered_order_items", "transfer_to_human_agents"]
    assert len(data["tool_blocks"]) == len(data["signatures"]) == 16
    assert not any("respond" in t for t in data["tool_blocks"] + data["signatures"])
    # The instruction and policy are the authors' text: the joined instruction reorders no word.
    assert data["instruction_joined"].split() == data["instruction"].split()
    assert data["policy"].startswith("# Retail agent policy") and "\n\n\n" not in data["policy"]


def test_the_system_prompt_has_the_records_sections_in_order_and_no_few_shot() -> None:
    text = _system()
    data = read_prompt_data()
    # Anchored on line starts: the instruction itself mentions `<policy>`.
    sections = ["<instructions>\n", "\n</instructions>\n", "\n<policy>\n", "\n</policy>\n",
                "\n<available_tools>\n", "\n</available_tools>\n", "\n" + NOTES]
    positions = [text.index(s) for s in sections]
    assert positions == sorted(positions)
    assert text.startswith(f"<instructions>\n{data['instruction_joined']}\n</instructions>\n\n<policy>\n{data['policy']}\n</policy>\n\n<available_tools>\n{HEADER}\n\n")
    assert text.count("<tool name=") == 16 and text.count("</tool>") == 16
    param_lines = re.findall(r"^- \w+ \(.*?\): ", text, flags=re.M)
    n_params = sum(len(fn["parameters"].get("properties", {})) for fn in data["tools"])
    assert len(param_lines) == n_params == 39 and all(", required): " in ln for ln in param_lines)
    assert text.count("\ncall: {") == 16
    assert text.count("Runtime notes:") == 1 and text.count("\n- ") >= 4
    # No example episode, no memory block, and the notes close the prompt.
    assert not re.search(r"^> ", text, flags=re.M) and "<memory_documents>" not in text
    assert text.endswith(f"\n\n{RUNTIME_NOTES}")


def test_the_runtime_notes_are_the_records_block_verbatim() -> None:
    template = REPO / "src" / "wikitrace" / "envs" / "tau3" / "prompts" / "solver" / "_runtime_notes.j2"
    assert template.read_text(encoding="utf-8") == RUNTIME_NOTES + "\n"
    assert _system(access="inject", substrate="RCK", memory_documents="DOC").count(RUNTIME_NOTES) == 1


def test_the_user_prompt_is_the_lead_in_and_the_first_message() -> None:
    first = "Hi, I'd like to return two items from order #W1234567.\nCan you help?"
    assert render_user("tau3_solver", task=first, past_runs="") == f"Here is the task.\n{first}"


def test_the_compiler_excerpt_is_the_signature_list_between_the_policy_and_the_notes() -> None:
    excerpt = resolve("tau3", "tau3").prompt_profile("retail").solver_prompt
    data = read_prompt_data()
    assert excerpt.startswith(f"<instructions>\n{data['instruction_joined']}\n</instructions>\n\n<policy>\n{data['policy']}\n</policy>\n\n")
    assert "\n".join(data["signatures"]) in excerpt and "<tool name=" not in excerpt
    assert len(re.findall(r"^- \w+\(.*\)$", excerpt, flags=re.M)) == 16
    assert excerpt.endswith(f"\n\n{RUNTIME_NOTES}")
    assert 8_500 < len(excerpt) < 9_500


@pytest.mark.skipif(not VENV_PYTHON.exists(), reason="data/tau3/.venv is not installed")
def test_the_committed_tools_data_is_what_the_generator_writes_from_the_pinned_clone() -> None:
    done = subprocess.run([str(VENV_PYTHON), str(GENERATOR), "--check"], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads(PROMPT_DATA_PATH.read_text(encoding="utf-8"))["domain"] == "retail"


