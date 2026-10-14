"""The tau3 `take_action` extension: one JSON step string, string or object argument, one call per turn, the citation mark."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from wikitrace.harness.environment import EnvTool
from wikitrace.harness.environment.tool import CITATION_ENV_VAR, EXTENSIONS_DIR, SOCKET_ENV_VAR, SOCKET_NAME
from tests.unit.test_environment_harness import _CITATION_BLOCK_EXT, _JITI, _LOAD_EXT

EXT = EXTENSIONS_DIR / "take_action_tau3.ts"


def test_the_extension_registers_take_action_with_a_string_or_object_argument() -> None:
    tool = EnvTool("take_action", Path("/tmp"), "take_action_tau3")
    assert tool.extension == EXT and tool.name == "take_action"
    text = EXT.read_text()
    assert 'name: "take_action"' in text and "registerTool" in text
    assert 'required: ["action", "argument"]' in text
    assert 'anyOf: [{ type: "string" }, { type: "object", additionalProperties: true }]' in text
    assert "enum" not in text and "additionalProperties: false" in text
    assert f"process.env.{SOCKET_ENV_VAR}" in text and f"process.env.{CITATION_ENV_VAR}" in text
    assert 'op: "step"' in text and "JSON.stringify({ action:" in text
    assert "one take_action per turn" in text
    for line in text.splitlines():
        if line.startswith("import "):
            assert 'from "node:' in line, line
    # The description is the record's harness decision, never the crafting one: no `respond`,
    # since a reply without a tool call is the message to the user.
    assert "retail tools" in text and "respond" not in text and "crafting" not in text


@pytest.mark.skipif(not shutil.which("node") or not _JITI.exists(), reason="needs node and jiti")
@pytest.mark.parametrize("args,sent", [
    ('action: "respond", argument: "hi there"', '{"action":"respond","argument":"hi there"}'),
    ('action: "get_order_details", argument: { order_id: "#W1", item_ids: ["a", "b"] }',
     '{"action":"get_order_details","argument":{"order_id":"#W1","item_ids":["a","b"]}}'),
])
def test_the_extension_sends_one_json_string_and_admits_one_call_per_turn(tmp_path, args, sent):
    script = tmp_path / "load_ext.mjs"
    script.write_text(_LOAD_EXT.format(jiti=_JITI, ext=EXT, name="take_action", args=args))
    proc = subprocess.run(
        ["node", str(script), str(tmp_path / SOCKET_NAME)], capture_output=True, text=True,
        timeout=60, check=True,
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    first, second, other, third = out["results"]
    assert first is None and other is None and third is None
    assert second["block"] is True and second["reason"].startswith("one take_action per turn")
    assert out["served"] == 1 and out["inputs"] == [sent]
    assert out["result"] == {"content": [{"type": "text", "text": "ok"}]}


# The citation prompt the runner sends after a message ended the episode marks it queued, so a
# later `Episode over` reply queues no second steer.
_INPUT_EXT = """
import {{ createJiti }} from "{jiti}";
import net from "node:net";
const sock = process.argv[2];
process.env.AGENT_WIKI_ENV_SOCKET = sock;
const srv = net.createServer((c) => c.on("data", () => c.end('{{"observation":"Episode over","done":true,"over":true}}\\n')));
await new Promise((r) => srv.listen(sock, r));
const tools = [], handlers = {{}}, steers = [];
const mod = await createJiti(import.meta.url).import("{ext}");
(mod.default ?? mod)({{ registerTool: (t) => tools.push(t), on: (ev, fn) => {{ handlers[ev] = fn; }},
                     sendUserMessage: (m, o) => steers.push([m, o]) }});
for (const text of JSON.parse(process.argv[3])) handlers.input({{ type: "input", text, source: "rpc" }});
await tools[0].execute("id", {{ action: "get_user_details", argument: {{}} }});
srv.close();
console.log(JSON.stringify({{ steers }}));
"""


@pytest.mark.skipif(not shutil.which("node") or not _JITI.exists(), reason="needs node and jiti")
@pytest.mark.parametrize("inputs,steers", [
    (["Here is the task.", "a user reply"], [["cite now", {"deliverAs": "steer"}]]),
    (["Here is the task.", "cite now"], []),
])
def test_a_citation_prompt_sent_by_the_runner_marks_the_citation_queued(tmp_path, monkeypatch, inputs, steers):
    monkeypatch.setenv(CITATION_ENV_VAR, "cite now")
    script = tmp_path / "input_ext.mjs"
    script.write_text(_INPUT_EXT.format(jiti=_JITI, ext=EXT))
    proc = subprocess.run(
        ["node", str(script), str(tmp_path / SOCKET_NAME), json.dumps(inputs)], capture_output=True, text=True,
        timeout=60, check=True,
    )
    assert json.loads(proc.stdout.strip().splitlines()[-1])["steers"] == steers


_BLOCK_ARGS = 'action: "get_user_details", argument: { user_id: "u1" }'
_BLOCKED = {"block": True, "reason": "The episode is over. Answer with the JSON object only."}


@pytest.mark.skipif(not shutil.which("node") or not _JITI.exists(), reason="needs node and jiti")
def test_every_tool_is_blocked_from_the_turn_the_steered_citation_reaches_the_model(tmp_path, monkeypatch):
    monkeypatch.setenv(CITATION_ENV_VAR, "cite now")
    script = tmp_path / "block_ext.mjs"
    script.write_text(_CITATION_BLOCK_EXT.format(jiti=_JITI, ext=EXT, name="take_action", args=_BLOCK_ARGS))
    proc = subprocess.run(
        ["node", str(script), str(tmp_path / SOCKET_NAME)], capture_output=True, text=True,
        timeout=60, check=True,
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["mid"] == [None] * 5 and out["ending"] == [None, None]
    assert out["steers"] == [["cite now", {"deliverAs": "steer"}]]
    assert out["after"] == [_BLOCKED] * 3


# The citation prompt the runner sends after a message ended the episode blocks every tool
# from the turn it starts; a user reply does not.
_INPUT_BLOCK_EXT = """
import {{ createJiti }} from "{jiti}";
process.env.AGENT_WIKI_ENV_SOCKET = process.argv[2];
const handlers = {{}};
const mod = await createJiti(import.meta.url).import("{ext}");
(mod.default ?? mod)({{ registerTool: () => {{}}, on: (ev, fn) => {{ handlers[ev] = fn; }}, sendUserMessage: () => {{}} }});
const call = (toolName) => handlers.tool_call({{ toolName, toolCallId: "x", input: {{}} }}) ?? null;
const out = [];
for (const text of ["a user reply", "cite now"]) {{
  handlers.input({{ type: "input", text, source: "rpc" }});
  handlers.turn_start({{ turnIndex: out.length }});
  out.push([call("read"), call("take_action")]);
}}
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(not shutil.which("node") or not _JITI.exists(), reason="needs node and jiti")
def test_a_citation_prompt_sent_by_the_runner_blocks_every_tool_from_its_turn(tmp_path, monkeypatch):
    monkeypatch.setenv(CITATION_ENV_VAR, "cite now")
    script = tmp_path / "input_block_ext.mjs"
    script.write_text(_INPUT_BLOCK_EXT.format(jiti=_JITI, ext=EXT))
    proc = subprocess.run(
        ["node", str(script), str(tmp_path / SOCKET_NAME)], capture_output=True, text=True,
        timeout=60, check=True,
    )
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == [[None, None], [_BLOCKED, _BLOCKED]]
