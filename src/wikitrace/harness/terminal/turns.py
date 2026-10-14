"""Turn-stream helpers for terminal-agent trajectories, domain-neutral."""

from __future__ import annotations

import json
import re
from pathlib import Path

from wikitrace.harness.terminal.sentinel import extract_json_fallback


def event_to_turn(turn_idx: int, event: dict) -> dict:
    """Map a Pi RPC ``tool_execution_start`` event to a ``function_call`` turn."""
    return {
        "turn": turn_idx,
        "type": "function_call",
        "role": "assistant",
        "call_id": event.get("toolCallId") or "",
        "name": event.get("toolName"),
        "arguments": event.get("args"),
    }


def persisted_turns(turns: list[dict]) -> list[dict]:
    """The on-disk copy of a turn stream: Pi's ``toolCallId`` never leaves memory."""
    return [{k: v for k, v in t.items() if k != "call_id"} for t in turns]


# Matched on the raw command STRING; tokenizing the JSON arguments corrupts path tokens and drops every `cat`-read.
_CORPUS_PATH_RE = re.compile(r"(?:knowledge|compilation|raw|insight)/[\w./-]+")


def parse_read_set(event: dict, corpus_dir: Path) -> list[str]:
    """Best-effort: corpus-relative paths the agent opened in this event."""
    # A bare `grep -rin ./corpus/` names no concrete file or dir, so it is not a read.
    args = event.get("args")
    if not args:
        return []
    if isinstance(args, dict):
        text = " ".join(str(v) for v in args.values() if v)
    else:
        text = str(args)
    found = [m.group(0).rstrip(".,;:)('\"`") for m in _CORPUS_PATH_RE.finditer(text)]
    if re.search(r"\bindex\.md\b", text):
        found.append("index.md")
    return list(dict.fromkeys(found))


# Anchored on the layer, not the bare id: a compilation shares its id one layer up, and reading it is the EXPECTED first hop DOWN.
_LAYER_ID_RE = {
    "R": re.compile(r"raw/(step-\d{4})"),
    "C": re.compile(r"compilation/(step-\d{4})"),
    "K": re.compile(r"knowledge/(document-\d{4})"),
    "I": re.compile(r"insight/(insight-\d{4})"),
}


def read_set_by_layer(read_set: list[str], layers: tuple[str, ...]) -> dict[str, list[str]]:
    """The ids opened per present layer, in order of first opening."""
    out: dict[str, list[str]] = {L: [] for L in layers}
    for p in read_set:
        for L in layers:
            for sid in _LAYER_ID_RE[L].findall(p):
                if sid not in out[L]:
                    out[L].append(sid)
    return out


_CITED_ID_RE = {
    "R": re.compile(r"step-\d{4}"),
    "C": re.compile(r"step-\d{4}"),
    "K": re.compile(r"document-\d{4}"),
    "I": re.compile(r"insight-\d{4}"),
}


def parse_citation(text: str, layers: tuple[str, ...]) -> dict[str, list[str]] | None:
    """The citation turn's answer as ids per present layer; ``None`` when unparseable."""
    data = extract_json_fallback(text)
    if data is None:
        return None
    out: dict[str, list[str]] = {}
    for L in layers:
        ids: list[str] = []
        for item in data.get(L) or []:
            for sid in _CITED_ID_RE[L].findall(str(item)):
                if sid not in ids:
                    ids.append(sid)
        out[L] = ids
    return out


LAYERS = ("knowledge", "compilation", "raw", "schema", "insight")
_LAYER_PATH_RE = re.compile(r"/work/corpus/((?:knowledge|compilation|raw|schema|insight)/[^\s\"'`)]+)")


def call_text(name: str | None, args: dict) -> str:
    """Command string for any tool call: bash carries ``command``, Pi's ``read`` a ``path``."""
    if isinstance(args, dict):
        if args.get("command"):
            return args["command"]
        if args.get("path"):
            return f"{name or 'read'} {args['path']}"
    return f"{name or 'tool'} {json.dumps(args)}"


def reasoning_from_context(final_context) -> tuple[list[str], str]:
    """Pull per-tool-call reasoning out of a saved request context (domain-neutral)."""
    per_call: list[str] = []
    trailing = ""
    for m in final_context or []:
        if (m or {}).get("role") != "assistant":
            continue
        blocks = m.get("content") if isinstance(m.get("content"), list) else []
        think = " ".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking").strip()
        calls = [b for b in blocks if b.get("type") == "toolCall"]
        if calls:
            per_call.extend(think for _ in calls)
        elif think:
            trailing = think
    return per_call, trailing


def reasoning_by_call_id(final_context) -> dict[str, str]:
    """``{toolCallId: thinking}`` -- the reasoning that produced each call, keyed."""
    # Keyed by id, not position, so a call with missing neighbours gets ITS OWN reasoning.
    out: dict[str, str] = {}
    for m in final_context or []:
        if (m or {}).get("role") != "assistant":
            continue
        blocks = m.get("content") if isinstance(m.get("content"), list) else []
        think = " ".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking").strip()
        for b in blocks:
            if b.get("type") == "toolCall" and b.get("id"):
                out[str(b["id"])] = think
    return out


def align_calls(fc_turns: list[dict], final_context) -> list[dict]:
    """Pair each ``function_call`` turn with its reasoning and its result."""
    # Pairs by id, or by position only when both streams have the same length: a wrong pairing would feed the compiler a false ``result``.
    results = tool_results_from_context(final_context)
    by_id = {r["call_id"]: r for r in results if r.get("call_id")}
    reasons_by_id = reasoning_by_call_id(final_context)
    turn_ids = [str(t.get("call_id") or "") for t in fc_turns]

    if all(turn_ids) and len(by_id) == len(results):
        return [
            {"reasoning": reasons_by_id.get(cid, ""), "result": by_id.get(cid)}
            for cid in turn_ids
        ]

    per_call, _ = reasoning_from_context(final_context)
    if len(results) == len(fc_turns):
        reasons = per_call if len(per_call) == len(fc_turns) else [""] * len(fc_turns)
        return [{"reasoning": reasons[i], "result": results[i]} for i in range(len(fc_turns))]

    return [{"reasoning": "", "result": None} for _ in fc_turns]


def tool_results_from_context(final_context) -> list[dict]:
    """Per-tool-call OUTPUT + error flag, in call order (zips onto function_call turns)."""
    out: list[dict] = []
    for m in final_context or []:
        if (m or {}).get("role") != "toolResult":
            continue
        blocks = m.get("content") if isinstance(m.get("content"), list) else []
        text = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        out.append({
            "call_id": m.get("toolCallId") or m.get("tool_call_id") or "",
            "text": text,
            "is_error": bool(m.get("isError") or m.get("is_error")),
            "state": _result_state(text),
        })
    return out


# Anchored to the END so an output that merely QUOTES the phrase mid-text is not mislabeled.
_TRUNC_TAIL_RE = re.compile(r"\[\.\.\.truncated, \d+ chars omitted\]\s*\Z")


def _result_state(text: str) -> str:
    """The runtime context-management mark the terminal harness left on a tool result."""
    # Full compaction DROPS a call's messages, surfacing as a length mismatch in ``tool_steps_from_log``.
    if text.strip() == "[cleared]":
        return "erased"
    if _TRUNC_TAIL_RE.search(text):
        return "truncated"
    return ""


def last_model_returned(usage_calls: list[dict] | None) -> str | None:
    """The model that answered the LAST provider round-trip of a terminal phase."""
    for call in reversed(usage_calls or []):
        model = call.get("model_returned")
        if model:
            return str(model)
    return None
