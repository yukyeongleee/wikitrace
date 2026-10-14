"""The terminal solver's tool results read back from Pi's final context, and the
sequence elision the compiler render applies to them."""

from __future__ import annotations

from wikitrace.harness.terminal import tool_results_from_context
from wikitrace.memory.compiler import SEQUENCE_HEAD_CHARS, _elide_sequences

# 100 residues from the elide alphabet — safely over the sequence-run threshold.
SEQ = "ACDEFGHIKLMNPQRSTUVWY" * 5


def _ctx() -> list[dict]:
    """A canonical solver ``final_context``: two calls, then a trailing thought."""
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "probe env"},
            {"type": "toolCall"},
        ]},
        {"role": "toolResult", "content": [{"type": "text", "text": "python: not found"}], "isError": True},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "use bash"},
            {"type": "toolCall"},
        ]},
        {"role": "toolResult", "content": [{"type": "text", "text": "42"}], "isError": False},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "answer now"}]},
    ]


def test_tool_results_from_context_extracts_ordered_output_and_error_flag():
    assert tool_results_from_context(_ctx()) == [
        {"call_id": "", "text": "python: not found", "is_error": True, "state": ""},
        {"call_id": "", "text": "42", "is_error": False, "state": ""},
    ]


def test_tool_results_from_context_accepts_both_error_keys_and_skips_non_results():
    ctx = [
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "x"}]},
        {"role": "toolResult", "content": [{"type": "text", "text": "a"}], "is_error": True},
        {"role": "toolResult", "content": [{"type": "text", "text": "b"}], "isError": True},
        {"role": "toolResult", "content": [{"type": "text", "text": "c"}]},
    ]
    assert tool_results_from_context(ctx) == [
        {"call_id": "", "text": "a", "is_error": True, "state": ""},
        {"call_id": "", "text": "b", "is_error": True, "state": ""},
        {"call_id": "", "text": "c", "is_error": False, "state": ""},
    ]


def test_tool_results_from_context_handles_empty_none_and_nonlist_content():
    assert tool_results_from_context(None) == []
    assert tool_results_from_context([]) == []
    assert tool_results_from_context([{"role": "toolResult", "content": "oops"}]) == [
        {"call_id": "", "text": "", "is_error": False, "state": ""},
    ]


def test_tool_results_from_context_joins_text_blocks_ignoring_non_text():
    ctx = [{"role": "toolResult", "content": [
        {"type": "text", "text": "line1"},
        {"type": "image", "data": "..."},
        {"type": "text", "text": "line2"},
    ]}]
    assert tool_results_from_context(ctx) == [{"call_id": "", "text": "line1 line2", "is_error": False, "state": ""}]


def test_tool_results_from_context_flags_runtime_erased_and_truncated_states():
    # Pi's level3 context management rewrites a result before it re-enters context:
    # micro-compaction blanks it to "[cleared]"; the size cap appends a truncation tail.
    ctx = [
        {"role": "toolResult", "content": [{"type": "text", "text": "[cleared]"}]},
        {"role": "toolResult", "content": [{"type": "text", "text": "head\n[...truncated, 5000 chars omitted]"}]},
        {"role": "toolResult", "content": [{"type": "text", "text": "intact output"}]},
        # a genuine output that merely QUOTES the marker mid-text is NOT Pi's tail:
        # Pi only ever appends it at the very end, so end-anchored detection ignores this.
        {"role": "toolResult", "content": [{"type": "text", "text": "log: [...truncated, 5 chars omitted] and more"}]},
    ]
    assert [r["state"] for r in tool_results_from_context(ctx)] == ["erased", "truncated", "", ""]


def test_elide_sequences_marker_is_compact_and_carries_no_pointer():
    """The marker itself states only head + total + alphabet."""
    assert _elide_sequences(SEQ) == f"{SEQ[:SEQUENCE_HEAD_CHARS]}…[{len(SEQ)}aa]"
    assert "raw/" not in _elide_sequences(SEQ)
