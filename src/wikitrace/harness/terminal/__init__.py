"""The DCI terminal-agent harness; task glue lives in each domain's `traverse.py`."""

from wikitrace.harness.terminal.runner import (
    FakeTraverseRunner,
    PiTraverseRunner,
    TerminalResult,
    TraverseRunner,
)
from wikitrace.harness.terminal.docker import prepare_session_dir
from wikitrace.harness.terminal.sentinel import extract_json_fallback, read_sentinel
from wikitrace.harness.terminal.turns import (
    align_calls,
    last_model_returned,
    persisted_turns,
    reasoning_by_call_id,
    reasoning_from_context,
    tool_results_from_context,
)

__all__ = [
    "TraverseRunner",
    "TerminalResult",
    "FakeTraverseRunner",
    "PiTraverseRunner",
    "prepare_session_dir",
    "read_sentinel",
    "extract_json_fallback",
    "align_calls",
    "persisted_turns",
    "reasoning_from_context",
    "reasoning_by_call_id",
    "tool_results_from_context",
    "last_model_returned",
]
