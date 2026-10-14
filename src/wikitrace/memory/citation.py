"""The test-time citation turn's message, and the citation record built from an answer."""

from __future__ import annotations

from typing import Iterable


#: The ``llm_calls.jsonl`` role the citation turn is costed under, apart from the generator.
CITATION_ROLE = "citation"

_LAYER_LINES = {
    "R": '- `R`: cases, e.g. ["step-0023"]',
    "C": '- `C`: compilations, e.g. ["step-0023"]',
    "K": '- `K`: documents, e.g. ["document-0007"]',
    "I": '- `I`: insights, e.g. ["insight-0040"]',
}


def citation_message(layers: Iterable[str]) -> str:
    """The message asking the agent for the ids it used, per present layer, as JSON."""
    present = set(layers)
    lines = dict(_LAYER_LINES)
    if "I" in present:
        lines["R"] = '- `R`: raw trajectories, e.g. ["step-0023"]'
    return "\n".join([
        "The episode is over. Do not call any tool. Output a single JSON object and nothing",
        "else, no markdown and no code fence. Each key holds a list of the materially-used ids,",
        "possibly empty:",
        *(lines[L] for L in "RCKI" if L in present),
        "Cite only what affected your actions. Do not cite an item merely because it appeared.",
    ])

