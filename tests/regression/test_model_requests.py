"""The final transport request of a malformed structured reply and its corrective retry."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from wikitrace.config import ModelSpec, ModelsConfig
from tests.fixtures.structured_model import SelectorOutput
from wikitrace.harness.llm import LLMClient

GOLDEN = Path(__file__).parents[1] / "fixtures/prompts/model_requests.json"


def capture_requests(fake_transport_factory):
    transport = fake_transport_factory(n_add=0)
    requests = []
    original_chat = transport.chat
    llm = LLMClient(ModelsConfig(default=ModelSpec(model="fake/model")), transport, max_retries=1)

    # Exercise the complete corrective message added after malformed structured output.
    first = True

    def malformed_then_valid(**kwargs):
        nonlocal first
        requests.append(deepcopy(kwargs))
        response = original_chat(**kwargs)
        if first:
            first = False
            response.content = "not json"
        return response

    transport.chat = malformed_then_valid
    llm.generate_structured("selector", SelectorOutput, "Fixed system.\n", "Fixed question.\n")
    return requests


def test_the_retry_request_pair_matches_the_recorded_capture(fake_transport_factory):
    actual = capture_requests(fake_transport_factory)
    assert actual == json.loads(GOLDEN.read_text())
    assert len(actual) == 2
