"""A structured-output model whose JSON schema the transport pins were recorded with."""

from __future__ import annotations

from typing import Literal, Union

from pydantic import Field

from wikitrace.config import StrictModel


TableSelection = Union[Literal["keep_all", "drop_all"], list[str]]


class SelectorOutput(StrictModel):
    """MAC-SQL selector format: a per-table keep_all / drop_all / [columns] map.

    Citation is our extension (not MAC-SQL): agent-wiki cites distilled knowledge docs in
    `document_ids`; raw-only cites raw train steps in `step_ids`. A given run populates
    exactly one (the prompt asks for one), the other stays empty.
    """

    schema_selection: dict[str, TableSelection] = Field(default_factory=dict)
    decision_summary: str = ""
    document_ids: list[str] = Field(default_factory=list)
    step_ids: list[str] = Field(default_factory=list)
