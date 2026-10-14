"""Register the compiler prompt role."""

from __future__ import annotations

from wikitrace.harness.prompts import register_prompt

register_prompt("compiler", "memory/prompts/compiler")
