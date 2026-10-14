"""Register the tau3 generator's prompt role."""

from wikitrace.harness.prompts import register_prompt

register_prompt("tau3_solver", "envs/tau3/prompts/solver")
