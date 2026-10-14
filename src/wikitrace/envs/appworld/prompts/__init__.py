"""Register the appworld generator's prompt role."""

from wikitrace.harness.prompts import register_prompt

register_prompt("appworld_solver", "envs/appworld/prompts/solver")
