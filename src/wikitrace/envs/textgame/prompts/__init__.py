"""Register the textgame generator's prompt role."""

from wikitrace.harness.prompts import register_prompt

register_prompt("textgame_solver", "envs/textgame/prompts/solver")
