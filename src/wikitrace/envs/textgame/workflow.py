"""textgame generator: one Pi session per episode, acting on a host-side simulator."""
# Every method runs this loop; only what is mounted at /work/corpus differs.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from wikitrace.config import CitationRecord, Phase, ToolStep
from wikitrace.envs.base import _unresolved_reason
from wikitrace.envs.textgame.models import Action, TextGameOutcome, TextGameResult, TextGameSample
from wikitrace.harness.environment import EnvTool, EnvironmentSession, close_reason, step_input
from wikitrace.harness.prompts import render_system, render_user
from wikitrace.harness.terminal import TraverseRunner, align_calls
from wikitrace.harness.terminal.turns import call_text, read_set_by_layer
from wikitrace.memory.citation import citation_message

ROLE = "textgame_solver"
PHASE = "solver"
ENV_TOOL = "act"

@dataclass
class TextGamePhaseLog:
    """The one terminal phase's trajectory, in the shape `experiments/` reads for cost."""

    phase: str
    turns: list[dict]
    read_set: list[str]
    sentinel: Optional[dict]
    final_text: str
    final_context: object = None
    final_reasoning: str = ""
    system_prompt: str = ""
    user_prompt: str = ""
    usage: dict = field(default_factory=dict)
    usage_calls: list = field(default_factory=list)
    stopped_by: str = ""
    last_stop_reason: str = ""
    last_had_tool_call: bool = False
    truncated_calls: int = 0
    provider_error: str = ""


def _command(turn: dict, tool: str = ENV_TOOL) -> str:
    """A tool call as one line; an environment command keeps the transcript's `> ` mark."""
    name = turn.get("name")
    args = turn.get("arguments") or {}
    if name == tool and isinstance(args, dict):
        return f"> {step_input(tool, args)}"
    return call_text(name, args)


def tool_steps_from_log(log: TextGamePhaseLog, tool: str = ENV_TOOL) -> list[ToolStep]:
    """Every Pi tool call in order — commands and corpus reads — paired with its result."""
    fcs = [t for t in log.turns if t.get("type") == "function_call"]
    unresolved = _unresolved_reason(log)
    steps: list[ToolStep] = []
    for t, p in zip(fcs, align_calls(fcs, log.final_context)):
        res = p["result"]
        steps.append(ToolStep(
            reasoning=p["reasoning"], command=_command(t, tool),
            result=(res or {}).get("text", ""), is_error=bool((res or {}).get("is_error")),
            state=(res or {}).get("state", ""),
            unresolved="" if res is not None else unresolved,
        ))
    return steps


def actions_from_trajectory(lines: list[dict]) -> list[Action]:
    """The server's step record as command/observation pairs (step 0 is the reset)."""
    out: list[Action] = []
    pending: dict[int, str] = {}
    for rec in lines:
        step = int(rec.get("step", 0))
        if rec.get("role") == "agent":
            pending[step] = str(rec.get("command", ""))
        elif rec.get("role") == "environment" and step > 0:
            info = rec.get("info") or {}
            out.append(Action(
                step=step, command=pending.pop(step, ""), observation=str(rec.get("observation", "")),
                score=float(info.get("score", 0.0)), invalid=bool(info.get("invalid_command")),
            ))
    return out


def result_from_server(
    sample: TextGameSample, served: dict[str, Any], actions: list[Action], initial_observation: str = "",
) -> TextGameResult:
    """The result the environment's own accounting states, before the agent's logs are attached."""
    return TextGameResult(
        sample_id=sample.id, db_id=sample.db_id, question=sample.task_text,
        task_type=sample.task_type, variant=sample.variant, meta=dict(sample.meta),
        initial_observation=initial_observation,
        success=bool(served.get("success")), score=float(served.get("score", 0.0)),
        failed=bool(served.get("failed")), steps=int(served.get("steps", len(actions))),
        step_limit=int(served.get("step_limit", 0)),
        step_limit_hit=bool(served.get("step_limit_hit")),
        invalid_commands=int(served.get("invalid_commands", 0)),
        ended_by=str(served.get("ended_by", "")), actions=actions,
        ended_on_end_command=bool(served.get("end_command")),
        generation_error=str(served.get("env_error", "")),
    )


def _citation(
    sample_id: str, read_set: list[str], cited: Optional[dict], *, layers: tuple[str, ...],
) -> CitationRecord:
    """What the agent said it used and what it opened, per layer."""
    return CitationRecord(
        sample_id=sample_id, phase=Phase.SOLVER,
        cited=cited or {L: [] for L in layers},
        read=read_set_by_layer(read_set, layers),
    )


SOURCE_CHECK_LINE = ("The examples show only game commands and leave out the `read` and `bash` calls that check "
                 "insights in the wiki.")


def source_check_exemplars(exemplars: str) -> str:
    """The exemplar block with the `-sourcecheck` line after its last example, trailing newlines kept."""
    body = exemplars.rstrip("\n")
    if not body:
        raise ValueError("no exemplar block to follow")
    return f"{body}\n\n{SOURCE_CHECK_LINE}{exemplars[len(body):]}"


def run_textgame_episode(
    sample: TextGameSample, runner: TraverseRunner, corpus_dir: Path | None, session_root: Path,
    *, bench: Any, memory_text: str = "", user_prefix: str = "", tools: str = "read,bash",
    max_turns: int = 50, sentinel_name: str = "__out.json", memory: bool = False,
    memory_layers: tuple[str, ...] = (), descend_block: str = "", source_check: bool = False,
    readonly_files: Mapping[str, Path] | None = None, **_unused: Any,
) -> TextGameOutcome:
    """Play one episode: start the simulator, run the Pi session against it, read the outcome."""
    cite_layers = memory_layers
    sdir = Path(session_root) / sample.id / PHASE
    sdir.mkdir(parents=True, exist_ok=True)
    logs: list[TextGamePhaseLog] = []
    try:
        with EnvironmentSession(
            factory=bench.env_factory, spec=sample.spec, out_dir=sdir,
            step_limit=bench.step_limit, end_command=bench.end_command, python=bench.python,
            env=dict(bench.server_env),
        ) as env:
            observation, info = env.reset()
            initial = bench.initial_observation(observation, info)
            if not bench.multiline_observations:
                initial = " ".join(initial.split())
            exemplars = bench.exemplars_for(sample)
            system = render_system(
                ROLE, intro=bench.solver_intro,
                exemplars=source_check_exemplars(exemplars) if source_check else exemplars,
                env_tool=bench.env_tool, memory=memory, descend_block=descend_block,
            )
            user = render_user(ROLE, task=bench.task_block(observation, info), past_runs=memory_text)
            if user_prefix.strip("\n"):
                user = f"{user_prefix.strip(chr(10))}\n\n{user}"
            res = runner.run_session(
                system_prompt=system, user_msg=user, corpus_dir=corpus_dir, out_dir=sdir,
                tools=tools, max_turns=max_turns,
                sentinel_name=sentinel_name, label=PHASE,
                env_tool=EnvTool(bench.env_tool, env.socket_dir),
                citation_prompt=citation_message(cite_layers) if cite_layers else None,
                citation_layers=cite_layers, readonly_files=readonly_files,
            )
            log = TextGamePhaseLog(
                phase=PHASE, turns=res.turns, read_set=res.read_set, sentinel=None,
                final_text=res.final_text, final_context=res.final_context,
                final_reasoning=res.final_reasoning,
                system_prompt=system, user_prompt=user, usage=dict(res.usage or {}),
                usage_calls=list(res.usage_calls or []), stopped_by=res.stopped_by,
                last_stop_reason=res.last_stop_reason, last_had_tool_call=res.last_had_tool_call,
                truncated_calls=int(res.truncated_calls or 0), provider_error=res.provider_error,
            )
            logs.append(log)
            served = env.finish(close_reason(res.stopped_by or res.provider_error))
            if not served:
                raise RuntimeError("environment server died before the episode was closed")
            actions = actions_from_trajectory(served.get("trajectory") or [])
    except BaseException as exc:
        # The session's round-trips were billed even when the episode threw; the log rides out on the exception.
        try:
            exc.traverse_phase_logs = logs
        except Exception:
            pass
        raise
    result = result_from_server(sample, served, actions, initial)
    if res.provider_error and not result.generation_error:
        result.generation_error = f"provider_error: {res.provider_error}"
    result.citations = [_citation(sample.id, res.read_set, res.citation, layers=cite_layers)]
    result.tool_steps = tool_steps_from_log(log, bench.env_tool)
    return TextGameOutcome(result=result, phase_logs=logs,
                           generation_failed=bool(result.generation_error))
