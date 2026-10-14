"""tau3 generator: one Pi session per conversation, acting on the host-side tau3 environment."""
# Every method runs this loop; only what is mounted at /work/corpus differs.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from wikitrace.config import LLMCallRecord, ToolStep
from wikitrace.envs.tau3.benchmarks.tau3 import read_prompt_data
from wikitrace.envs.tau3.envs.tau3 import parse_command
from wikitrace.envs.tau3.models import TauAction, TauBenchResult, TauMessage
from wikitrace.envs.textgame.models import TextGameOutcome, TextGameSample
from wikitrace.envs.textgame.workflow import (
    PHASE,
    TextGamePhaseLog,
    _citation,
    _unresolved_reason,
    result_from_server,
)
from wikitrace.harness.environment import EnvTool, EnvironmentSession, close_reason, step_input
from wikitrace.harness.prompts import render_system, render_template, render_user, replace_each
from wikitrace.harness.terminal import TraverseRunner, align_calls
from wikitrace.harness.terminal.turns import call_text
from wikitrace.memory.citation import citation_message

ROLE = "tau3_solver"


SOURCE_CHECK_EDITS = {
    "instruction_joined": ((
        "You are a customer service agent that helps the user according to the <policy> provided below. "
        "In each turn you can either:\n- Send a message to the user.\n- Make a tool call.\nYou cannot do both at "
        "the same time. Try to be helpful and always follow the policy. Always make sure you generate valid JSON "
        "only.",
        "You are a customer service agent that helps the user according to the <policy> provided below. "
        "Insights from past runs come with the task message, and the wiki at /work/corpus holds the runs behind "
        "them. In each turn you can either:\n- Send a message to the user.\n- Make a tool call to act on the "
        "store.\n- Read the wiki with the `read` or `bash` tool. The user does not see this.\nYou cannot send a "
        "message and make a tool call at the same time. The insights can guide how you handle a request. Before "
        "relying on one, it helps to see how its source run used it. Try to be helpful and always follow the "
        "policy. Always make sure you generate valid JSON only.",
    ),),
    "policy": ((
        "At the beginning of the conversation, you have to authenticate the user identity by locating their "
        "user id via email, or via name + zip code. This has to be done even when the user already provides "
        "the user id.",
        "Do not reply to the user right away. First check the insights you will rely on in their source runs "
        "in the wiki, which the user does not see. Then, before anything else in the conversation, you have to "
        "authenticate the user identity by locating their user id via email, or via name + zip code. This has "
        "to be done even when the user already provides the user id.",
    ), (
        "You should not make up any information or knowledge or procedures not provided by the user or the "
        "tools, or give subjective recommendations or comments.",
        "You should not make up any information or knowledge, or give subjective recommendations or comments. "
        "Information comes only from the user and the tools. For how to handle a request, you may use the "
        "reusable procedural insights in the task message, after you check their evidence in the wiki and find "
        "that it fits this request and this policy.",
    ),),
}


NOTES = "envs/tau3/prompts/solver/_runtime_notes.j2"
SOURCE_CHECK_NOTES = ((
    "after ten rejected or failed tool\n  calls, or when the step budget runs out. After that, `take_action` only "
    "answers\n  `Episode over`.",
    "after ten rejected or failed\n  `take_action` calls, or when the step budget runs out. After that, "
    "`take_action` only\n  answers `Episode over`.",
),)


_SOURCE_CHECK_V2_ANCHOR = ("First check the insights you will rely on in their source runs in the wiki, which the "
                           "user does not see.")
SOURCE_CHECK_V2_EDITS = ((
    _SOURCE_CHECK_V2_ANCHOR,
    _SOURCE_CHECK_V2_ANCHOR + " A source run handled another request, even when it is the same user, so take "
    "from it how the policy was applied, never which order, item, address or payment method it chose. Ask this "
    "user.",
),)


def source_check_prompt_data(data: dict[str, Any], version: int = 1) -> dict[str, Any]:
    """The authors' prompt data with the `-sourcecheck` edits of the given version applied."""
    out = {**data, **{key: replace_each(data[key], edits) for key, edits in SOURCE_CHECK_EDITS.items()}}
    if version == 2:
        out["policy"] = replace_each(out["policy"], SOURCE_CHECK_V2_EDITS)
    return out


def source_check_system(system: str) -> str:
    """A rendered solver system prompt with its runtime notes swapped for the `-sourcecheck` wording."""
    notes = render_template(NOTES)
    return replace_each(system, ((notes, replace_each(notes, SOURCE_CHECK_NOTES)),))


_META_FIELDS = ("orchestrator_steps", "messages", "num_errors", "stop_token", "judge_attempts", "judge_failed")


def command_text(step: str) -> str:
    """A step string as the renders show it: `<tool> <argument JSON>`."""
    parsed = parse_command(step)
    if isinstance(parsed, str):
        return step
    action, argument = parsed
    return f"{action} {json.dumps(argument, ensure_ascii=False, separators=(',', ':'))}"


def actions_from_trajectory(lines: list[dict]) -> list[TauAction]:
    """The server's step record as command/observation pairs, each observation with its source mark."""
    out: list[TauAction] = []
    pending: dict[int, str] = {}
    for rec in lines:
        step = int(rec.get("step", 0))
        if rec.get("role") == "agent":
            pending[step] = str(rec.get("command", ""))
        elif rec.get("role") == "environment" and step > 0:
            info = rec.get("info") or {}
            out.append(TauAction(
                step=step, command=pending.pop(step, ""), observation=str(rec.get("observation", "")),
                score=float(info.get("score", 0.0)), invalid=bool(info.get("invalid_command")),
                source=str(rec.get("source", "")),
            ))
    return out


def messages_from_turns(turns: list[dict]) -> list[TauMessage]:
    """Each message the agent sent the user, with the reply, placed by the tool calls before it."""
    out: list[TauMessage] = []
    calls = 0
    for t in turns:
        kind, role = t.get("type"), t.get("role")
        if kind == "request_context":
            # The conversation ends here; a final reply row after it was never sent to the user.
            break
        if kind == "function_call":
            calls += 1
        elif kind == "message" and role == "assistant":
            out.append(TauMessage(after=calls, text=str(t.get("content") or ""),
                                  reasoning=str(t.get("reasoning") or "")))
        elif kind == "message" and role == "user" and out and out[-1].reply is None and out[-1].after == calls:
            out[-1].reply = str(t.get("content") or "")
    return out


def conversation(tool_steps: Sequence[ToolStep], messages: Sequence[TauMessage]) -> list[ToolStep | TauMessage]:
    """Tool steps and messages in the order the agent took them."""
    keyed = [((j, 1), s) for j, s in enumerate(tool_steps)] + [((m.after, 0), m) for m in messages]
    return [x for _, x in sorted(keyed, key=lambda p: p[0])]


def _episode_meta(evaluation: dict[str, Any]) -> dict[str, Any]:
    """The authors' counters and the per-role simulator and judge accounting, as analysis facts."""
    meta = {k: evaluation[k] for k in _META_FIELDS if k in evaluation}
    for role, key in (("user_simulator", "simulator"), ("judge", "judge")):
        for kind in ("calls", "cost"):
            if role in (evaluation.get(kind) or {}):
                meta[f"{key}_{kind}"] = evaluation[kind][role]
    return meta


def env_call_records(evaluation: dict[str, Any], sample: TextGameSample) -> list[LLMCallRecord]:
    """The simulator's and the judge's calls as rows of the run's `llm_calls.jsonl`, by role."""
    return [
        LLMCallRecord(
            **row, call_id=f"tau3-{row.get('role', '')}:{sample.split_role}:{sample.id}:{i}",
            phase=sample.split_role, sample_id=sample.id,
        )
        for i, row in enumerate(evaluation.get("llm_calls") or []) if isinstance(row, dict)
    ]


def _deliver_calls(served: dict[str, Any], sample: TextGameSample,
                   on_call: Callable[[LLMCallRecord], None] | None) -> None:
    """Hand the simulator's and the judge's call rows of a closed episode to ``on_call``."""
    if on_call is not None:
        for record in env_call_records(dict(served.get("evaluation") or {}), sample):
            on_call(record)


def _command(turn: dict, tool: str, extension: str) -> str:
    """A tool call as one line; an environment action keeps the transcript's `> ` mark."""
    name = turn.get("name")
    args = turn.get("arguments") or {}
    if name == tool and isinstance(args, dict):
        return f"> {step_input(tool, args, extension)}"
    return call_text(name, args)


def tool_steps_from_log(log: TextGamePhaseLog, tool: str, extension: str) -> list[ToolStep]:
    """Every Pi tool call in order — actions and corpus reads — paired with its result."""
    fcs = [t for t in log.turns if t.get("type") == "function_call"]
    unresolved = _unresolved_reason(log)
    steps: list[ToolStep] = []
    for t, p in zip(fcs, align_calls(fcs, log.final_context)):
        res = p["result"]
        steps.append(ToolStep(
            reasoning=p["reasoning"], command=_command(t, tool, extension),
            result=(res or {}).get("text", ""), is_error=bool((res or {}).get("is_error")),
            state=(res or {}).get("state", ""),
            unresolved="" if res is not None else unresolved,
        ))
    return steps


def run_tau3_episode(
    sample: TextGameSample, runner: TraverseRunner, corpus_dir: Path | None, session_root: Path,
    *, bench: Any, memory_text: str = "", user_prefix: str = "", tools: str = "read,bash",
    max_turns: int = 50, sentinel_name: str = "__out.json", memory: bool = False,
    memory_layers: tuple[str, ...] = (), descend_block: str = "", bind_memory: Any = None,
    source_check: bool = False, source_check_version: int = 1,
    readonly_files: Mapping[str, Path] | None = None,
    on_call: Callable[[LLMCallRecord], None] | None = None, **_unused: Any,
) -> TextGameOutcome:
    """Play one conversation: start the environment, run the Pi session against it, read the reward."""
    # The simulator and the judge call from the environment server under this worker's key; `on_call` takes their rows.
    server_env = dict(bench.server_env)
    if getattr(runner, "api_key", None):
        server_env["OPENROUTER_API_KEY"] = runner.api_key
    sdir = Path(session_root) / sample.id / PHASE
    sdir.mkdir(parents=True, exist_ok=True)
    logs: list[TextGamePhaseLog] = []
    prompt_data = source_check_prompt_data(read_prompt_data(), source_check_version) if source_check else read_prompt_data()
    session: EnvironmentSession | None = None
    calls_delivered = False
    try:
        with EnvironmentSession(
            factory=bench.env_factory, spec=sample.spec, out_dir=sdir,
            step_limit=bench.step_limit, end_command=bench.end_command, python=bench.python,
            env=server_env,
        ) as env:
            session = env
            try:
                observation, info = env.reset()
            except Exception:
                if bind_memory is not None:
                    sample.meta["memory_binding"] = "skipped: the reset failed"
                raise
            # The task exists only now; an method that queries memory by it binds here, and its metadata seeds the failure path too.
            sample.task_text = observation
            if bind_memory is not None:
                binding = bind_memory(sample)
                memory_text = binding.memory_text
                user_prefix = binding.user_prefix
                sample.meta.update(binding.meta)
            cite_layers = memory_layers
            system = render_system(ROLE, prompt_data=prompt_data, env_tool=bench.env_tool, memory=memory,
                                   descend_block=descend_block)
            system = source_check_system(system) if source_check else system
            user = render_user(ROLE, task=bench.task_block(observation, info), past_runs=memory_text)
            if user_prefix.strip("\n"):
                user = f"{user_prefix.strip(chr(10))}\n\n{user}"
            res = runner.run_session(
                system_prompt=system, user_msg=user, corpus_dir=corpus_dir, out_dir=sdir,
                tools=tools, max_turns=max_turns,
                sentinel_name=sentinel_name, label=PHASE,
                env_tool=EnvTool(bench.env_tool, env.socket_dir, bench.env_extension,
                                 messages=bench.message_channel),
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
            calls_delivered = True
            _deliver_calls(served, sample, on_call)
            actions = actions_from_trajectory(served.get("trajectory") or [])
    except BaseException as exc:
        # The session's round-trips were billed even when the episode threw; the log rides out on the exception.
        try:
            exc.traverse_phase_logs = logs
        except Exception:
            pass
        if session is not None and not calls_delivered:
            # Leaving the `with` closed the episode, so its evaluation holds every billed call.
            try:
                _deliver_calls(session.finish(), sample, on_call)
            except Exception:  # noqa: BLE001 — accounting must not mask the episode's own error
                pass
        raise
    # The reward is known only at close, so the evaluation, not the step info, says whether the episode succeeded.
    evaluation = dict(served.get("evaluation") or {})
    scored = {**served, "success": evaluation.get("success", False), "score": evaluation.get("score", 0.0)}
    base = result_from_server(sample, scored, actions, bench.initial_observation(observation, info))
    result = TauBenchResult(**base.model_dump(exclude={"actions"}), actions=actions)
    result.meta.update(_episode_meta(evaluation))
    if evaluation.get("generation_error") and not result.generation_error:
        result.generation_error = str(evaluation["generation_error"])
    if res.provider_error and not result.generation_error:
        result.generation_error = f"provider_error: {res.provider_error}"
    if "reward" not in evaluation and not result.generation_error:
        result.generation_error = "episode ended without an evaluation"
    if result.generation_error:
        result.success, result.score = False, 0.0
    result.citations = [_citation(sample.id, res.read_set, res.citation, layers=cite_layers)]
    result.tool_steps = tool_steps_from_log(log, bench.env_tool, bench.env_extension)
    result.messages = messages_from_turns(log.turns)
    return TextGameOutcome(result=result, phase_logs=logs,
                           generation_failed=bool(result.generation_error))
