"""AppWorld episode flow: one environment server plus one Pi session per sample, outcome from the simulator."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from jinja2 import Template

from wikitrace.config import CitationRecord, Phase, ToolStep, TraverseOutcome
from wikitrace.envs.appworld.benchmarks.appworld import (
    EXEMPLAR_PATH,
    AppWorldResult,
    AppWorldSample,
    gold_answer_is_null,
    lenient_tgc,
)
from wikitrace.envs.appworld.session import AppWorldSession
from wikitrace.envs.base import _unresolved_reason
from wikitrace.harness.environment import EPISODE_OVER, EnvTool, close_reason
from wikitrace.harness.prompts import render_system, render_user, replace_each
from wikitrace.harness.terminal import TerminalResult, TraverseRunner, align_calls
from wikitrace.harness.terminal.turns import call_text, read_set_by_layer
from wikitrace.memory.citation import citation_message

ENV_TOOL = "exec"
# The exemplar ends with the task block, which is the user message; the system prompt takes the text before this marker.
_TASK_MARKER = "USER:\nUsing these APIs, now generate code to solve the actual task:"

SOURCE_CHECK_EDITS = (
    ("This environment will let you interact with app(s) using their associated APIs on my behalf.",
     "This environment will let you interact with app(s) using their associated APIs on my behalf. Insights "
     "from past runs come with the task message, and the wiki at /work/corpus holds the runs behind them."),
    ("- Remember that the email addresses, access tokens and variables (e.g. spotify_password) in the example "
     "above are not valid anymore.",
     "- Remember that the email addresses, access tokens and variables (e.g. spotify_password) in the example "
     "above and in the past runs in the wiki are not valid anymore."),
    ("- Always look at API specifications (using apis.api_docs.show_api_doc) before calling an API.",
     "- Do not look up API specifications right away. First check the insights you will rely on in their "
     "source runs in the wiki, with `read` or `bash`. Then always look at API specifications (using "
     "apis.api_docs.show_api_doc) before calling an API."),
    ("Task: How many playlists do I have in my Spotify playlist library?\n\n> exec\n",
     "Task: How many playlists do I have in my Spotify playlist library?\n\n"
     "[read / bash: checking the source runs of the insights to rely on, omitted]\n\n> exec\n"),
)

_KEY_APIS = "\n\nHere are three key APIs that you need to know to get more information"
SOURCE_CHECK_V2_EDITS = ((
    _KEY_APIS,
    "\n\nBefore your first `exec`, check the insights you will rely on in their source runs in the wiki, with "
    "`read` or `bash`. A source run solved another task from another state, so take from it which APIs and steps "
    "worked, never its values or IDs. Then use the APIs below." + _KEY_APIS,
),)

_RUNS_BEHIND = "the wiki at /work/corpus holds the runs behind them."
_INVENT = ("- Never invent or guess values. For example, if I ask you to play a song, do not assume the ID is 123. "
           "Instead, look it up properly through the right API.\n")
SOURCE_CHECK_V3_EDITS = (
    (_RUNS_BEHIND, _RUNS_BEHIND + " The insights can guide how you solve a task. Before relying on one, it helps "
                                  "to see how its source run used it."),
    (_INVENT, _INVENT + "- For how to solve a task, you may use the insights in the task message, after you check "
                        "their evidence in the wiki and find that it fits this task.\n"),
)


@dataclass
class AppWorldPhaseLog:

    """The solver session's trajectory, for persistence and cost accounting."""
    phase: str
    turns: list[dict]
    read_set: list[str]
    final_text: str = ""
    final_context: Any = None
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


def _as_exec_transcript(text: str) -> str:
    """The shipped example with role markers replaced by `> exec` and `output:`, no prose."""
    # A chat transcript invites code as reply text; this shape names the tool call each ASSISTANT turn stands for.
    lead, *parts = re.split(r"^(USER|ASSISTANT):\n", text, flags=re.M)
    if lead.strip():
        raise ValueError(f"{EXEMPLAR_PATH} does not open with a role marker")
    out = []
    for role, body in zip(parts[::2], parts[1::2]):
        if role == "ASSISTANT":
            _reasoning, fence, code = body.partition("```python")
            if not fence:
                raise ValueError(f"an ASSISTANT turn of {EXEMPLAR_PATH} has no python block")
            body = f"> exec\n{fence}{code}"
        elif body.startswith("Output:\n"):
            body = "output:" + body[len("Output:"):]
        elif out:
            # With the markers gone, the authors' text after the example would read as its output.
            body = "The example ends here.\n\n" + body
        out.append(body)
    return "".join(out)


def authors_instructions() -> tuple[str, str]:
    """The shipped prompt's intro and its Key instructions, the text around the worked example."""
    text = EXEMPLAR_PATH.read_text(encoding="utf-8").lstrip()
    _lead, *parts = re.split(r"^(USER|ASSISTANT):\n", text, flags=re.M)
    users = [body for role, body in zip(parts[::2], parts[1::2])
             if role == "USER" and not body.startswith("Output:\n")]
    intro, sep, _example_task = users[0].partition("\n\nMy name is:")
    keys = [body for body in users if body.startswith("**Key instructions**")]
    if not sep or len(keys) != 1:
        raise ValueError(f"{EXEMPLAR_PATH} lacks the intro's example task or one Key instructions turn")
    return intro.strip(), keys[0].strip()


def exemplar_template(source_check: bool = False, version: int = 1) -> str:
    """The shipped instruction + worked example as `exec` calls, task block cut, slots unfilled."""
    text = EXEMPLAR_PATH.read_text(encoding="utf-8").lstrip()
    head, sep, _task = text.rpartition(_TASK_MARKER)
    if not sep:
        raise ValueError(f"{EXEMPLAR_PATH} lacks the task block marker")
    head = _as_exec_transcript(head)
    if not source_check:
        return head
    head = replace_each(head, SOURCE_CHECK_EDITS)
    if version >= 2:
        head = replace_each(head, SOURCE_CHECK_V2_EDITS)
    return replace_each(head, SOURCE_CHECK_V3_EDITS) if version == 3 else head


def exemplar_text(supervisor: dict[str, str], app_descriptions: str, source_check: bool = False,
                  version: int = 1) -> str:
    """The exemplar template with the supervisor and the app descriptions filled in."""
    return Template(exemplar_template(source_check, version)).render(
        main_user=supervisor, app_descriptions=app_descriptions,
    ).rstrip()


def served_outputs(log: AppWorldPhaseLog, outputs: list[str]) -> list[str | None]:
    """Per tool call, the server's whole output of an executed `exec`, else None."""
    # An `exec` the server ran was answered before the episode ended; when the counts disagree, Pi's copies stand.
    fcs = [t for t in log.turns if t.get("type") == "function_call"]
    executed = [
        t.get("name") == ENV_TOOL and p["result"] is not None
        and not p["result"].get("is_error") and p["result"].get("text") != EPISODE_OVER
        for t, p in zip(fcs, align_calls(fcs, log.final_context))
    ]
    if sum(executed) != len(outputs):
        return [None] * len(executed)
    it = iter(outputs)
    return [next(it) if e else None for e in executed]


def tool_steps_from_log(log: AppWorldPhaseLog, outputs: Sequence[str] = ()) -> list[ToolStep]:
    # An `exec` call's command IS the code it ran; the other tools keep the shared spelling.
    """Every tool call of the session, paired with its reasoning and result by call id."""
    fcs = [t for t in log.turns if t.get("type") == "function_call"]
    unresolved = _unresolved_reason(log)
    steps: list[ToolStep] = []
    for t, p, served in zip(fcs, align_calls(fcs, log.final_context), served_outputs(log, list(outputs))):
        args = t.get("arguments") or {}
        res = p["result"]
        command = (
            str(args.get("code", "")) if t.get("name") == ENV_TOOL and isinstance(args, dict)
            else call_text(t.get("name"), args)
        )
        # A result Pi size-capped stays capped; the server's output fills in only where Pi's copy is missing.
        capped = (res or {}).get("state") == "truncated"
        steps.append(ToolStep(
            reasoning=p["reasoning"], command=command,
            result=served if served is not None and not capped else (res or {}).get("text", ""),
            is_error=bool((res or {}).get("is_error")),
            state=(res or {}).get("state", ""), unresolved="" if res is not None else unresolved,
        ))
    return steps


def _phase_log(res: TerminalResult, system: str, user: str) -> AppWorldPhaseLog:
    return AppWorldPhaseLog(
        phase="solver", turns=res.turns, read_set=res.read_set, final_text=res.final_text,
        final_context=res.final_context, final_reasoning=res.final_reasoning,
        system_prompt=system, user_prompt=user,
        usage=dict(res.usage or {}), usage_calls=list(res.usage_calls or []),
        stopped_by=res.stopped_by, last_stop_reason=getattr(res, "last_stop_reason", ""),
        last_had_tool_call=bool(getattr(res, "last_had_tool_call", False)),
        truncated_calls=int(getattr(res, "truncated_calls", 0) or 0),
        provider_error=res.provider_error,
    )


def _run_episode(
    logs: list[AppWorldPhaseLog], sample: AppWorldSample, runner: TraverseRunner,
    corpus_dir: Path | None, session_root: Path, *, memory_text: str = "", user_prefix: str = "",
    tools: str = "read,bash", max_turns: int = 50, sentinel_name: str = "__out.json",
    memory: bool = False, memory_layers: tuple[str, ...] = (), descend_block: str = "",
    source_check: bool = False, source_check_version: int = 1,
    readonly_files: Mapping[str, Path] | None = None, **_ignored: Any,
) -> TraverseOutcome:
    """Run one episode; only the memory in the prompts and the mount at /work/corpus vary by method."""
    cite_layers = memory_layers
    sdir = Path(session_root) / sample.id / "solver"
    sdir.mkdir(parents=True, exist_ok=True)
    with AppWorldSession(Path(sample.root), sample.id, sdir) as env:
        observation, info = env.reset()
        supervisor = dict(info.get("supervisor") or sample.supervisor)
        world_datetime = str(info.get("world_datetime") or sample.world_datetime)
        system = render_system(
            "appworld_solver", "traverse",
            exemplar=exemplar_text(supervisor, str(info.get("app_descriptions", "")), source_check,
                                  source_check_version),
            memory=memory, descend_block=descend_block,
        )
        user = render_user(
            "appworld_solver", "traverse", instruction=sample.question,
            main_user=supervisor, observation=observation, past_runs=memory_text,
        )
        if user_prefix.strip("\n"):
            user = f"{user_prefix.strip(chr(10))}\n\n{user}"
        res = runner.run_session(
            system_prompt=system, user_msg=user, corpus_dir=corpus_dir, out_dir=sdir,
            tools=tools, max_turns=max_turns, sentinel_name=sentinel_name,
            label="solver", env_tool=EnvTool("exec", env.socket_dir),
            citation_prompt=citation_message(cite_layers) if cite_layers else None,
            citation_layers=cite_layers, readonly_files=readonly_files,
        )
        log = _phase_log(res, system, user)
        logs.append(log)
        episode = env.finish(close_reason(res.stopped_by or res.provider_error))
        outputs = [
            str(r.get("observation", "")) for r in episode.get("trajectory") or []
            if r.get("role") == "environment" and int(r.get("step", 0)) > 0
        ]
    evaluation = dict(episode.get("evaluation") or {})
    tests = dict(evaluation.get("tests") or {})
    error = (
        str(episode.get("env_error", ""))
        or (f"provider_error: {res.provider_error}" if res.provider_error else "")
        or ("" if tests else "episode ended without an evaluation")
    )
    tgc = bool(tests) and bool(evaluation.get("success", False))
    result = AppWorldResult(
        sample_id=sample.id, db_id=sample.db_id, question=sample.question,
        scenario_id=sample.scenario_id, difficulty=sample.difficulty,
        supervisor=supervisor, world_datetime=world_datetime,
        tgc=tgc,
        tgc_lenient=lenient_tgc(
            tgc, list(tests.get("failures") or []), gold_answer_is_null(sample.root, sample.id),
        ) if sample.lenient_answer else None,
        tests_passed=int(tests.get("passed", 0)), tests_total=int(tests.get("total", 0)),
        steps=int(episode.get("steps", 0)),
        step_limit_hit=bool(episode.get("step_limit_hit", False)),
        invalid_commands=int(episode.get("invalid_commands", 0)),
        ended_by=str(episode.get("ended_by", "")), outputs=outputs,
        report=tests,
        generation_error=error,
        citations=[CitationRecord(
            sample_id=sample.id, phase=Phase.SOLVER,
            cited=res.citation or {L: [] for L in cite_layers},
            read=read_set_by_layer(res.read_set, cite_layers),
        )],
        tool_steps=tool_steps_from_log(log, outputs),
    )
    return TraverseOutcome(result=result, phase_logs=logs, generation_failed=bool(error))


def run_appworld_episode(sample: AppWorldSample, *args: Any, **kwargs: Any) -> TraverseOutcome:
    """Run one episode, and never let a raise discard the session log already collected."""
    logs: list[AppWorldPhaseLog] = []
    try:
        return _run_episode(logs, sample, *args, **kwargs)
    except BaseException as exc:
        try:
            exc.traverse_phase_logs = logs
        except Exception:
            pass
        raise
