"""tau3 retail: the authors' environment, user simulator and evaluator behind the Environment protocol."""
# Runs under data/tau3/.venv/bin/python: standard library plus the authors' package only.

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from copy import deepcopy
from typing import Any

DOMAIN = "retail"
MAX_STEPS = 200
MAX_ERRORS = 10
JUDGE_ATTEMPTS = 3
STOP_TOKENS = ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###")
PARSE_ERROR = "Error: take_action expects a JSON object with `action` and `argument`."

_JSON_TYPE_NAMES = {str: "string", bool: "boolean", int: "integer", float: "number",
                    list: "array", dict: "object", type(None): "null"}


def parse_command(command: str) -> tuple[str, Any] | str:
    """The `(action, argument)` of one step string, or our one-line error."""
    try:
        data = json.loads(command)
    except ValueError:
        return PARSE_ERROR
    if not isinstance(data, dict) or not isinstance(data.get("action"), str) or "argument" not in data:
        return PARSE_ERROR
    return data["action"], data["argument"]


def _json_type(value: Any) -> str:
    return _JSON_TYPE_NAMES.get(type(value), type(value).__name__)


def _schema_type(schema: dict[str, Any]) -> str:
    """A parameter schema's type as the error names it: `string`, `array of string`, `string or null`."""
    if "anyOf" in schema:
        return " or ".join(_schema_type(s) for s in schema["anyOf"])
    t = schema.get("type", "value")
    if t == "array":
        return f"array of {_schema_type(schema.get('items', {}))}"
    return str(t)


def _accepts(schema: dict[str, Any], value: Any) -> bool:
    """Whether a JSON value has the type a parameter schema states (`anyOf`, `items` included)."""
    if "anyOf" in schema:
        return any(_accepts(s, value) for s in schema["anyOf"])
    t = schema.get("type")
    if t is None:
        return True
    if t == "boolean":
        return isinstance(value, bool)
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "string":
        return isinstance(value, str)
    if t == "null":
        return value is None
    if t == "array":
        return isinstance(value, list) and all(_accepts(schema.get("items", {}), v) for v in value)
    if t == "object":
        return isinstance(value, dict)
    return True


def check_argument(action: str, argument: Any, parameters: dict[str, Any] | None) -> str:
    """Our own rejection of a call, or empty when the authors' code takes it (and names its own errors)."""
    # ``parameters`` is None for an unknown tool (`respond` included), which the authors answer with `Error: Tool 'x' not found.`.
    if parameters is None:
        return ""
    if not isinstance(argument, dict):
        return f"Error: `{action}` takes its parameters as a JSON object, got {_json_type(argument)}."
    for key, value in argument.items():
        schema = (parameters or {}).get("properties", {}).get(key)
        if schema is not None and not _accepts(schema, value):
            return (f"Error: `{action}` parameter `{key}` must be {_schema_type(schema)}, "
                    f"got {_json_type(value)}.")
    return ""


def _stop_token(text: str | None) -> str:
    return next((t for t in STOP_TOKENS if t in (text or "")), "")


def call_record(role: str, model: str, message: Any) -> dict[str, Any]:
    """One simulator or judge call as an `llm_calls.jsonl` row: tokens, cost, generation id and provider."""
    # litellm keeps OpenRouter's `provider` and `usage.cost` on the raw response; a scripted reply records zeros.
    raw = message.raw_data if isinstance(message.raw_data, dict) else {}
    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else dict(message.usage or {})
    choices = raw.get("choices") or [{}]
    return {
        "role": role, "model_requested": model, "model_returned": str(raw.get("model", "")),
        "generation_id": str(raw.get("id", "")), "provider": str(raw.get("provider", "")),
        "finish_reason": str((choices[0] or {}).get("finish_reason", "")),
        "latency": float(message.generation_time_seconds or 0.0),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "reasoning_tokens": int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
        "cached_tokens": int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
        "cost": float(usage.get("cost") if usage.get("cost") is not None else (message.cost or 0.0)),
    }


class Judge:
    """The NL-assertion judge: its model settings, per-call accounting and the JSON-parse retry."""
    # The authors' evaluator reads the judge model from module globals and parses the reply
    # with a bare `json.loads`; both are reached by rebinding the module, never by editing it.

    def __init__(self, model: str, llm_args: dict[str, Any], script: list[str] | None = None) -> None:
        self.model = model
        self.llm_args = dict(llm_args)
        self.script = list(script or [])
        self.calls = 0
        self.cost = 0.0
        self.attempts = 0
        self.failed = False
        self.error = ""
        self.records: list[dict[str, Any]] = []

    def install(self) -> None:
        import tau2.evaluator.evaluator_nl_assertions as nl

        global _GENERATE, _EVALUATE
        if _GENERATE is None:
            _GENERATE = nl.generate
            _EVALUATE = nl.NLAssertionsEvaluator.evaluate_nl_assertions.__func__
        nl.DEFAULT_LLM_NL_ASSERTIONS = self.model
        nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = dict(self.llm_args)
        nl.generate = self._generate
        nl.NLAssertionsEvaluator.evaluate_nl_assertions = classmethod(self._evaluate)

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        from tau2.data_model.message import AssistantMessage

        if self.script:
            message = AssistantMessage(role="assistant", content=self.script.pop(0), cost=0.0)
        else:
            message = _GENERATE(*args, **kwargs)
        self.calls += 1
        self.cost += message.cost or 0.0
        self.records.append(call_record("judge", self.model, message))
        return message

    def _evaluate(self, cls: Any, trajectory: Any, nl_assertions: list[str]) -> list[Any]:
        from tau2.data_model.simulation import NLAssertionCheck

        for attempt in range(1, JUDGE_ATTEMPTS + 1):
            self.attempts = attempt
            try:
                checks = _EVALUATE(cls, trajectory, nl_assertions)
            except Exception as exc:  # noqa: BLE001
                self.error = f"{type(exc).__name__}: {exc}"
                continue
            self.error = ""
            return checks
        self.failed = True
        return [NLAssertionCheck(nl_assertion=a, met=False, justification="") for a in nl_assertions]


_GENERATE: Any = None
_EVALUATE: Any = None


def scripted_user(script: list[str]) -> Any:
    """A user simulator that replays canned replies, for a run with no model call."""
    from tau2.data_model.message import UserMessage
    from tau2.user.user_simulator import UserSimulator

    class ScriptedUser(UserSimulator):
        def __init__(self) -> None:
            super().__init__(llm="scripted")
            self.script = list(script)

        def _generate_next_message(self, message: Any, state: Any) -> Any:
            state.messages.append(message)
            return UserMessage(role="user", content=self.script.pop(0), cost=0.0)

    return ScriptedUser()


class Tau3Environment:
    """One retail task: the authors' Environment and UserSimulator, scored by their evaluator at close."""

    def __init__(self, task: Any, retail: Any, user: Any, judge: Judge) -> None:
        self.task = task
        self.retail = retail
        self.user = user
        self.judge = judge
        self.parameters = {t.name: t.openai_schema["function"]["parameters"] for t in retail.get_tools()}
        self.user_state: Any = None
        self.messages: list[Any] = []
        self.step_count = 0
        self.num_errors = 0
        self.messages_sent = 0
        self.invalid_commands = 0
        self.stop_token = ""
        self.termination: Any = None
        self.error = ""
        self.user_calls = 0
        self.user_cost = 0.0
        self.llm_calls: list[dict[str, Any]] = []
        self.started_at = ""
        self.started_perf = 0.0


    def _check_termination(self) -> None:
        """The authors' `_check_termination`, run wherever their loop runs it."""
        from tau2.data_model.simulation import TerminationReason

        if self.step_count >= MAX_STEPS:
            self.termination = TerminationReason.MAX_STEPS
        if self.num_errors >= MAX_ERRORS:
            self.termination = TerminationReason.TOO_MANY_ERRORS

    def _info(self, source: str = "", invalid: bool = False) -> dict[str, Any]:
        from tau2.data_model.simulation import TerminationReason

        return {
            "success": False, "score": 0.0,
            "failed": self.termination == TerminationReason.TOO_MANY_ERRORS,
            "step_limit_hit": self.termination == TerminationReason.MAX_STEPS,
            "invalid_command": invalid, "source": source,
            "orchestrator_steps": self.step_count, "messages": self.messages_sent,
            "num_errors": self.num_errors, "stop_token": self.stop_token,
            "termination_reason": self.termination.value if self.termination else "",
        }

    @property
    def done(self) -> bool:
        return self.termination is not None


    def _user_turn(self, message: Any) -> Any:
        """Send the agent's message to the user; the reply is one of the authors' steps."""
        from tau2.data_model.simulation import TerminationReason
        from tau2.user.user_simulator import UserSimulator

        try:
            reply, self.user_state = self.user.generate_next_message(message, self.user_state)
            self.user_calls += 1
            self.user_cost += reply.cost or 0.0
            self.llm_calls.append(call_record("user_simulator", str(getattr(self.user, "llm", "")), reply))
            reply.validate()
        except Exception as exc:
            self.error = f"user_simulator: {type(exc).__name__}: {exc}"
            raise
        self.step_count += 1
        if UserSimulator.is_stop(reply):
            self.stop_token = _stop_token(reply.content)
            self.termination = TerminationReason.USER_STOP
        self.messages.append(reply)
        self._check_termination()
        return reply

    def reset(self) -> tuple[str, dict[str, Any]]:
        from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
        from tau2.utils.utils import get_now

        self.started_at = get_now()
        self.started_perf = time.perf_counter()
        try:
            initial = self.task.initial_state
            self.retail.set_state(
                initialization_data=initial.initialization_data if initial else None,
                initialization_actions=initial.initialization_actions if initial else None,
                message_history=[],
            )
            self.user_state = self.user.get_init_state()
            # The authors' fixed greeting counts as sent; the user's answer to it is the task.
            greeting = deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)
            greeting.timestamp = get_now()
            self.messages = [greeting]
            first = self._user_turn(greeting).content or ""
        except Exception as exc:
            self.error = self.error or f"reset: {type(exc).__name__}: {exc}"
            raise
        return first, {**self._info(source="user"), "task": first}

    def _rejected(self, error: str) -> tuple[str, bool, dict[str, Any]]:
        """Our own rejection: an error round-trip in the authors' count, outside their message list."""
        self.step_count += 2
        self.num_errors += 1
        self.invalid_commands += 1
        self._check_termination()
        return error, self.done, self._info(source="env", invalid=True)

    def step(self, command: str) -> tuple[str, bool, dict[str, Any]]:
        from tau2.data_model.message import AssistantMessage, ToolCall

        if self.done:
            return "", True, self._info()
        parsed = parse_command(command)
        if isinstance(parsed, str):
            return self._rejected(parsed)
        action, argument = parsed
        if error := check_argument(action, argument, self.parameters.get(action)):
            return self._rejected(error)
        self.step_count += 1
        # An unknown tool's argument never reaches a function, so a non-object one is dropped.
        call = ToolCall(id=f"call_{len(self.messages)}", name=action,
                        arguments=argument if isinstance(argument, dict) else {})
        self.messages.append(AssistantMessage(role="assistant", tool_calls=[call], cost=0.0))
        result = self.retail.get_response(call)
        if result.error:
            self.num_errors += 1
        self.messages.append(result)
        self.step_count += 1
        self._check_termination()
        return result.content or "", self.done, self._info(source="env")

    def message(self, text: str) -> tuple[str, bool, dict[str, Any]]:
        """The agent's message to the user; the observation is the user's reply."""
        from tau2.data_model.message import AssistantMessage

        if self.done:
            return "", True, self._info()
        self.step_count += 1
        self.messages_sent += 1
        message = AssistantMessage(role="assistant", content=text, cost=0.0)
        self.messages.append(message)
        self._check_termination()
        if self.done:
            return "", True, self._info(source="user")
        reply = self._user_turn(message)
        return reply.content or "", self.done, self._info(source="user")

    def close(self) -> None:
        pass

    def _reward(self) -> Any:
        """The authors' reward over the message list, judge included."""
        from tau2.data_model.simulation import SimulationRun
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
        from tau2.utils.utils import get_now

        messages = deepcopy(self.messages)
        for i, m in enumerate(messages):
            m.turn_idx = i
        run = SimulationRun(
            id=str(uuid.uuid4()), task_id=self.task.id, start_time=self.started_at, end_time=get_now(),
            duration=time.perf_counter() - self.started_perf, termination_reason=self.termination,
            messages=messages,
        )
        self.judge.install()
        return evaluate_simulation(run, self.task, EvaluationType.ALL, solo_mode=False, domain=DOMAIN)

    def evaluation(self, ended_by: str = "") -> dict[str, Any]:
        """The authors' reward for an episode the user ended; every other end scores 0, and a failed reset or simulator call is not scored."""
        # As the authors' `evaluate_simulation`: budget exhaustion and ends from outside the environment give 0, no judge call.
        from tau2.data_model.simulation import TerminationReason

        scored = not self.error and self.termination == TerminationReason.USER_STOP
        reward = self._reward() if scored else None
        checks = None if reward is None or self.judge.failed or reward.nl_assertions is None else [
            c.model_dump() for c in reward.nl_assertions]
        out: dict[str, Any] = {
            "success": reward is not None and reward.reward == 1.0,
            "score": reward.reward if reward is not None else 0.0,
            "scored": scored,
            "termination_reason": self.termination.value if self.termination else ended_by,
            "stop_token": self.stop_token,
            "orchestrator_steps": self.step_count,
            "messages": self.messages_sent,
            "num_errors": self.num_errors,
            "reward": reward.reward if reward is not None else (None if self.error else 0.0),
            "reward_basis": [b.value for b in reward.reward_basis] if reward and reward.reward_basis else None,
            "reward_breakdown": ({k.value: v for k, v in reward.reward_breakdown.items()}
                                 if reward and reward.reward_breakdown else None),
            "db_check": reward.db_check.model_dump() if reward and reward.db_check else None,
            "nl_assertions": checks,
            "judge_attempts": self.judge.attempts,
            "judge_failed": self.judge.failed,
            "judge_error": self.judge.error,
            "calls": {"user_simulator": self.user_calls, "judge": self.judge.calls},
            "cost": {"user_simulator": self.user_cost, "judge": self.judge.cost},
            "llm_calls": [*self.llm_calls, *self.judge.records],
        }
        if self.error:
            out["generation_error"] = self.error
        return out


def quiet_authors_log() -> None:
    """Keep the authors' loguru output at WARNING; its DEBUG default echoes every turn into the env log."""
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")


def load_task(task_id: str) -> Any:
    from tau2.registry import registry

    task = next((t for t in registry.get_tasks_loader(DOMAIN)(None) if t.id == task_id), None)
    if task is None:
        raise KeyError(f"no {DOMAIN} task {task_id!r}")
    return task


def make_env(spec: dict[str, Any]) -> Tau3Environment:
    """``spec``: ``task_id``, the simulator's and judge's ``model`` and ``llm_args``, or offline scripts."""
    quiet_authors_log()
    from tau2.registry import registry
    from tau2.user.user_simulator import UserSimulator

    task = load_task(str(spec["task_id"]))
    if spec.get("user_script") is not None:
        user = scripted_user(list(spec["user_script"]))
    else:
        sim = UserSimulator
        if spec.get("user_guidelines_path"):
            guidelines = Path(str(spec["user_guidelines_path"])).read_text(encoding="utf-8")

            class sim(UserSimulator):  # noqa: N801
                global_simulation_guidelines = guidelines

        user = sim(llm=str(spec["user_model"]), instructions=str(task.user_scenario),
                   llm_args=dict(spec.get("user_llm_args") or {}))
    judge = Judge(str(spec.get("judge_model", "")), dict(spec.get("judge_llm_args") or {}),
                  spec.get("judge_script"))
    return Tau3Environment(task, registry.get_env_constructor(DOMAIN)(), user, judge)
