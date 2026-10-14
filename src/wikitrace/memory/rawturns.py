#!/usr/bin/env python3
"""Read a past run in /work/corpus/raw by the `turn` numbers of its log, standard library only.
Installed as tools/actions (every turn's action, one line each) and tools/turn (chosen turns
whole); the program name picks which."""
import ast
import json
import re
import signal
import sys
from pathlib import Path

RAW = Path("/work/corpus/raw")
# Pi keeps only the last 20K characters of a tool's output, so a longer answer loses its start.
LIMIT = 18000


def fail(message):
    """End with one line on stdout and exit status 1."""
    print(message, flush=True)
    sys.exit(1)


def flat(text):
    return " ".join(str(text or "").split())


def arguments(raw):
    """A call's arguments as logged, parsed when they are JSON or a Python literal."""
    if not isinstance(raw, str):
        return raw
    for parse in (json.loads, ast.literal_eval):
        try:
            return parse(raw)
        except Exception:  # noqa: BLE001 - a log line that parses neither way is shown as written
            pass
    return raw


def action_text(name, args):
    """A call's action: the game command, the code, or the tool's arguments."""
    if not isinstance(args, dict):
        return str(args)
    if name == "take_action":
        arg = args.get("argument", "")
        if isinstance(arg, str):
            return f"{args.get('action', '')}: {arg}"
        return f"{args.get('action', '')} {json.dumps(arg, ensure_ascii=False)}"
    if name in ("act", "exec"):
        return str(args.get("command" if name == "act" else "code", ""))
    return str(args.get("command") or args.get("path") or json.dumps(args, ensure_ascii=False))


def load(step):
    """The run's metadata, its task and its message and call rows by turn number."""
    if not re.fullmatch(r"step-\d{4}", step):
        fail(f"give the run as step-NNNN, for example step-0005 (got {step!r})")
    path = RAW / step / "solver.full.jsonl"
    if not path.exists():
        fail(f"no run {step} under {RAW}")
    meta_path = RAW / step / "metadata.md"
    head = dict(re.findall(r"^- (\w+): (.*)$", meta_path.read_text(encoding="utf-8"), re.M)) \
        if meta_path.exists() else {}
    rows = []
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                fail(f"cannot read line {n} of {step}/solver.full.jsonl")
            if r.get("type") in ("message", "function_call") and str(r.get("turn", "")).isdigit():
                rows.append((int(r["turn"]), r))
    task = head.get("user_request") or head.get("instruction")
    if task is None:
        task = next((r.get("content", "") for _, r in rows if r.get("role") == "user"), "")
    return head, flat(task), rows


def header(step, head, task, rows):
    span = f"turns {rows[0][0]}..{rows[-1][0]}" if rows else "no turns"
    return (f"{step} | outcome: {'success' if head.get('success') == 'True' else 'failure'} | {span}\n"
            f"task: {task}")


def actions_main(argv):
    if len(argv) != 1 or argv[0] in ("-h", "--help"):
        fail("usage: actions step-NNNN  (every turn's action, one line each)")
    head, task, rows = load(argv[0])
    print(header(argv[0], head, task, rows))
    for n, r in rows:
        if r.get("type") == "function_call":
            name = r.get("name") or "call"
            print(f"[{n}] {name}: {flat(action_text(name, arguments(r.get('arguments'))))}")
        elif n >= 2:  # turns 0 and 1 are the system prompt and the task
            print(f"[{n}] {'message' if r.get('role') == 'assistant' else r.get('role')}: {flat(r.get('content'))}")


def block(n, r):
    """One turn in full, labelled."""
    reason = str(r.get("reasoning") or "").strip()
    call = r.get("type") == "function_call"
    parts = [f"[turn {n}] {r.get('name') or 'call'}" if call else f"[turn {n}] {r.get('role')} message"]
    if reason:
        parts.append(f"reasoning:\n{reason}")
    if call:
        name = r.get("name") or "call"
        parts.append(f"action: {name}\n{action_text(name, arguments(r.get('arguments')))}")
        parts.append(f"observation:\n{str(r.get('result') or '(no output)')}")
    else:
        parts.append(f"{'message' if r.get('role') == 'assistant' else 'content'}:\n{r.get('content') or ''}")
    return "\n".join(parts) + "\n"


def turn_main(argv):
    if len(argv) not in (2, 3) or not all(re.fullmatch(r"\d+", a) for a in argv[1:]):
        fail("usage: turn step-NNNN FROM [TO]  (turns FROM to TO in full, as numbered in solver.full.jsonl)")
    head, task, rows = load(argv[0])
    lo, hi = int(argv[1]), int(argv[2] if len(argv) == 3 else argv[1])
    span = f"this run has turns {rows[0][0]}..{rows[-1][0]}." if rows else "this run has no turns."
    if lo > hi:
        fail(f"FROM {lo} is after TO {hi}; give FROM <= TO. {span[0].upper()}{span[1:]}")
    chosen = [(n, r) for n, r in rows if lo <= n <= hi]
    if not chosen:
        fail(f"No turn {lo if lo == hi else f'in {lo}..{hi}'}; {span}")
    out = header(argv[0], head, task, rows) + "\n"
    for n, r in chosen:
        text = "\n" + block(n, r)
        if len(out) + len(text) > LIMIT:
            print(out, end="")
            print(f"\nturns {n}..{chosen[-1][0]} left out: the output stops at {LIMIT:,} characters; "
                  f"read them with a narrower range.")
            return
        out += text
    print(out, end="")


def main(argv=None):
    argv = sys.argv if argv is None else argv
    {"actions": actions_main}.get(Path(argv[0]).name, turn_main)(argv[1:])


if __name__ == "__main__":
    # A reader that stops early, such as `| head`, ends the script quietly.
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    main()
