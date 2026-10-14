"""Insight documents, their index, and the inlined and recommended renderings."""

from __future__ import annotations

from dataclasses import dataclass

from wikitrace.harness.prompts import render_template
from wikitrace.memory.store import _doc_template

CORPUS = "/work/corpus"


@dataclass(frozen=True)
class InsightItem:
    """One insight, with the train run it was drawn from."""

    id: str  # insight-NNNN
    title: str
    sections: tuple[tuple[str, str], ...]  # (heading, text) in document order
    source_step: str  # step-NNNN
    outcome: str  # success | failure
    task: str
    task_type: str  # empty on AppWorld
    apps: tuple[str, ...] = ()  # AppWorld only


@dataclass(frozen=True)
class Episode:
    """One train run as the insight layer names it."""

    step: str
    sample_id: str
    outcome: str
    task: str
    task_type: str
    apps: tuple[str, ...] = ()


def task_types_differ(items: list[InsightItem]) -> bool:
    return len({i.task_type for i in items}) > 1


def _tag_lines(item: InsightItem, task_type: bool) -> list[str]:
    # AppWorld has no task type; its tag is the apps the source run called.
    if item.task_type:
        return [f"task_type: {item.task_type}"] if task_type else []
    return [f"apps: [{', '.join(item.apps)}]"]


def render_insight_document(item: InsightItem, *, task_type: bool) -> str:
    """The text of ``insight/insight-NNNN.md``."""
    lines = ["---", "type: insight", f"id: {item.id}", f"title: {item.title}",
             *_tag_lines(item, task_type), f"source_step: {item.source_step}", "---", ""]
    for heading, text in item.sections:
        lines += [f"## {heading}", text.strip(), ""]
    lines += ["## Source episode", f"{item.source_step} ({item.outcome})", f"- Task: {item.task}",
              f"- Compilation: {CORPUS}/compilation/{item.source_step}.md",
              f"- Raw trajectory: {CORPUS}/raw/{item.source_step}/"]
    return "\n".join(lines) + "\n"


def _title_line(item: InsightItem) -> str:
    return f"- {item.id}: {item.title}"


_SECTION_HINTS = {"Insight": "the lesson", "Applicability": "when it applies"}
_SECTION_LINES = {
    "task_type": "Sections group the insights by the task type of the run each came from.",
    "apps": "Sections group the insights by the apps the run each came from called; a section lists every app "
            "of that run.",
}


def _index_opening(items: list[InsightItem], section_line: str) -> str:
    """The index's opening: the shape of an insight document and the entry format."""
    typed = task_types_differ(items)
    fm = ["type: insight", "id: insight-NNNN", "title: <title>",
          *(["task_type: <task type>"] if typed else []),
          *(["apps: [<app>, ...]"] if any(not i.task_type for i in items) else []),
          "source_step: step-NNNN"]
    headings = dict.fromkeys(h for i in items for h, _ in i.sections)
    episode = ("step-NNNN (success | failure)\n  - Task: <the source run's instruction>"
               "\n  - Compilation: /work/corpus/compilation/step-NNNN.md"
               "\n  - Raw trajectory: /work/corpus/raw/step-NNNN/")
    sections = [*((f"## {h}", _SECTION_HINTS[h]) for h in headings), ("## Source episode", episode)]
    entry = "Each entry: - insight-NNNN: <title>"
    if typed:
        entry += ", under a `## <task_type>` section per task type, sorted by name"
    return (
        "# Insight documents — read an insight's full document before applying it.\n"
        + (f"{section_line}\n" if section_line else "") + "\n"
        + _doc_template(f"{CORPUS}/insight/insight-NNNN.md", fm, sections) + "\n"
        + entry + "\n\n"
    )


def render_insight_index(items: list[InsightItem]) -> str:
    """``insight/_index.md``: the opening, then a section per app set, per task type, or one list."""
    by_apps = not any(i.task_type for i in items)
    typed = task_types_differ(items)
    opening = _index_opening(items, _SECTION_LINES["apps"] if by_apps else
                             _SECTION_LINES["task_type"] if typed else "")
    if by_apps:
        ordered = sorted(items, key=lambda i: int(i.id.split("-")[1]))
        return opening + "\n\n".join(
            "\n".join([f"## {', '.join(apps)}", *(_title_line(i) for i in ordered if i.apps == apps)])
            for apps in sorted({i.apps for i in items})) + "\n"
    if not typed:
        return opening + "\n".join(_title_line(i) for i in items) + "\n"
    blocks = []
    for task_type in sorted({i.task_type for i in items}):
        lines = [_title_line(i) for i in items if i.task_type == task_type]
        blocks.append("\n".join([f"## {task_type}", *lines]))
    return opening + "\n\n".join(blocks) + "\n"


def render_recommendation(steps: list[str], items: list[InsightItem], episodes: dict[str, Episode]) -> str:
    """The task message's block: the recommended train runs, each with the insights drawn from it."""
    parts = ["The past train runs most similar to this task, with the insights drawn from each. The\n"
             "insights themselves are at the start of this message."]
    for step in steps:
        ep = episodes[step]
        lines = [f"### {step} ({ep.outcome})", f"Task: {ep.task}", "Insights from this run"]
        lines += [_title_line(i) for i in items if i.source_step == step]
        parts.append("\n".join(lines))
    return "<similar_past_runs>\n" + "\n\n".join(parts) + "\n</similar_past_runs>"


def _inject_item(item: InsightItem, task_type: bool, source: bool) -> str:
    if item.task_type:
        tag = [f"- **Task type:** {item.task_type}"] if task_type else []
    else:
        tag = [f"- **Apps:** {', '.join(item.apps)}"]
    lines = [f'<insight id="{item.id}">', f"- **Title:** {item.title}", *tag]
    lines += [f"- **{heading}:** {text.strip()}" for heading, text in item.sections]
    if source:
        lines.append(f"- **Source:** {item.source_step} ({item.outcome})")
    lines.append("</insight>")
    return "\n".join(lines)


def render_inject(items: list[InsightItem], *, source: bool) -> str:
    """Every insight inlined; ``source`` names each one's run, as WikiTrace does."""
    ordered = sorted(items, key=lambda i: (i.task_type, i.source_step, i.id))
    typed = task_types_differ(items)
    return "\n\n".join(_inject_item(i, typed, source) for i in ordered)


def _wrap(text: str, width: int = 92) -> str:
    """Break each unindented line past ``width``, carrying its tail into the line it continues."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if len(line) <= width or line.startswith(" "):
            continue
        cut = line.rfind(" ", 0, width + 1)
        lines[i], tail = line[:cut], line[cut + 1:]
        indent = "  " if line.startswith("- ") else ""
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        follows = nxt.startswith("  ") and not nxt.startswith("   ") if indent else (
            bool(nxt) and not nxt.startswith((" ", "- ")))
        if follows:
            lines[i + 1] = f"{indent}{tail} {nxt.lstrip()}"
        else:
            lines.insert(i + 1, f"{indent}{tail}")
    return "\n".join(lines)


def render_descend(benchmark: str) -> str:
    """WikiTrace's system memory block for one benchmark, wrapped to 92 columns."""
    return _wrap(render_template("memory/prompts/descend.j2", benchmark=benchmark))
