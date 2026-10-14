"""The compiler: one train run into a compilation and its insights, and the corpus read back as insights."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wikitrace.harness.llm.client import LLMClient, StructuredResult
from wikitrace.harness.prompts import render_system, render_user
from wikitrace.memory.insights import CORPUS, Episode, InsightItem
from wikitrace.memory.store import render_compilation_index
from wikitrace.config import (
    Compilation,
    CompilationInputs,
    ConfidenceLevel,
    EnvCompilerFailureOutput,
    EnvCompilerFailureOutputWithConfidence,
    EnvCompilerSuccessOutput,
    Outcome,
    PromptProfile,
)


@dataclass
class CompilationResult:
    """Everything the runner needs from one compilation pass."""

    compilation: Compilation
    structured: StructuredResult


# Minimum unbroken run the compiler-facing render elides: 100 sits between the longest all-caps prose token (16) and the shortest sequence literal (875).
SEQUENCE_RUN_LIMIT = 100

# The 20 residues plus U, shared with the file-ifier so "a sequence" has one definition.
SEQUENCE_ALPHABET = "ACDEFGHIKLMNPQRSTUVWY"


# How much of a shortened literal's HEAD to keep, for the compiler marker and the solver's file preview alike.
SEQUENCE_HEAD_CHARS = 60

# No template mentions these numbers (``_elision_legend_for`` briefs at runtime), so no .j2 byte moves the fingerprint.


# B/J/O/X/Z are OMITTED and matching is uppercase-only, so prose cannot form a long run.
_NUCLEOTIDE_ALPHABET = frozenset("ACGTUN")
_SEQUENCE_ALPHABET = SEQUENCE_ALPHABET
_SEQUENCE_RUN_LIMIT = SEQUENCE_RUN_LIMIT
_SEQUENCE_HEAD_CHARS = SEQUENCE_HEAD_CHARS
# PUBLIC: the ONE definition of "a sequence literal", bound by the elision, the option render and the installer.
SEQUENCE_RUN = re.compile(f"[{_SEQUENCE_ALPHABET}]{{{_SEQUENCE_RUN_LIMIT},}}")


def sequence_kind(run: str) -> str:
    """``nt`` for a nucleotide literal, ``aa`` for a protein one."""
    return "nt" if set(run) <= _NUCLEOTIDE_ALPHABET else "aa"


def _elide_sequences(text: str) -> str:
    """Replace long biological sequence literals with a marker; keep all prose."""
    # A payload rule, not a length cap: excising only the literal keeps every surrounding word. Idempotent.

    def _mark(m: re.Match[str]) -> str:
        run = m.group()
        kind = sequence_kind(run)
        marked = f"{run[:_SEQUENCE_HEAD_CHARS]}…[{len(run)}{kind}]"
        # An "elision" that makes the text LONGER is not one; compared, not thresholded.
        return marked if len(marked) < len(run) else run

    return SEQUENCE_RUN.sub(_mark, text)


# Where an elided literal survives in full, per elision SITE; a tool step's command and output pass their own pointer.
_QUESTION_RAW_POINTER = "full question in this step's raw/ trajectory (metadata.md)"
# It names the QUESTION only; an option's full text survives as a mounted FASTA file, stated in the options header.


def _elision_legend(raw_pointer: str) -> str:
    """The one line a block prints when an elision marker is visible in it."""
    return (
        f"(sequence literals elided: first {_SEQUENCE_HEAD_CHARS} chars then "
        f"…[total length + aa=amino-acid / nt=nucleotide]; {raw_pointer})"
    )


# What `_elide_sequences` leaves behind; decides whether a FINISHED block still shows a marker.
_ELISION_MARKER = re.compile(r"…\[\d+(?:aa|nt)\]")


def _elision_legend_for(rendered: str, raw_pointer: str) -> list[str]:
    """``[legend_line]`` if ``rendered`` still SHOWS an elision marker, else ``[]``."""
    # Judged from the finished text, not a flag: the length cap can delete the very marker the legend explains.
    return [_elision_legend(raw_pointer)] if _ELISION_MARKER.search(rendered) else []


def _build_choices(ci: CompilationInputs) -> str:
    """The task's answer options for the compiler; ``""`` when it has none."""
    # ``""`` makes the template drop the block; ``ci.choices`` normally arrives processed, so the elision here is a net.
    return _elide_sequences(ci.choices.strip())


def _build_task_metadata(
    ci: CompilationInputs, evidence: str, has_evidence: bool = True
) -> str:
    """Compact question (+ evidence, when the benchmark has that channel) block."""
    # No evidence channel OMITS the line (`evidence: (none)` would assert one exists); the legend is decided by what BOTH finished blocks show.
    lines = [f"{k}: {_elide_sequences(v)}" for k, v in ci.task_fields] or [
        f"db_id: {ci.db_id}",
        f"question: {_elide_sequences(ci.question)}",
    ]
    if has_evidence:
        lines.append(f"evidence: {evidence.strip() or '(none)'}")
    body = "\n".join(lines)
    choices = _build_choices(ci)
    legend = _elision_legend_for(f"{body}\n{choices}", _QUESTION_RAW_POINTER)
    return "\n".join([*legend, body])


# Trajectory and execution-outcome blocks are DOMAIN renders supplied by the caller.


_CONFIDENCE_RANK = {
    ConfidenceLevel.HIGH: 0,
    ConfidenceLevel.MEDIUM: 1,
    ConfidenceLevel.LOW: 2,
}


def _render_uncertainties(parsed) -> list[str]:
    """The compilation's open questions, as its own section."""
    # Kept off the individual causes: a per-cause field lets the compiler grade a cause `high` while recording doubt.
    return [f"- {u}" for u in parsed.uncertainties] or ["(none)"]


def _field(label: str, value: str) -> list[str]:
    """``label: value``, or a ``label:`` header over an indented block when multi-line."""
    text = value.strip() or "(none)"
    if "\n" not in text:
        return [f"{label}: {text}"]
    return [f"{label}:", *(f"  {ln}" if ln.strip() else "" for ln in text.splitlines())]


def _render_task_summary(
    ci: CompilationInputs, evidence: str, profile: PromptProfile
) -> str:
    """The ``## Task`` block: what this trajectory was ASKED."""
    # Carries the option list but deliberately NOT the answer, which sits one hop down in ``raw/``.
    task = [f"{k}: {v}" for k, v in ci.task_fields] or [
        f"db_id: {ci.db_id}", f"question: {ci.question}"
    ]
    lines = ["## Task", *task]
    if profile.has_evidence:
        # Deliberately NOT through `_field`: a helper that blocks a multi-line value would move evidence's bytes.
        lines.append(f"evidence: {evidence.strip() or '(none)'}")
    if ci.choices:
        lines += _field("options", ci.choices)
    # The standing document, at the raw layer's preview budget (cut by the adapter); omitted, not emptied, when absent.
    if ci.task_context_preview.strip():
        lines += _field("context", ci.task_context_preview)
    return "\n".join(lines)


def _cards(heading: str, cards: list[tuple[str, list[tuple[str, str]]]]) -> str:
    """A ``## heading`` section of ``### title`` cards, one ``**field**: value`` line each."""
    blocks = [
        "\n".join([f"### {title}", *(f"**{k}**: {v}" for k, v in fields)])
        for title, fields in cards
    ]
    return f"## {heading}\n" + "\n\n".join(blocks)


def _render_env_decisions(decisions) -> str:
    """The Key Decisions section: what was chosen, then what shows it mattered."""
    return _cards("Key Decisions", [
        (f"Decision {n}", [("Decision", d.decision), ("Support", d.support)])
        for n, d in enumerate(decisions, 1)
    ])


def _render_env_failure_causes(causes) -> str:
    """Causes by descending confidence when they carry one; ties keep the compiler's order."""
    # ``sorted`` is stable, so the section is a fixed point under render -> parse -> render.
    graded = all(getattr(c, "confidence", None) is not None for c in causes)
    ordered = (
        sorted(causes, key=lambda c: _CONFIDENCE_RANK[c.confidence]) if graded else causes
    )
    return _cards("Failure Causes", [
        (f"Cause {n}", [("Decision", c.decision), ("Support", c.support)]
         + ([("Confidence", c.confidence.value)] if graded else []))
        for n, c in enumerate(ordered, 1)
    ])


def _render_env_insights(parsed) -> str:
    """The Candidate Insights section, one titled card per insight."""
    return _cards("Candidate Insights", [
        (f"Insight {n}: {it.title}",
         [("Guidelines", it.guideline), ("Applicability", it.applicability)])
        for n, it in enumerate(parsed.candidate_insights, 1)
    ])


def _render_env_success_body(
    ci: CompilationInputs, parsed, evidence: str, profile: PromptProfile,
) -> str:
    return "\n\n".join([
        "# Compilation (success)",
        _render_task_summary(ci, evidence, profile),
        _render_env_decisions(parsed.key_decisions),
        _render_env_insights(parsed),
    ])


def _render_env_failure_body(
    ci: CompilationInputs, parsed, evidence: str, profile: PromptProfile,
) -> str:
    sections = [
        "# Compilation (failure)",
        _render_task_summary(ci, evidence, profile),
        _render_env_failure_causes(parsed.failure_causes),
        _render_env_insights(parsed),
    ]
    if hasattr(parsed, "uncertainties"):
        sections.append("## Uncertainties\n" + "\n".join(_render_uncertainties(parsed)))
    return "\n\n".join(sections)


def _build_compilation(
    ci: CompilationInputs, step_n: int, parsed, body: str, evidence: str
) -> Compilation:
    """The persisted record around one compiler output, shared by both prompt families."""
    outcome = Outcome.SUCCESS if ci.success else Outcome.FAILURE
    agent_title = (getattr(parsed, "title", "") or "").strip()
    return Compilation(
        step_id=f"step-{step_n:04d}",
        outcome=outcome,
        metrics=dict(ci.metrics),
        # Carried verbatim: ``outcome`` collapses a refusal into ``failure``, so this flag alone marks an abstention.
        refused=ci.refused,
        # A harness stop is not the agent concluding wrongly, and ``outcome`` cannot say so.
        generation_failed=ci.generation_failed,
        title=f"[{outcome.value}] {agent_title}".rstrip(),
        description=(getattr(parsed, "description", "") or "").strip(),
        question=ci.question,
        evidence=evidence,
        prediction=ci.prediction,
        body=body,
    )


def compile_trajectory(
    llm: LLMClient,
    ci: CompilationInputs,
    step_n: int,
    evidence: str = "",
    *,
    trajectory_text: str,
    outcome_text: str,
    profile: PromptProfile,
    confidence: bool = False,
) -> CompilationResult:
    """Compile one no-memory train run into a Compilation and its candidate insights."""
    success = ci.success
    role = "compiler_success_nomem" if success else "compiler_failure_nomem"
    variant = "success" if success else "failure"
    if success:
        model_cls = EnvCompilerSuccessOutput
    elif confidence:
        model_cls = EnvCompilerFailureOutputWithConfidence
    else:
        model_cls = EnvCompilerFailureOutput
    user = render_user(
        "compiler", variant,
        task_metadata=_build_task_metadata(ci, evidence, profile.has_evidence),
        trajectory=trajectory_text, execution_outcome=outcome_text, memory=False,
        cited_memory_documents="None",
    )
    system = render_system("compiler", variant, memory=False, confidence=confidence, **profile.model_dump())
    structured = llm.generate_structured(
        role=role, model_cls=model_cls, system=system, user=user, phase=role, sample_id=ci.sample_id,
    )
    parsed = structured.parsed
    body = (_render_env_success_body(ci, parsed, evidence, profile) if success
            else _render_env_failure_body(ci, parsed, evidence, profile))
    return CompilationResult(compilation=_build_compilation(ci, step_n, parsed, body, evidence),
                             structured=structured)


_INSIGHTS_SECTION = re.compile(r"## Candidate Insights\n.*?(?=\n\n## |\Z)", re.DOTALL)
REBUILD = "build it with `wikitrace build`"


def require_built_corpus(corpus_dir: str | Path) -> Path:
    """``corpus_dir`` if it is a built and compiled corpus; raise otherwise."""
    corpus_dir = Path(corpus_dir)
    stamp = corpus_dir / "corpus.json"
    if not stamp.exists():
        raise FileNotFoundError(f"no built corpus at {corpus_dir}; {REBUILD}")
    if not json.loads(stamp.read_text(encoding="utf-8")).get("insight_items"):
        raise ValueError(f"{corpus_dir} holds no insights; {REBUILD}")
    return corpus_dir


def _records(corpus_dir: Path) -> list[dict[str, Any]]:
    """The compilation records of a corpus, in step order."""
    if any((corpus_dir / "_failures").glob("*")):
        raise RuntimeError(f"{corpus_dir} has failed steps in _failures/")
    path = corpus_dir / "compilation" / "compilations.jsonl"
    recs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if any("insight_items" not in r for r in recs):
        raise ValueError(f"{corpus_dir} holds no insights; {REBUILD}")
    return sorted(recs, key=lambda r: r["step_id"])


def read_episodes(dom: Any, corpus_dir: str | Path) -> dict[str, Episode]:
    """Every raw step of a corpus as an episode, keyed by step id."""
    called_apps = getattr(dom, "called_apps", None)
    episodes = {}
    for step_dir in sorted((Path(corpus_dir) / "raw").glob("step-*")):
        result = dom.read_result(step_dir)
        episodes[step_dir.name] = Episode(
            step=step_dir.name, sample_id=result.sample_id,
            outcome="success" if dom.benchmark.metrics_from_result(result).success else "failure",
            task=result.question, task_type="" if called_apps else str(result.task_type),
            apps=called_apps(result) if called_apps else (),
        )
    return episodes


def read_items(dom: Any, corpus_dir: str | Path) -> list[InsightItem]:
    """The insights, numbered in step order and then in the compiler's order."""
    corpus_dir = Path(corpus_dir)
    episodes = read_episodes(dom, corpus_dir)
    items: list[InsightItem] = []
    for rec in _records(corpus_dir):
        ep = episodes[rec["step_id"]]
        for it in rec["insight_items"]:
            items.append(InsightItem(
                id=f"insight-{len(items):04d}", title=it["title"].strip(),
                sections=(("Insight", it["guideline"]), ("Applicability", it["applicability"])),
                source_step=ep.step, outcome=ep.outcome, task=ep.task,
                task_type=ep.task_type, apps=ep.apps,
            ))
    return items


def relink_compilation(text: str, items: list[InsightItem]) -> str:
    """A step-NNNN.md with its Candidate Insights section replaced by links to its insights."""
    lines = [f"- {i.id}: {i.title} ({CORPUS}/insight/{i.id}.md)" for i in items]
    section = "\n".join(["## Insights from this run", *lines])
    new, n = _INSIGHTS_SECTION.subn(lambda _: section, text)
    if n != 1:
        raise ValueError(f"expected one Candidate Insights section, found {n}")
    return new


def relinked_compilations(corpus_dir: str | Path, items: list[InsightItem]) -> dict[str, str]:
    """Every compiled step's relinked document, keyed by step id."""
    comp = Path(corpus_dir) / "compilation"
    return {
        rec["step_id"]: relink_compilation(
            (comp / f"{rec['step_id']}.md").read_text(encoding="utf-8"),
            [i for i in items if i.source_step == rec["step_id"]],
        )
        for rec in _records(Path(corpus_dir))
    }


def compilation_index(dom: Any, corpus_dir: str | Path, raw_shape: Any) -> str:
    """The mounted ``compilation/_index.md``: each compiled step under its title."""
    corpus_dir = Path(corpus_dir)
    stamp = json.loads((corpus_dir / "corpus.json").read_text(encoding="utf-8"))
    return render_compilation_index(
        [Compilation.model_validate(rec) for rec in _records(corpus_dir)],
        entry_source=dom.compilation_entry_source, raw_shape=raw_shape,
        compiler_confidence=bool(stamp.get("compiler_confidence")),
    )
