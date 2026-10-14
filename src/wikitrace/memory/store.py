"""The on-disk corpus: raw-layer shapes, code-generated indexes and compilation documents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields, replace
from string import Formatter

from wikitrace.config import Compilation, Outcome


_STEP_DIR = "/work/corpus/raw/step-NNNN"


_INJECTED_MEMORY_CLAUSE = " and embeds that run's injected memory"


@dataclass(frozen=True)
class RawLayerShape:
    """What ONE ``raw/step-NNNN/`` directory holds and how its step reads, per domain."""
    # Prose fields are slotted VERBATIM, their wrapping is the rendering; the map is what a traverse agent navigates by.

    agent_logs: tuple[str, ...]
    optional_agent_logs: tuple[str, ...]
    outcome_files: tuple[str, ...]
    step_contents: str
    # A STEM: no terminal punctuation and no opening-turn clause, which :attr:`read_hint_text` appends.
    read_hint: str
    step_contents_outcome_only: str = (
        "Each `/work/corpus/raw/step-NNNN/` holds: `metadata.md` (sample id, question, EX,\n"
        "Soft-F1); `final.sql` (executed SQL) + `final.csv` (its rows)."
    )
    has_task_files: bool = False
    has_task_context: bool = False
    no_memory: bool = False
    # A projection mounts each declared turn log without its `request_context` row; the stored step keeps both.
    mount_without_request_context: bool = False

    # The prediction file the raw metadata's ``resource:`` points at; must name a file ``persist_trajectory`` writes.
    prediction_file: str = "final.sql"
    metadata_description: str = (
        "One sample's executed SQL, result rows, and per-phase agent turn logs."
    )
    # Must match the domain's ``_metadata_text``, or the frontmatter invents a score the run never measured.
    metadata_keys: tuple[str, ...] = ("db_id", "ex", "soft_f1", "refiner_used")
    task_key: str = "question"
    outcome_key: str = "ex"
    outcome_default: str = "0"
    # When set, ``outcome_key`` holds a SCORE mapped through ``outcome_labels``; when ``""`` the value IS the label.
    outcome_success_value: str = "1"
    outcome_labels: tuple[str, str] = ("success", "failure")
    metadata_artifacts: str = (
        "- [final.sql](/work/corpus/raw/{step_id}/final.sql) — executed SQL · "
        "[final.csv](/work/corpus/raw/{step_id}/final.csv) — result rows\n"
        "- turn logs: `selector.full.jsonl`, `decomposer.full.jsonl` "
        "(one JSON turn per line; final answer = last line's `parsed`)."
    )
    # Sections okfify carries through verbatim, since it rebuilds ``metadata.md`` from a fixed template.
    metadata_body_sections: tuple[str, ...] = ()
    index_score: str = "EX={ex:.0f}, SF1={soft_f1:.4f}"
    # ``index_score``'s inverse; it MUST round-trip it, which keeps re-rendering a fixed point.
    index_score_pattern: str = r"EX=(?P<ex>\d+),\s*SF1=(?P<soft_f1>[\d.]+)"
    index_score_columns: str = "(EX=.., SF1=..)"
    index_metric_names: str = "EX / Soft-F1"
    compilation_layer_desc: str = "(outcome, EX/SF1, one-line summary)"
    compilation_task_hint: str = "db_id, question, evidence"
    compilation_outcome_hint: str = "what EX/Soft-F1 do/don't establish"
    # The section is failure-only; a domain without partial credit must point the reader at something groundable.
    compilation_correct_parts_hint: str = "(failure only) parts Soft-F1 supports"
    env_compilation_sections: bool = False
    # An environment episode: the raw index lists success or failure, and the closing `answer` turn carries no answer.
    episode: bool = False
    turn_log_text: str = ""
    knowledge_provenance_hint: str = (
        "seeded train steps, each with a one-line label (links to compilation/)"
    )

    # The FALLBACK ``description`` when the compiler authored none; no ``refused`` slot, that lives on the index entry.
    compilation_description: str = "{outcome} analysis; EX={ex}, Soft-F1={soft_f1}."
    # ONE source for the frontmatter metrics and the skeleton lines: `_lines` must match `_keys` in order and members.
    compilation_metric_keys: tuple[str, ...] = ("ex", "soft_f1")
    compilation_fm_metric_lines: tuple[str, ...] = ("ex: 0 | 1", "soft_f1: <float>")
    writer_metric_header: tuple[tuple[str, str], ...] = (("ex", "EX"), ("soft_f1", "SF1"))

    @property
    def read_hint_text(self) -> str:
        """The finished "how to read a turn log" sentence for THIS corpus."""
        clause = "" if self.no_memory else _INJECTED_MEMORY_CLAUSE
        return f"{self.read_hint}{clause}."


@dataclass(frozen=True)
class RawMetricVocabulary:
    """The METRIC-naming subset of a :class:`RawLayerShape`, overridable per BENCHMARK."""
    # No field has a default, so an override states its whole vocabulary and a render/parse pair cannot change one side.

    step_contents: str
    step_contents_outcome_only: str
    metadata_keys: tuple[str, ...]
    index_score: str
    index_score_pattern: str
    index_score_columns: str
    index_metric_names: str
    compilation_layer_desc: str
    compilation_task_hint: str
    compilation_outcome_hint: str
    compilation_correct_parts_hint: str
    compilation_description: str
    compilation_metric_keys: tuple[str, ...]
    compilation_fm_metric_lines: tuple[str, ...]
    writer_metric_header: tuple[tuple[str, str], ...]


def with_metric_vocabulary(
    shape: RawLayerShape, vocab: RawMetricVocabulary | None
) -> RawLayerShape:
    """``shape`` (the domain's structure) speaking ``vocab`` (the benchmark's metrics)."""
    if vocab is None:
        return shape
    return replace(shape, **{f.name: getattr(vocab, f.name) for f in fields(vocab)})


_TASK_FILES_BULLET = (
    f"  - {_STEP_DIR}/files/: this train sample's own data files — its question's input "
    "and/or one file per answer option that carries a sequence — copied in so the frozen "
    "corpus is self-contained. Almost always a FASTA; a few are a GenBank genome, a "
    "JSON primer set or a tabular data file."
)
_TASK_CONTEXT_BULLET = (
    f"  - {_STEP_DIR}/context.md: the document this question is asked ABOUT (a protocol), "
    "copied in so the frozen corpus is readable without the benchmark. `metadata.md` "
    "shows its opening under `## Context`."
)
_TASK_FILES_CLAUSE = (
    "A sample that ships data files also holds `/work/corpus/raw/step-NNNN/files/`: its "
    "question's input, and one file per answer option carrying a sequence — almost "
    "always a FASTA, occasionally a GenBank genome, a JSON primer set or a tabular "
    "file. `metadata.md` previews each under `## Files`, so open a file itself only "
    "when its preview stops short."
)
_TASK_CONTEXT_CLAUSE = (
    "Where the question is asked ABOUT a document, that document is "
    "`/work/corpus/raw/step-NNNN/context.md`; `metadata.md` shows its opening under "
    "`## Context`, so read the file itself only when that stops short."
)


def with_no_memory(shape: RawLayerShape) -> RawLayerShape:
    """This shape, marked as describing a corpus generated WITHOUT memory."""
    return replace(shape, no_memory=True)


def with_task_context(shape: RawLayerShape) -> RawLayerShape:
    """This shape, marked as also holding a per-step ``context.md``."""
    return replace(shape, has_task_context=True)


def with_task_files(shape: RawLayerShape) -> RawLayerShape:
    """This shape, marked as also holding a per-step ``files/`` subdir."""
    return replace(shape, has_task_files=True)


DEFAULT_RAW_SHAPE = RawLayerShape(
    agent_logs=("selector.full.jsonl", "decomposer.full.jsonl"),
    optional_agent_logs=("refiner.full.jsonl",),
    outcome_files=("final.sql", "final.csv", "metadata.md"),
    step_contents=(
        "Each `/work/corpus/raw/step-NNNN/` holds: `metadata.md` (sample id, question, EX,\n"
        "Soft-F1, refiner_used); `final.sql` (executed SQL) + `final.csv` (its rows); the\n"
        "generator turn logs `selector.full.jsonl`, `decomposer.full.jsonl`,\n"
        "`refiner.full.jsonl` (refiner only if it ran)."
    ),
    read_hint=(
        "e.g. `tail -n 1 /work/corpus/raw/step-0003/decomposer.full.jsonl`. Skip the opening\n"
        "`system` message turn — large"
    ),
)


def _numeric_id(prefix: str, value: str) -> int:
    """Extract the trailing integer from ``prefix-NNNN`` (``-1`` if malformed)."""
    if not value.startswith(prefix):
        return -1
    try:
        return int(value[len(prefix) :])
    except ValueError:
        return -1


def _render_index(opening: str, columns: str, entries: list[tuple[str, str, str]]) -> str:
    """Common index layout shared by the layer indexes."""
    # The EXACT absolute path the traverse agent cats; it must match the runner's bind mount at ``/work/corpus``.
    lines = [opening, columns]
    if entries:
        lines.append("")
        lines.extend(
            f"{i}. [{title}](/work/corpus/{path}): {contents}"
            for i, (title, path, contents) in enumerate(entries, 1)
        )
    return "\n".join(lines) + "\n"


LAYER_INDEX = "_index.md"


def _doc_template(
    name: str, frontmatter: list[str], sections: list[tuple[str, str]]
) -> str:
    """A multi-line, grep-friendly skeleton of one document under this layer."""
    block = ["  ---", *(f"  {fm}" for fm in frontmatter), "  ---"]
    for header, hint in sections:
        block.append(f"  {header}")
        block.append(f"  {hint}")
    return f"Each `{name}` has this shape:\n" + "\n".join(block)


def _jsonl_template(shape: RawLayerShape) -> str:
    """A concise description of the `*.full.jsonl` turn-log format."""
    if shape.turn_log_text:
        return shape.turn_log_text
    if shape.episode:
        return (
            "Each `solver.full.jsonl` is one JSON row per line. The episode's steps are the "
            "`function_call` rows, each with the call's `arguments`, the agent's `reasoning` "
            "and the environment's `result`."
        )
    return (
        "Each `*.full.jsonl` is one JSON turn per line (OpenAI Responses-API style). "
        "Every line is `{\"turn\": <int>, \"type\": ..., \"role\": ..., ...}` where "
        "`type` is one of message | function_call | request_context | answer and "
        "`role` is one of system | user | assistant | tool. Read the LAST line — the "
        "`answer` turn — and take its `parsed` for the final structured answer."
    )


def _one_line(text: str, limit: int = 500) -> str:
    """Squeeze arbitrary text onto ONE index line."""
    # Entries are re-parsed per line, so text spilling across lines would break the render -> parse -> render fixed point.
    squeezed = " ".join(text.split())
    return squeezed if len(squeezed) <= limit else squeezed[: limit - 1] + "…"


def _strip_outcome_tag(title: str, outcome: Outcome) -> str:
    """Drop the code-prepended ``[<outcome>] `` tag(s) from a compilation title."""
    # Repeatedly, since a headline can open with its own tag; only this compilation's OWN outcome at position 0, no loose bracket sweep.
    tag = f"[{outcome.value}]"
    while title.startswith(tag):
        title = title[len(tag):].lstrip()
    return title


def _entry_tail_source(c: Compilation, entry_source: str) -> str:
    """The raw field feeding an index entry's tail, before ``_one_line`` squeezes it."""
    if entry_source == "title":
        headline = _strip_outcome_tag((c.title or "").strip(), c.outcome)
        return headline or c.description or c.question
    if entry_source == "description":
        return c.description or c.question
    return c.question


def _entry_contents(c: Compilation, entry_source: str, shape: RawLayerShape) -> str:
    """One index entry's tail: ``<outcome> (<score>) | <one-line summary>``."""
    # ``refused`` is a score slot because ``outcome`` reads ``failure`` for both a wrong answer and a refusal.
    head = entry_head(c.outcome.value, shape.index_score.format(**_score_slots(shape, c)))
    return f"{head} | {_one_line(_entry_tail_source(c, entry_source))}"


def entry_head(outcome: str, score: str) -> str:
    """``<outcome> (<score>)``, or the bare outcome for a shape that declares no score."""
    return f"{outcome} ({score})" if score else outcome


def _score_slot_names(fmt: str) -> set[str]:
    """The field names ``fmt`` interpolates, spec and conversion stripped."""
    return {name for _, name, _, _ in Formatter().parse(fmt) if name}


def _score_slots(shape: RawLayerShape, c: Compilation) -> dict[str, float]:
    """The values ``index_score`` formats, each under the name it declares."""
    # ``refused`` is added regardless: a semantic flag, not a score, never named by a domain that cannot abstain.
    slots = {k: c.metric(k) for k in shape.compilation_metric_keys}
    slots["refused"] = float(c.refused)
    return slots


def _entry_legend(shape: RawLayerShape) -> str:
    """The "Each entry: ..." legend shared by the raw + compilation indexes."""
    # The leading word is the shared roll-up and the parenthetical the domain's metric; both following sentences are CONDITIONAL.
    legend = (
        "Each entry: N. [step](path): <outcome> "
        + (f"{shape.index_score_columns} " if shape.index_score_columns else "")
        + "| <one-line summary>\n"
        "`<outcome>` is `success`|`failure`"
        + (
            " — the two-state pass/fail roll-up of the score\nbeside it, shared by "
            "every domain." if shape.index_score_columns else "."
        )
    )
    if not shape.outcome_success_value:
        legend += (
            " A step's own per-sample outcome word is the\n"
            f"`{shape.outcome_key}:` field of `/work/corpus/raw/step-NNNN/metadata.md`."
        )
    # By the SLOT NAMES the format declares, never a substring: a slot carries a format spec.
    if "refused" in _score_slot_names(shape.index_score):
        legend += (
            " An entry with\n`refused=1` ABSTAINED: it rolls up to `failure`, but the "
            "score beside it reads 0\nonly because no answer was committed — a refusal "
            "is not a wrong answer and has\nno score of its own."
        )
    return legend


def _compilation_sections(
    raw_shape: RawLayerShape, compiler_confidence: bool
) -> list[tuple[str, str]]:
    """The skeleton's section list, following the family the compiler writes in."""
    if not raw_shape.env_compilation_sections:
        return [
            ("## Task", raw_shape.compilation_task_hint),
            ("## Outcome Interpretation", raw_shape.compilation_outcome_hint),
            ("## Key Decisions", "(success only) decisions that mattered, per agent"),
            ("## Failure Causes",
             "(failure only) confidence-ranked: what went wrong and what in the\n  trajectory shows it"),
            ("## Correct Parts", raw_shape.compilation_correct_parts_hint),
            ("## Candidate Insights",
             "transferable lessons; on a failure each carries its confidence"),
            ("## Uncertainties",
             "(failure only) what the analysis could not confirm, and what would settle it"),
        ]
    sections = [
        ("## Task", raw_shape.compilation_task_hint),
        ("## Key Decisions",
         "(success only) decisions necessary for success, each with its support"),
        ("## Failure Causes",
         ("(failure only) confidence-ranked: " if compiler_confidence else "(failure only) ")
         + "decisions that caused the failure, each with its support"),
        ("## Candidate Insights", "transferable lessons"),
    ]
    if compiler_confidence:
        sections.append(
            ("## Uncertainties", "(failure only) what the analysis could not confirm")
        )
    return sections


_INSIGHT_LINKS = ("## Insights from this run",
                  "one line per insight drawn from this run: id, title and path")


def render_compilation_index(
    comps: list[Compilation], *, entry_source: str, raw_shape: RawLayerShape,
    compiler_confidence: bool = False,
) -> str:
    """The mounted ``compilation/_index.md``: one entry per compilation, in step order."""
    ordered = sorted(comps, key=lambda c: _numeric_id("step-", c.step_id))
    entries = [
        (c.step_id, f"compilation/{c.step_id}.md", _entry_contents(c, entry_source, raw_shape))
        for c in ordered
    ]
    fm = ["type: trajectory-compilation", "title: [<outcome>] <agent headline>",
          "description: <one-line>", "step_id: step-NNNN",
          "outcome: success | failure", *raw_shape.compilation_fm_metric_lines]
    sections = [(_INSIGHT_LINKS if h == "## Candidate Insights" else (h, t))
                for h, t in _compilation_sections(raw_shape, compiler_confidence)]
    opening = (
        "# Trajectory compilations — one per finished train step.\n\n"
        + _doc_template("/work/corpus/compilation/step-NNNN.md", fm, sections) + "\n"
    )
    return _render_index(opening, _entry_legend(raw_shape), entries)


def step_contents_paragraph(shape: RawLayerShape, *, outcome_only: bool = False) -> str:
    """The "what a step dir holds" paragraph for THIS shape's gates."""
    base = shape.step_contents_outcome_only if outcome_only else shape.step_contents
    tail = [c for c, on in ((_TASK_FILES_CLAUSE, shape.has_task_files),
                            (_TASK_CONTEXT_CLAUSE, shape.has_task_context)) if on]
    return base + ("\n" + "\n".join(tail) if tail else "")


_R_ENTRY = re.compile(
    r"\[(step-\d+)\]\([^)]*raw/[^)]*\):\s*(\w+)\s*(?:\(([^)]*)\))?\s*\|\s*(.*)"
)


def _render_frontmatter(fields: list[tuple[str, str]]) -> str:
    """Render ordered ``key: value`` pairs inside a ``---`` fence."""
    inner = "\n".join(f"{key}: {value}" for key, value in fields)
    return f"---\n{inner}\n---\n"


#: How each DECLARED metric key spells its value in frontmatter; `refused` alone comes off the record's own flag.
_METRIC_VALUE = {
    "ex": lambda c: str(int(c.metric("ex"))),
    "soft_f1": lambda c: format(c.metric("soft_f1"), ".4f"),
    "correct": lambda c: str(bool(c.metric("correct"))),
    "refused": lambda c: str(c.refused),
}


def metric_value(key: str, comp: Compilation) -> str:
    """One declared metric's frontmatter spelling; a key the table lacks prints its number."""
    # A whole number prints bare and anything else at four places, so a count and a ratio both read as what they are.
    fn = _METRIC_VALUE.get(key)
    if fn is not None:
        return fn(comp)
    value = comp.metric(key)
    return str(int(value)) if value == int(value) else format(value, ".4f")


def compilation_to_markdown(
    comp: Compilation, *, raw_shape: RawLayerShape | None = None
) -> str:
    """Serialize a compilation to its ``step-NNNN.md`` form (body verbatim)."""
    shape = raw_shape or DEFAULT_RAW_SHAPE
    keys = shape.compilation_metric_keys
    outcome = comp.outcome.value if isinstance(comp.outcome, Outcome) else str(comp.outcome)
    rows = [
        ("type", "trajectory-compilation"),
        ("title", comp.title),
        ("description", comp.description),
        ("step_id", comp.step_id),
        ("outcome", outcome),
    ]
    rows += [(k, metric_value(k, comp)) for k in keys]
    return f"{_render_frontmatter(rows)}\n{comp.body}"


@dataclass(frozen=True)
class RawStepFacts:
    """What one frozen step contributes to a raw-only index entry."""

    outcome: str
    tail: str
    metrics: dict[str, float] = field(default_factory=dict)
    refused: bool = False

    def metric(self, key: str, default: float = 0.0) -> float:
        """This step's value for a declared metric name."""
        return self.metrics.get(key, default)


def render_raw_index_from_steps(
    steps: list[tuple[str, RawStepFacts]],
    *,
    raw_shape: RawLayerShape = DEFAULT_RAW_SHAPE,
    outcome_subset: bool = False,
) -> str:
    """``raw/_index.md`` for a corpus with no compilation layer, one entry per step."""
    # Forced no-memory: the case library re-serves the frozen no-memory corpus.
    raw_shape = with_no_memory(raw_shape)
    ordered = sorted(steps, key=lambda s: s[0])
    entries = [
        (
            sid,
            f"raw/{sid}/",
            entry_head(f.outcome, raw_shape.index_score.format(**_score_slots(raw_shape, f)))
            + f" | {_one_line(f.tail)}",
        )
        for (sid, f) in ordered
    ]
    # An outcome subset holds one outcome only, so the title does not promise both.
    title = ("# Past tasks — one train step each" + ("" if outcome_subset else ", successes and failures")
             if raw_shape.episode
             else "# Past tasks — each is one solved train step")
    if raw_shape.agent_logs:
        opening = (
            f"{title}.\n"
            + step_contents_paragraph(raw_shape) + "\n"
            "\n"
            + _jsonl_template(raw_shape) + "\n"
            + raw_shape.read_hint_text + "\n"
        )
    else:
        opening = (
            f"{title} (outcomes only; no agent conversation).\n"
            + step_contents_paragraph(raw_shape, outcome_only=True) + "\n"
        )
    return _render_index(opening, _entry_legend(raw_shape), entries)
