"""The wiki a WikiTrace run mounts read-only at /work/corpus: raw, compilation and insight layers."""

from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

from wikitrace.harness.terminal.docker import make_readonly, make_writable
from wikitrace.memory.insights import (
    InsightItem, render_insight_document, render_insight_index, task_types_differ,
)
from wikitrace.memory.store import LAYER_INDEX, RawLayerShape, render_raw_index_from_steps, with_no_memory

MOUNTED_READ_HINT = "Skip the `system` message: it is large"
# The rawturns script, mounted read-only under both names; its program name picks the view.
DESCEND_SCRIPT = Path(__file__).with_name("rawturns.py")
DESCEND_FILES = ("/work/tools/actions", "/work/tools/turn")


def _without_request_context(log: Path) -> str:
    lines = log.read_text(encoding="utf-8").splitlines(keepends=True)
    return "".join(ln for ln in lines if not ln.strip() or json.loads(ln).get("type") != "request_context")


def _copy_raw_step(src: Path, dst: Path, raw_shape: RawLayerShape) -> None:
    """Copy one frozen train step with the files its shape declares, and never the compiler's log."""
    dst.mkdir(parents=True, exist_ok=True)
    logs = (*raw_shape.agent_logs, *raw_shape.optional_agent_logs)
    for name in (*raw_shape.outcome_files, *logs):
        if not (src / name).exists():
            continue
        if raw_shape.mount_without_request_context and name in logs:
            (dst / name).write_text(_without_request_context(src / name), encoding="utf-8")
        else:
            shutil.copy2(src / name, dst / name)
    if (src / "files").is_dir():
        shutil.copytree(src / "files", dst / "files", dirs_exist_ok=True)


def build_wiki(
    dest: Path, *, items: list[InsightItem], compilations: dict[str, str], source_corpus: Path,
    dom: Any, raw_shape: RawLayerShape, compilation_index: str,
) -> Path:
    """Write the mounted wiki into ``dest``, writable; ``compilations`` maps step to text."""
    missing = sorted({i.source_step for i in items} - set(compilations))
    if missing:
        raise ValueError(f"insights link to steps with no compilation: {missing}")
    dest = Path(dest)
    if dest.exists():
        # A killed run skips the `finally` that unfreezes this tree.
        make_writable(dest)
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    raw_shape = with_no_memory(raw_shape)
    if raw_shape.mount_without_request_context:
        raw_shape = replace(raw_shape, read_hint=MOUNTED_READ_HINT)
    if (source_corpus / "corpus.json").exists():
        shutil.copy2(source_corpus / "corpus.json", dest / "corpus.json")
    raw_steps = []
    for sd in sorted((source_corpus / "raw").glob("step-*")):
        if sd.is_dir():
            _copy_raw_step(sd, dest / "raw" / sd.name, raw_shape)
            raw_steps.append((sd.name, dom.read_raw_step(sd)))
    (dest / "raw" / LAYER_INDEX).write_text(render_raw_index_from_steps(raw_steps, raw_shape=raw_shape),
                                            encoding="utf-8")
    (dest / "compilation").mkdir()
    for step, text in sorted(compilations.items()):
        (dest / "compilation" / f"{step}.md").write_text(text, encoding="utf-8")
    (dest / "compilation" / LAYER_INDEX).write_text(compilation_index, encoding="utf-8")
    insight = dest / "insight"
    insight.mkdir()
    typed = task_types_differ(items)
    for item in items:
        (insight / f"{item.id}.md").write_text(render_insight_document(item, task_type=typed), encoding="utf-8")
    (insight / LAYER_INDEX).write_text(render_insight_index(items), encoding="utf-8")
    return dest


@contextmanager
def frozen(corpus: Path | None) -> Iterator[Path | None]:
    """Hold ``corpus`` read-only while the test phase greps it."""
    if corpus is None:
        yield None
        return
    make_readonly(corpus)
    try:
        yield corpus
    finally:
        make_writable(corpus)
