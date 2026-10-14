"""AppWorld's three methods render the pinned prompts over the host-served fake episode."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from wikitrace.harness.terminal import FakeTraverseRunner
from wikitrace.memory.insights import Episode, InsightItem, render_inject, render_recommendation
from wikitrace.run.cell import memory_kwargs_for
from tests.regression.test_method_prompts import _check
from tests.unit.test_appworld import _SCORED_FAKE_ENV, _dom, _sample, _serve_on_host, _turns_and_context

ITEMS = [InsightItem(id=f"insight-000{n}", title=f"Title {n}",
                     sections=(("Insight", f"Do {n}."), ("Applicability", f"When {n}.")),
                     source_step=f"step-000{n}", outcome=("success", "failure")[n], task=f"Task {n}",
                     task_type="", apps=("venmo",)) for n in range(2)]
EPISODES = {f"step-000{n}": Episode(step=f"step-000{n}", sample_id=f"s{n}", outcome=("success", "failure")[n],
                                    task=f"Task {n}", task_type="", apps=("venmo",)) for n in range(2)}


def arm_kwargs(method: str, corpus: Path) -> tuple[Path | None, dict]:
    """The memory a cell hands AppWorld's episode for one method."""
    if method == "no-memory":
        return None, {}
    kw = dict(memory_text=render_recommendation(["step-0000", "step-0001"], ITEMS, EPISODES),
              user_prefix=f"<memory_documents>\n{render_inject(ITEMS, source=method == 'wikitrace')}\n"
                          f"</memory_documents>",
              **memory_kwargs_for(method, "appworld"))
    if method == "wikitrace":
        (corpus / "insight").mkdir(parents=True, exist_ok=True)
        return corpus, kw
    return None, kw


def test_appworld_method_prompts(tmp_path, monkeypatch):
    (tmp_path / "scored_env.py").write_text(_SCORED_FAKE_ENV, encoding="utf-8")
    _serve_on_host(monkeypatch, f"{tmp_path / 'scored_env.py'}:make_env")
    _, turns, context = _turns_and_context()
    from wikitrace.harness.terminal import TerminalResult

    record = {}
    for method in ("no-memory", "kinj", "wikitrace"):
        seen: dict = {}

        def handler(**kw):
            seen.update({k: kw[k] for k in ("system_prompt", "user_msg", "citation_prompt")})
            seen["readonly_files"] = {k: hashlib.sha256(Path(v).read_bytes()).hexdigest()
                                      for k, v in (kw["readonly_files"] or {}).items()}
            return TerminalResult(turns=turns, read_set=[], final_context=context, usage={"cost": 0.01})

        corpus, kw = arm_kwargs(method, tmp_path / "corpus")
        _dom().run_traverse(
            _sample(root=str(tmp_path / "install")), FakeTraverseRunner(handler), corpus,
            tmp_path / method, tools="read,bash", max_turns=70, sentinel_name="__out.json", max_retries=1,
            repair_fn=None, **{"memory_text": "", **kw},
        )
        record[method] = seen
    _check("appworld", json.loads(json.dumps(record).replace(str(tmp_path), "<TMP>")))
