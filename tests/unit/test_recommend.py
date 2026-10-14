"""Frozen train-run recommendations: ranking, the file, and the loader's refusals."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import wikitrace.envs  # noqa: F401  (registers the built-in domains)
from wikitrace.config import append_result
from wikitrace.config import load_config
from wikitrace.config import make_split
from wikitrace.envs import resolve
from wikitrace.envs.textgame import manifest as tg_manifest
from wikitrace.envs.textgame.models import TextGameOutcome, TextGameResult
from wikitrace.harness.llm.openrouter import EmbedResponse
from wikitrace.memory.compiler import read_episodes
from wikitrace.memory.recommend import ROLE, load_recommendations, rank, recommendation_path
from wikitrace.run.build import ensure_recommendations
from wikitrace.run.spec import resolve_spec

# Three train games and two test games; each task text names the axis of its fake vector.
_GAMES = [
    ("train", "pick_and_place_simple-Kettle-None-Shelf-1", "pick_and_place_simple",
     "Your task is to: put a kettle in shelf."),
    ("train", "look_at_obj_in_light-Pen-None-DeskLamp-2", "look_at_obj_in_light",
     "Your task is to: look at a pen under the desklamp."),
    ("train", "pick_and_place_simple-Mug-None-Shelf-3", "pick_and_place_simple",
     "Your task is to: put a mug in shelf."),
    ("valid_seen", "pick_and_place_simple-Cup-None-Shelf-1", "pick_and_place_simple",
     "Your task is to: put a kettle in cabinet."),
    ("valid_seen", "look_at_obj_in_light-Book-None-DeskLamp-4", "look_at_obj_in_light",
     "Your task is to: look at a pen under the lamp."),
]


def _vector(text: str) -> list[float]:
    if "kettle" in text:
        return [1.0, 0.0, 0.1]
    if "pen" in text:
        return [0.0, 1.0, 0.0]
    return [0.8, 0.0, 0.0]


class _Embedder:
    """One deterministic vector per text, in the OpenRouter transport's shape."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, *, model, texts):
        self.texts.extend(texts)
        return EmbedResponse(vectors=[_vector(t) for t in texts], model_returned=model,
                             raw={"usage": {"prompt_tokens": 5, "cost": 0.0001}})


@pytest.fixture
def suite(tmp_path, monkeypatch, temp_config_writer):
    """An ALFWorld suite and a frozen corpus of its train half."""
    root = tmp_path / "alfworld"
    entries: dict[str, list[dict]] = {"train": [], "valid_seen": [], "valid_unseen": []}
    for split, name, task_type, task_text in _GAMES:
        (root / "json_2.1.1" / split / name / "trial_0").mkdir(parents=True)
        (root / "json_2.1.1" / split / name / "trial_0" / "game.tw-pddl").write_text("{}")
        entries[split].append({
            "game": f"json_2.1.1/{split}/{name}/trial_0", "split": split,
            "task_type": task_type, "scene_id": 1, "task_text": task_text,
        })
    data = tmp_path / "data"
    data.mkdir()
    (data / "alfworld.manifest.json").write_text(json.dumps(entries))
    monkeypatch.setattr(tg_manifest, "DATA_DIR", data)
    cfg_path = temp_config_writer({"dataset": {"alfworld_root": str(root)}})

    cfg = load_config(cfg_path)
    dom = resolve("textgame", "alfworld")
    split = make_split(dom.benchmark.load_suite(cfg.dataset, "valid-seen"), cfg.dataset,
                       "valid-seen")
    corpus = tmp_path / "outputs" / "corpus" / "alfworld" / "alfworld" / "rep-00"
    for i, sample in enumerate(split.train):
        _persist(dom, corpus, f"step-{i:04d}", sample)
    (corpus / "corpus.json").write_text(json.dumps({"fp8": "abcd1234", "insight_items": True}))
    return cfg_path, dom, corpus, split


def _persist(dom, corpus: Path, step: str, sample) -> None:
    result = TextGameResult(
        sample_id=sample.id, db_id=sample.db_id, question=sample.task_text,
        task_type=sample.task_type, variant=sample.variant, success=True, steps=0,
    )
    step_dir = corpus / "raw" / step
    dom.persist_traverse_trajectory(step_dir, sample, TextGameOutcome(result))
    (step_dir / "solver.full.jsonl").write_text(
        json.dumps({"type": "message", "role": "user", "content": sample.task_text}) + "\n")
    append_result(corpus, step, result)


def _build(suite, embedder=None) -> Path:
    cfg_path, _, corpus, _ = suite
    spec = resolve_spec(cfg_path, "alfworld", "valid-seen", 0)
    return ensure_recommendations(spec, corpus, embedder or _Embedder())


def _train(dom, corpus) -> dict[str, str]:
    return {s: ep.sample_id for s, ep in read_episodes(dom, corpus).items()}


def test_rank_is_cosine_then_step_id():
    train = [("step-0002", [2.0, 0.0]), ("step-0001", [1.0, 0.0]), ("step-0000", [0.0, 3.0]),
             ("step-0003", [1.0, 1.0])]
    # Scale does not matter, and the two equal scores keep step order.
    assert rank(train, [5.0, 0.0]) == [
        {"step": "step-0001", "score": 1.0}, {"step": "step-0002", "score": 1.0},
        {"step": "step-0003", "score": 0.707107},
    ]
    assert [r["step"] for r in rank(train, [0.0, 1.0], k=1)] == ["step-0000"]


def test_file_holds_train_runs_and_each_test_samples_top_three(suite):
    cfg_path, _, corpus, split = suite
    embedder = _Embedder()
    path = _build(suite, embedder)

    assert path == recommendation_path(corpus, "valid-seen")
    content = json.loads(path.read_text())
    assert content["model"] == "openai/text-embedding-3-small" and content["k"] == 3
    assert content["train"] == [
        {"step": f"step-{i:04d}", "sample_id": s.id, "instruction": s.task_text}
        for i, s in enumerate(split.train)
    ]
    steps = {row["instruction"]: row["step"] for row in content["train"]}
    kettle, pen, mug = (steps[f"Your task is to: {t}."] for t in
                        ("put a kettle in shelf", "look at a pen under the desklamp",
                         "put a mug in shelf"))
    ranking = content["ranking"]
    assert set(ranking) == {s.id for s in split.test}
    by_question = {s.task_text: ranking[s.id] for s in split.test}
    assert [r["step"] for r in by_question["Your task is to: put a kettle in cabinet."]] == [
        kettle, mug, pen]
    assert by_question["Your task is to: look at a pen under the lamp."][0] == {
        "step": pen, "score": 1.0}
    # Symmetric: the train and test texts are embedded bare, each logged under the role.
    assert sorted(embedder.texts) == sorted([s.task_text for s in split.train]
                                            + [s.task_text for s in split.test])
    calls = [json.loads(ln) for ln in path.with_name("valid-seen.llm_calls.jsonl")
             .read_text().splitlines()]
    assert len(calls) == 5 and {c["role"] for c in calls} == {ROLE}
    assert {c["model_requested"] for c in calls} == {"openai/text-embedding-3-small"}
    # Deterministic serialization: sorted keys, two-space indent, trailing newline.
    assert path.read_text() == json.dumps(content, indent=2, sort_keys=True,
                                          ensure_ascii=False) + "\n"


def test_an_existing_file_is_returned_without_embedding(suite):
    first = _build(suite).read_text()
    embedder = _Embedder()
    assert _build(suite, embedder).read_text() == first
    assert embedder.texts == []


def test_loader_returns_the_ranking_for_the_asked_test_samples(suite):
    _, dom, corpus, split = suite
    path = _build(suite)
    ids = [s.id for s in split.test]
    ranking = load_recommendations(path, train=_train(dom, corpus), test_ids=ids[:1])
    assert list(ranking) == ids[:1]
    assert len(ranking[ids[0]]) == 3 and all(isinstance(s, float) for _, s in ranking[ids[0]])


def test_loader_refuses_a_corpus_whose_steps_name_other_samples(suite):
    _, dom, corpus, split = suite
    path = _build(suite)
    content = json.loads(path.read_text())
    content["train"][0]["sample_id"], content["train"][1]["sample_id"] = (
        content["train"][1]["sample_id"], content["train"][0]["sample_id"])
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="other train runs"):
        load_recommendations(path, train=_train(dom, corpus), test_ids=[split.test[0].id])


def test_loader_refuses_a_corpus_with_an_extra_step(suite):
    _, dom, corpus, split = suite
    path = _build(suite)
    _persist(dom, corpus, "step-0099", split.test[0])
    with pytest.raises(ValueError, match="step-0099"):
        load_recommendations(path, train=_train(dom, corpus), test_ids=[split.test[0].id])


def test_loader_refuses_a_test_sample_missing_from_the_file(suite):
    _, dom, corpus, _ = suite
    path = _build(suite)
    with pytest.raises(ValueError, match="no recommendation"):
        load_recommendations(path, train=_train(dom, corpus), test_ids=["alfworld_9999_x"])
