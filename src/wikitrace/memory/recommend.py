"""The three train runs whose instruction embeds nearest each test task."""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from wikitrace.config import LLMCallRecord
from wikitrace.harness.llm.usage import parse_usage

K = 3
ROLE = "itemwiki-recommend-embed"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
# Scores are rounded before ranking, so the file's order is the order its numbers state.
_DIGITS = 6


@dataclass
class Embedder:
    """Embedding calls through the run's transport, each logged as an LLMCallRecord."""

    transport: Any
    on_call: Callable[[LLMCallRecord], None] | None = None
    max_retries: int = 0
    model: str = EMBEDDING_MODEL

    def embed(self, text: str, *, sample_id: str = "") -> list[float]:
        # A task text that does not exist yet would embed as "", ranking every query alike.
        if not text.strip():
            raise ValueError(f"{ROLE}: refusing to embed an empty text (sample {sample_id or '?'})")
        start = time.time()
        response = self._retrying(lambda: self.transport.embed(model=self.model, texts=[text]))
        usage = parse_usage(response.raw)
        if self.on_call:
            self.on_call(LLMCallRecord(
                call_id=uuid.uuid4().hex[:16], generation_id="", role=ROLE, phase="build",
                sample_id=sample_id, model_requested=self.model, model_returned=response.model_returned,
                prompt_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(), response_hash="",
                latency=time.time() - start, finish_reason="", prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens, reasoning_tokens=usage.reasoning_tokens,
                cached_tokens=usage.cached_tokens, cost=usage.cost, provider="",
            ))
        return response.vectors[0]

    def _retrying(self, call: Callable[[], Any]) -> Any:
        for attempt in range(self.max_retries):
            try:
                if response := call():
                    return response
            except Exception:  # noqa: BLE001 — the final attempt re-raises
                pass
            time.sleep(min(2 ** attempt, 8))
        return call()


def recommendation_path(corpus_dir: str | Path, suite: str) -> Path:
    """``<corpus>/recommendations/<suite>.json``."""
    return Path(corpus_dir) / "recommendations" / f"{suite}.json"


def _unit(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        raise ValueError("cannot rank against a zero embedding")
    return [x / norm for x in vector]


def rank(train: Sequence[tuple[str, Sequence[float]]], query: Sequence[float], k: int = K) -> list[dict[str, Any]]:
    """The k train steps nearest the query by cosine, ties broken by step id."""
    q = _unit(query)
    scored = [(round(sum(a * b for a, b in zip(q, _unit(vector))), _DIGITS), step) for step, vector in train]
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [{"step": step, "score": score} for score, step in scored[:k]]


def _train_vectors(train_steps: Sequence[tuple[str, str, str]], embedder: Embedder) -> list[tuple[str, list[float]]]:
    return [(step, embedder.embed(text, sample_id=sample_id)) for step, sample_id, text in train_steps]


def recommend_at_reset(
    train_steps: Sequence[tuple[str, str, str]], embedder: Embedder, k: int = K,
) -> Callable[[str, str], list[dict[str, Any]]]:
    """Embed the train instructions now; the returned ranker embeds a query inside its episode."""
    train_vectors = _train_vectors(train_steps, embedder)
    return lambda query, sample_id: rank(train_vectors, embedder.embed(query, sample_id=sample_id), k)


def compute_recommendations(
    train_steps: Sequence[tuple[str, str, str]], test_samples: Sequence[tuple[str, str]],
    embedder: Embedder, k: int = K,
) -> dict[str, Any]:
    """The file's content from (step, sample_id, instruction) and (sample_id, question) rows."""
    train_vectors = _train_vectors(train_steps, embedder)
    return {
        "model": embedder.model,
        "k": k,
        "train": [{"step": step, "sample_id": sample_id, "instruction": text}
                  for step, sample_id, text in train_steps],
        "ranking": {sample_id: rank(train_vectors, embedder.embed(question, sample_id=sample_id), k)
                    for sample_id, question in test_samples},
    }


def write_recommendations(path: Path, content: dict[str, Any]) -> None:
    """Write whole, then rename, so a killed build never leaves a file a run would read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(content, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_recommendations(
    path: str | Path, *, train: dict[str, str], test_ids: Sequence[str],
) -> dict[str, list[tuple[str, float]]]:
    """Each test sample's ranked (step, score) pairs, refused if ``train`` is not the file's."""
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {row["step"]: row["sample_id"] for row in content["train"]}
    if train != expected:
        differ = sorted(k for k in expected.keys() | train.keys() if expected.get(k) != train.get(k))
        raise ValueError(f"the corpus names other train runs than {path} at {differ[:5]} "
                         f"({len(differ)} steps differ)")
    ranking = content["ranking"]
    missing = [t for t in test_ids if t not in ranking]
    if missing:
        raise ValueError(f"{path} has no recommendation for test samples {missing[:5]}")
    return {t: [(r["step"], r["score"]) for r in ranking[t]] for t in test_ids}
