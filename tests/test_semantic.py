"""Tests for the Voyage-backed semantic search module."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from pycon_cal_scraper.models import Event, EventType
from pycon_cal_scraper.semantic import (
    DEFAULT_MODEL,
    EmbeddingCache,
    coverage,
    embed_events,
    event_to_text,
    filter_by_negative_phrases,
    semantic_search,
    semantic_search_with_scores,
)

PACIFIC = ZoneInfo("America/Los_Angeles")


def _event(eid: str, title: str, **extra: object) -> Event:
    payload: dict[str, object] = {
        "id": eid,
        "url": f"https://us.pycon.org/2026/schedule/presentation/{eid}/",
        "title": title,
        "type": EventType.talk,
        "speakers": ["Pat"],
        "start": datetime(2026, 5, 15, 9, 0, tzinfo=PACIFIC),
        "end": datetime(2026, 5, 15, 9, 30, tzinfo=PACIFIC),
        "room": "Room 1",
    }
    payload.update(extra)
    return Event.model_validate(payload)


class _StubEmbedder:
    """In-memory embedder: maps each input text to a fixed vector for tests."""

    def __init__(self, vectors_by_text: dict[str, list[float]]) -> None:
        self.vectors_by_text = vectors_by_text
        self.calls: list[list[str]] = []

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.vectors_by_text[t] for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self.vectors_by_text[text]


def test_event_to_text_includes_key_fields() -> None:
    e = _event("1", "Async patterns", abstract="A primer.", audience_level="Some experience")
    text = event_to_text(e)
    assert "Title: Async patterns" in text
    assert "Speakers: Pat" in text
    assert "Audience level: Some experience" in text
    assert "Abstract: A primer." in text


def test_embedding_cache_roundtrip(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "embeddings.npz", model=DEFAULT_MODEL)
    cache.put("a", np.array([0.1, 0.2, 0.3], dtype=np.float32))
    cache.put("b", np.array([0.4, 0.5, 0.6], dtype=np.float32))
    cache.save()

    reloaded = EmbeddingCache.load(tmp_path / "embeddings.npz", model=DEFAULT_MODEL)
    assert "a" in reloaded
    assert "b" in reloaded
    a_back = reloaded.get("a")
    assert a_back is not None
    assert np.allclose(a_back, [0.1, 0.2, 0.3])


def test_embedding_cache_discards_on_model_mismatch(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "embeddings.npz", model="voyage-3-lite")
    cache.put("a", np.ones(4, dtype=np.float32))
    cache.save()

    fresh = EmbeddingCache.load(tmp_path / "embeddings.npz", model="voyage-3")
    assert "a" not in fresh
    assert len(fresh) == 0


def test_embedding_cache_load_missing_returns_empty(tmp_path: Path) -> None:
    fresh = EmbeddingCache.load(tmp_path / "absent.npz", model=DEFAULT_MODEL)
    assert len(fresh) == 0


async def test_embed_events_skips_already_cached(tmp_path: Path) -> None:
    e1 = _event("1", "Async")
    e2 = _event("2", "Logging")

    embedder = _StubEmbedder(
        {
            event_to_text(e1): [1.0, 0.0, 0.0],
            event_to_text(e2): [0.0, 1.0, 0.0],
        }
    )
    cache = EmbeddingCache(tmp_path / "embeddings.npz", model=DEFAULT_MODEL)
    await embed_events([e1, e2], embedder, cache)
    assert len(cache) == 2
    assert len(embedder.calls) == 1

    # Adding a third event should embed only the new one.
    e3 = _event("3", "Memory")
    embedder.vectors_by_text[event_to_text(e3)] = [0.0, 0.0, 1.0]
    embedder.calls.clear()
    await embed_events([e1, e2, e3], embedder, cache)
    assert len(cache) == 3
    assert embedder.calls == [[event_to_text(e3)]]


async def test_embed_events_fires_progress_callback(tmp_path: Path) -> None:
    events = [_event(str(i), f"Talk {i}") for i in range(5)]
    embedder = _StubEmbedder({event_to_text(e): [float(i), 0.0] for i, e in enumerate(events)})
    cache = EmbeddingCache(tmp_path / "embeddings.npz", model=DEFAULT_MODEL)

    seen: list[tuple[int, int]] = []

    def on_progress(completed: int, total: int) -> None:
        seen.append((completed, total))

    await embed_events(events, embedder, cache, batch_size=2, on_progress=on_progress)

    assert seen[0] == (0, 5)
    assert seen[-1] == (5, 5)
    assert all(0 <= c <= 5 and t == 5 for c, t in seen)


def test_semantic_search_ranks_by_cosine(tmp_path: Path) -> None:
    e_async = _event("1", "Async patterns")
    e_rust = _event("2", "Rust interop")
    e_logging = _event("3", "Logging tricks")

    cache = EmbeddingCache(tmp_path / "embeddings.npz", model=DEFAULT_MODEL)
    cache.put("1", np.array([1.0, 0.0, 0.0], dtype=np.float32))
    cache.put("2", np.array([0.0, 1.0, 0.0], dtype=np.float32))
    cache.put("3", np.array([0.9, 0.1, 0.0], dtype=np.float32))

    query = [1.0, 0.0, 0.0]
    results = semantic_search([e_async, e_rust, e_logging], query, cache)
    assert [e.id for e in results[:2]] == ["1", "3"]
    assert results[-1].id == "2"


def test_semantic_search_ignores_uncached_events(tmp_path: Path) -> None:
    e1 = _event("1", "Async")
    e2 = _event("2", "Memory")

    cache = EmbeddingCache(tmp_path / "embeddings.npz", model=DEFAULT_MODEL)
    cache.put("1", np.array([1.0, 0.0], dtype=np.float32))

    results = semantic_search([e1, e2], [1.0, 0.0], cache)
    assert [e.id for e in results] == ["1"]


def test_semantic_search_empty_cache_returns_empty(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "embeddings.npz", model=DEFAULT_MODEL)
    assert semantic_search([_event("1", "x")], [1.0, 0.0], cache) == []


def test_semantic_search_respects_top_k(tmp_path: Path) -> None:
    events = [_event(str(i), f"Talk {i}") for i in range(5)]
    cache = EmbeddingCache(tmp_path / "embeddings.npz", model=DEFAULT_MODEL)
    for i in range(5):
        cache.put(str(i), np.array([1.0, float(i)], dtype=np.float32))
    results = semantic_search(events, [1.0, 2.0], cache, top_k=2)
    assert len(results) == 2


def test_coverage_counts_cached_vs_total(tmp_path: Path) -> None:
    e1 = _event("1", "Async")
    e2 = _event("2", "Rust")
    e3 = _event("3", "Memory")
    cache = EmbeddingCache(tmp_path / "embeddings.npz", model=DEFAULT_MODEL)
    cache.put("1", np.array([1.0, 0.0], dtype=np.float32))
    assert coverage(cache, [e1, e2, e3]) == (1, 3)
    cache.put("2", np.array([0.0, 1.0], dtype=np.float32))
    cache.put("3", np.array([1.0, 1.0], dtype=np.float32))
    assert coverage(cache, [e1, e2, e3]) == (3, 3)


def test_semantic_search_with_scores_returns_pairs(tmp_path: Path) -> None:
    e1 = _event("1", "Async patterns")
    e2 = _event("2", "Rust interop")
    cache = EmbeddingCache(tmp_path / "embeddings.npz", model=DEFAULT_MODEL)
    cache.put("1", np.array([1.0, 0.0], dtype=np.float32))
    cache.put("2", np.array([0.0, 1.0], dtype=np.float32))

    results = semantic_search_with_scores([e1, e2], [1.0, 0.0], cache)
    # Closest is e1 with cosine 1.0; farthest is e2 with cosine 0.0.
    assert [e.id for e, _ in results] == ["1", "2"]
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)
    assert results[1][1] == pytest.approx(0.0, abs=1e-5)


def test_filter_by_negative_phrases_drops_close_events(tmp_path: Path) -> None:
    """Events whose embedding is too close to a negative phrase get dropped."""
    e_python = _event("1", "Python intro")
    e_rust = _event("2", "Rust intro")
    cache = EmbeddingCache(tmp_path / "embeddings.npz", model=DEFAULT_MODEL)
    cache.put("1", np.array([1.0, 0.0], dtype=np.float32))  # 'python' direction
    cache.put("2", np.array([0.0, 1.0], dtype=np.float32))  # 'rust' direction

    # Negative phrase pointing in 'python' direction (cosine 1.0 with e_python).
    survivors = filter_by_negative_phrases([e_python, e_rust], [[1.0, 0.0]], cache, threshold=0.5)
    assert [e.id for e in survivors] == ["2"]


def test_filter_by_negative_phrases_keeps_uncached(tmp_path: Path) -> None:
    """An event without an embedding can't be judged, so it survives."""
    e_unknown = _event("1", "Unknown")
    cache = EmbeddingCache(tmp_path / "embeddings.npz", model=DEFAULT_MODEL)
    survivors = filter_by_negative_phrases([e_unknown], [[1.0, 0.0]], cache, threshold=0.5)
    assert survivors == [e_unknown]


def test_filter_by_negative_phrases_no_phrases_passes_through(tmp_path: Path) -> None:
    e1 = _event("1", "x")
    cache = EmbeddingCache(tmp_path / "embeddings.npz", model=DEFAULT_MODEL)
    assert filter_by_negative_phrases([e1], [], cache) == [e1]


def test_voyage_embedder_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from pycon_cal_scraper.semantic import VoyageEmbedder

    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY"):
        VoyageEmbedder()
