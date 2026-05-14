"""Semantic search over scraped events using Voyage AI embeddings.

Anthropic does not ship a first-party embeddings endpoint; their docs
explicitly recommend Voyage AI as the embeddings backend for Claude-based
projects. This module:

* Renders each :class:`Event` into a short, searchable document
  (:func:`event_to_text`).
* Calls Voyage to embed those documents and the user's query.
* Persists the document vectors next to ``events.json`` as a ``.npz`` so
  embedding cost is paid once per event.
* Ranks events at query time with cosine similarity computed in-memory.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from pycon_cal_scraper.models import Event

DEFAULT_MODEL = "voyage-3-lite"
DEFAULT_BATCH_SIZE = 64
EMBED_API_KEY_ENV = "VOYAGE_API_KEY"

ProgressCallback = Callable[[int, int], None]


class Embedder(Protocol):
    """Minimal interface an embedding backend must expose."""

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single user-supplied query."""
        ...


def event_to_text(event: Event) -> str:
    """Render an :class:`Event` into a single string for embedding.

    Args:
        event: The event to render.

    Returns:
        A multi-line plain-text representation including title, speakers,
        track, audience level, and abstract (when present). The title is
        repeated at the bottom so short keyword queries (e.g. ``"rust"``)
        get a stronger signal when the keyword sits in the title.
    """
    parts: list[str] = [f"Title: {event.title}", f"Type: {event.type.value}"]
    if event.speakers:
        parts.append(f"Speakers: {', '.join(event.speakers)}")
    if event.track:
        parts.append(f"Track: {event.track}")
    if event.audience_level:
        parts.append(f"Audience level: {event.audience_level}")
    body = event.abstract or event.description
    if body:
        parts.append(f"Abstract: {body}")
    parts.append(event.title)
    return "\n".join(parts)


class VoyageEmbedder:
    """Concrete :class:`Embedder` backed by the Voyage AI async API.

    The API key is taken from the ``VOYAGE_API_KEY`` environment variable.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        api_key_env: str = EMBED_API_KEY_ENV,
    ) -> None:
        """Build a new embedder.

        Args:
            model: The Voyage model name (e.g. ``"voyage-3-lite"``).
            api_key: Optional API key; falls back to ``api_key_env``.
            api_key_env: Name of the environment variable holding the
                API key when ``api_key`` is not passed explicitly.

        Raises:
            RuntimeError: If no API key is available.
        """
        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(
                f"No Voyage API key. Set the {api_key_env} environment variable or pass api_key=..."
            )
        # Import here so test code can monkey-patch without paying the import cost
        # in environments that don't need it.
        import voyageai

        self.model = model
        self._client = voyageai.AsyncClient(api_key=key)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        response = await self._client.embed(list(texts), model=self.model, input_type="document")
        return [[float(x) for x in vec] for vec in response.embeddings]

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single user-supplied query string."""
        response = await self._client.embed([text], model=self.model, input_type="query")
        return [float(x) for x in response.embeddings[0]]


class EmbeddingCache:
    """On-disk cache of event-id -> embedding vector.

    The store is a single ``.npz`` with three arrays:

    * ``ids`` (1-D string array of event ids)
    * ``vectors`` (2-D float32 array, ``N x D``)
    * ``model`` (1-element string array recording which model produced
      the vectors)

    Attributes:
        path: The underlying ``.npz`` file path.
        model: The embedding model name. A cache loaded with a different
            ``model`` than what's stored is silently discarded.
    """

    def __init__(self, path: Path, model: str) -> None:
        """Build an empty cache rooted at ``path`` for the given ``model``."""
        self.path = Path(path)
        self.model = model
        self._vectors: dict[str, np.ndarray] = {}

    @classmethod
    def load(cls, path: Path, model: str) -> EmbeddingCache:
        """Load an existing cache or return an empty one.

        Args:
            path: Location of the ``.npz`` file.
            model: The model name we *want*. If the cached model differs,
                the cached vectors are discarded.

        Returns:
            An :class:`EmbeddingCache`.
        """
        cache = cls(path, model)
        if not cache.path.exists():
            return cache
        try:
            data = np.load(cache.path, allow_pickle=False)
        except OSError, ValueError:
            return cache
        stored_model = str(data["model"].item()) if "model" in data.files else ""
        if stored_model != model:
            return cache
        ids = data["ids"]
        vectors = data["vectors"]
        for i, eid in enumerate(ids):
            cache._vectors[str(eid)] = vectors[i].astype(np.float32, copy=False)
        return cache

    def save(self) -> None:
        """Persist the in-memory cache to disk atomically."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ids = np.asarray(list(self._vectors.keys()), dtype=np.str_)
        if self._vectors:
            vectors = np.stack(list(self._vectors.values())).astype(np.float32, copy=False)
        else:
            # Zero-row matrix; second dim is unknown until something is embedded.
            vectors = np.zeros((0, 0), dtype=np.float32)
        # np.savez appends ".npz" if the filename lacks it, so write into a file
        # handle to keep the staging path exact, then atomically rename.
        tmp = self.path.with_name(self.path.name + ".tmp")
        with tmp.open("wb") as fh:
            np.savez(fh, ids=ids, vectors=vectors, model=np.asarray(self.model))
        tmp.replace(self.path)

    def get(self, event_id: str) -> np.ndarray | None:
        """Return the embedding for ``event_id`` or ``None`` if uncached."""
        return self._vectors.get(event_id)

    def put(self, event_id: str, vector: np.ndarray) -> None:
        """Store ``vector`` under ``event_id``."""
        self._vectors[event_id] = vector.astype(np.float32, copy=False)

    def __contains__(self, event_id: object) -> bool:
        return isinstance(event_id, str) and event_id in self._vectors

    def __len__(self) -> int:
        return len(self._vectors)

    def known_ids(self) -> set[str]:
        """Return every event id present in the cache."""
        return set(self._vectors.keys())


async def embed_events(
    events: Sequence[Event],
    embedder: Embedder,
    cache: EmbeddingCache,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    on_progress: ProgressCallback | None = None,
) -> EmbeddingCache:
    """Compute embeddings for any event not yet in ``cache`` and persist.

    Args:
        events: The events to embed.
        embedder: The embedding backend.
        cache: Existing cache; updated in place and saved to disk.
        batch_size: Number of documents per embedding API call.
        on_progress: Optional ``(completed, total)`` callback fired before
            work begins (``0/N``) and after each batch.

    Returns:
        The (possibly updated) cache, also written to ``cache.path``.
    """
    missing = [e for e in events if e.id not in cache]
    total = len(missing)
    if on_progress is not None:
        on_progress(0, total)
    if total == 0:
        return cache

    completed = 0
    for start in range(0, total, batch_size):
        batch = missing[start : start + batch_size]
        texts = [event_to_text(e) for e in batch]
        vectors = await embedder.embed_documents(texts)
        for event, vec in zip(batch, vectors, strict=True):
            cache.put(event.id, np.asarray(vec, dtype=np.float32))
        completed += len(batch)
        if on_progress is not None:
            on_progress(completed, total)

    cache.save()
    return cache


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-normalize ``matrix`` for cosine similarity. Zero rows pass through."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def coverage(cache: EmbeddingCache, events: Iterable[Event]) -> tuple[int, int]:
    """Return ``(cached, total)`` counts for ``events`` against ``cache``.

    Args:
        cache: The embedding cache.
        events: The candidate events.

    Returns:
        How many of ``events`` have a cached vector, and how many were
        considered. ``cached < total`` means a re-embed is needed.
    """
    total = 0
    cached = 0
    for e in events:
        total += 1
        if e.id in cache:
            cached += 1
    return cached, total


def semantic_search_with_scores(
    events: Iterable[Event],
    query_vector: Sequence[float],
    cache: EmbeddingCache,
    *,
    top_k: int = 20,
) -> list[tuple[Event, float]]:
    """Rank ``events`` against ``query_vector`` by cosine similarity.

    Args:
        events: The candidate events.
        query_vector: Embedding of the user's query.
        cache: The document-embedding cache.
        top_k: Maximum number of events to return.

    Returns:
        ``(event, similarity)`` pairs, best first, with similarity in the
        range ``[-1, 1]``. Events not present in the cache are silently
        skipped — call :func:`coverage` first if you need to detect that.
    """
    by_id = {e.id: e for e in events}
    ids: list[str] = []
    rows: list[np.ndarray] = []
    for event_id in by_id:
        vec = cache.get(event_id)
        if vec is None:
            continue
        ids.append(event_id)
        rows.append(vec)
    if not rows:
        return []
    doc_matrix = _normalize(np.stack(rows))
    q = _normalize(np.asarray(query_vector, dtype=np.float32)[np.newaxis, :])[0]
    scores = doc_matrix @ q
    order = np.argsort(-scores)[:top_k]
    return [(by_id[ids[int(i)]], float(scores[int(i)])) for i in order]


def filter_by_negative_phrases(
    events: Iterable[Event],
    phrase_vectors: Sequence[Sequence[float]],
    cache: EmbeddingCache,
    *,
    threshold: float = 0.5,
) -> list[Event]:
    """Drop events whose embedding is too close to any negative-phrase vector.

    Args:
        events: Candidate events.
        phrase_vectors: One embedding per negative phrase.
        cache: The document-embedding cache.
        threshold: Cosine similarity at or above which an event is excluded.
            Defaults to ``0.5`` — tighten (raise) for less aggressive
            culling, loosen (lower) to cull harder.

    Returns:
        Events whose maximum cosine similarity against any phrase vector
        stays below ``threshold``. Events not present in the cache are
        always kept (we have no basis to judge them).
    """
    candidates = list(events)
    if not phrase_vectors or not candidates:
        return candidates
    neg_matrix = _normalize(np.asarray(phrase_vectors, dtype=np.float32))
    keep: list[Event] = []
    for event in candidates:
        vec = cache.get(event.id)
        if vec is None:
            keep.append(event)
            continue
        normed = _normalize(np.asarray(vec, dtype=np.float32)[np.newaxis, :])[0]
        # max cosine similarity between this event and any negative phrase
        sims = neg_matrix @ normed
        if float(sims.max()) < threshold:
            keep.append(event)
    return keep


def semantic_search(
    events: Iterable[Event],
    query_vector: Sequence[float],
    cache: EmbeddingCache,
    *,
    top_k: int = 20,
) -> list[Event]:
    """Rank ``events`` against ``query_vector`` by cosine similarity.

    Thin wrapper around :func:`semantic_search_with_scores` that drops the
    similarity scores. Prefer the scored variant when you want to render
    or threshold by similarity.
    """
    return [
        event for event, _ in semantic_search_with_scores(events, query_vector, cache, top_k=top_k)
    ]
