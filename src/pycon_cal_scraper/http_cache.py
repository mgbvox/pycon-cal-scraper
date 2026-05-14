"""Polite, on-disk-cached async HTTP client used by the scraper.

The :class:`CachedClient` wraps :class:`httpx.AsyncClient` with three
features the scraper relies on:

* On-disk caching keyed by SHA-256 of the URL, with a TTL.
* A minimum delay between *live* requests to be a good citizen.
* A concurrency semaphore so ``asyncio.gather`` over hundreds of URLs
  doesn't open hundreds of sockets at once.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType

import httpx

DEFAULT_USER_AGENT = "pycon-cal-scraper/0.1 (+https://us.pycon.org/2026/)"
DEFAULT_TTL = timedelta(hours=24)
DEFAULT_MIN_INTERVAL = 0.25  # seconds between *live* requests
DEFAULT_CONCURRENCY = 5


def _now() -> datetime:
    """Return the current UTC time; isolated so tests can monkey-patch it."""
    return datetime.now(tz=UTC)


class CachedClient:
    """Async HTTP client with an on-disk response cache.

    Cache keys are SHA-256 hashes of the request URL; fresh entries are read
    straight from disk and never touch the network. Live requests are throttled
    by ``min_interval`` between calls and bounded by a ``concurrency``
    semaphore — fast enough to fetch hundreds of presentation pages in
    seconds without hammering the conference server.

    The instance is an async context manager that closes the underlying
    ``httpx.AsyncClient`` on exit.
    """

    def __init__(
        self,
        cache_dir: Path,
        ttl: timedelta = DEFAULT_TTL,
        min_interval: float = DEFAULT_MIN_INTERVAL,
        concurrency: int = DEFAULT_CONCURRENCY,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 30.0,
    ) -> None:
        """Build a new client.

        Args:
            cache_dir: Directory used to store cached response bodies. Created
                if missing.
            ttl: How long a cached entry is considered fresh.
            min_interval: Minimum number of seconds between two live requests.
                Set to ``0`` to disable throttling (used in tests).
            concurrency: Maximum number of in-flight live requests.
            user_agent: Value of the ``User-Agent`` header.
            timeout: Per-request timeout passed to :class:`httpx.AsyncClient`.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        self.min_interval = min_interval
        self._semaphore = asyncio.Semaphore(concurrency)
        self._gate_lock = asyncio.Lock()
        self._next_allowed_at = 0.0
        self._client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": user_agent},
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> CachedClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    def cache_path(self, url: str) -> Path:
        """Return the on-disk path used to cache ``url``.

        Args:
            url: The URL whose cache file path you want.

        Returns:
            A path inside :attr:`cache_dir` derived from
            ``sha256(url)``; existence is not checked.
        """
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.html"

    def _is_fresh(self, path: Path) -> bool:
        """Return ``True`` if ``path`` exists and is younger than :attr:`ttl`."""
        if not path.exists():
            return False
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        return _now() - mtime < self.ttl

    async def _wait_throttle(self) -> None:
        """Sleep just long enough to honour :attr:`min_interval`."""
        if self.min_interval <= 0:
            return
        async with self._gate_lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait = self._next_allowed_at - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = loop.time()
            self._next_allowed_at = now + self.min_interval

    async def get_text(self, url: str, *, force_refresh: bool = False) -> str:
        """Fetch ``url`` as text, using the disk cache when possible.

        Args:
            url: Absolute URL to fetch.
            force_refresh: When ``True``, ignore any cached entry and re-fetch.

        Returns:
            The response body as text.

        Raises:
            httpx.HTTPStatusError: If the response is not 2xx.
        """
        path = self.cache_path(url)
        if not force_refresh and self._is_fresh(path):
            return path.read_text(encoding="utf-8")
        async with self._semaphore:
            await self._wait_throttle()
            response = await self._client.get(url)
            response.raise_for_status()
        path.write_text(response.text, encoding="utf-8")
        return response.text
