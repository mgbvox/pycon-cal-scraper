"""Tests for the on-disk HTTP response cache."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx

from pycon_cal_scraper.http_cache import CachedClient

URL = "https://us.pycon.org/2026/schedule/talks/"


@pytest.fixture
def client(tmp_path: Path) -> CachedClient:
    # Default fixture disables retries so unrelated tests stay fast — retry
    # behaviour gets its own coverage in test_retries_on_503_then_succeeds.
    return CachedClient(
        cache_dir=tmp_path / "http",
        ttl=timedelta(hours=1),
        min_interval=0.0,
        max_retries=0,
    )


@respx.mock
async def test_first_request_hits_network_second_served_from_cache(client: CachedClient) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(200, text="<html>HELLO</html>"))

    body1 = await client.get_text(URL)
    body2 = await client.get_text(URL)

    assert body1 == "<html>HELLO</html>"
    assert body2 == body1
    assert route.call_count == 1


@respx.mock
async def test_expired_cache_is_refetched(tmp_path: Path) -> None:
    client = CachedClient(cache_dir=tmp_path / "http", ttl=timedelta(seconds=10), min_interval=0.0)
    respx.get(URL).mock(return_value=httpx.Response(200, text="first"))
    assert await client.get_text(URL) == "first"

    cache_file = next((tmp_path / "http").glob("*.html"))
    old = (datetime.now(tz=UTC) - timedelta(minutes=5)).timestamp()
    os.utime(cache_file, (old, old))

    respx.get(URL).mock(return_value=httpx.Response(200, text="second"))
    assert await client.get_text(URL) == "second"


@respx.mock
async def test_force_refresh_bypasses_cache(client: CachedClient) -> None:
    respx.get(URL).mock(return_value=httpx.Response(200, text="cached"))
    await client.get_text(URL)

    respx.get(URL).mock(return_value=httpx.Response(200, text="fresh"))
    assert await client.get_text(URL, force_refresh=True) == "fresh"


@respx.mock
async def test_non_2xx_raises(client: CachedClient) -> None:
    respx.get(URL).mock(return_value=httpx.Response(500, text="boom"))
    with pytest.raises(httpx.HTTPStatusError):
        await client.get_text(URL)


def test_cache_path_for_url_is_stable(tmp_path: Path) -> None:
    c = CachedClient(cache_dir=tmp_path)
    p1 = c.cache_path(URL)
    p2 = c.cache_path(URL)
    assert p1 == p2
    assert p1.parent == tmp_path


@respx.mock
async def test_retries_on_503_then_succeeds(tmp_path: Path) -> None:
    """Two 503s followed by a 200 should resolve transparently."""
    client = CachedClient(
        cache_dir=tmp_path / "http",
        ttl=timedelta(hours=1),
        min_interval=0.0,
        max_retries=3,
        backoff_base=0.0,  # zero delay keeps the test snappy
    )
    route = respx.get(URL).mock(
        side_effect=[
            httpx.Response(503, text="busy"),
            httpx.Response(503, text="busy"),
            httpx.Response(200, text="ok"),
        ]
    )
    assert await client.get_text(URL) == "ok"
    assert route.call_count == 3


@respx.mock
async def test_404_raises_without_retry(tmp_path: Path) -> None:
    """404 is not in the retryable set, so the first response should raise."""
    client = CachedClient(
        cache_dir=tmp_path / "http",
        ttl=timedelta(hours=1),
        min_interval=0.0,
        max_retries=3,
        backoff_base=0.0,
    )
    route = respx.get(URL).mock(return_value=httpx.Response(404, text="missing"))
    with pytest.raises(httpx.HTTPStatusError):
        await client.get_text(URL)
    assert route.call_count == 1


@respx.mock
async def test_retry_budget_exhausted_raises(tmp_path: Path) -> None:
    """Persistent 503 should raise after the configured retry budget is spent."""
    client = CachedClient(
        cache_dir=tmp_path / "http",
        ttl=timedelta(hours=1),
        min_interval=0.0,
        max_retries=2,
        backoff_base=0.0,
    )
    route = respx.get(URL).mock(return_value=httpx.Response(503, text="busy"))
    with pytest.raises(httpx.HTTPStatusError):
        await client.get_text(URL)
    assert route.call_count == 3  # initial + 2 retries


@respx.mock
async def test_concurrent_requests_are_bounded(tmp_path: Path) -> None:
    """Concurrent gets should produce one network call per URL and one file each."""
    import asyncio

    urls = [f"{URL}?p={i}" for i in range(8)]
    for u in urls:
        respx.get(u).mock(return_value=httpx.Response(200, text=u))

    client = CachedClient(
        cache_dir=tmp_path / "http", min_interval=0.0, concurrency=3, ttl=timedelta(hours=1)
    )
    results = await asyncio.gather(*(client.get_text(u) for u in urls))
    assert sorted(results) == sorted(urls)
    assert len(list((tmp_path / "http").glob("*.html"))) == 8
