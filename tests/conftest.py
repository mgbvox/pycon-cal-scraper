"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_dir() -> Path:
    """Directory containing static HTML fixtures captured from the live PyCon site."""
    return FIXTURE_DIR


@pytest.fixture
def read_fixture() -> callable[[str], str]:  # type: ignore[type-arg]
    """Return a callable that reads a named fixture file as text."""

    def _read(name: str) -> str:
        return (FIXTURE_DIR / name).read_text(encoding="utf-8")

    return _read


@pytest.fixture
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Redirect all on-disk paths used by the library to a tmp directory."""
    monkeypatch.setenv("PYCON_CAL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PYCON_CAL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("PYCON_CAL_CONFIG_DIR", str(tmp_path / "config"))
    yield tmp_path


@pytest.fixture
def fixed_now(monkeypatch: pytest.MonkeyPatch) -> datetime:
    """A deterministic 'now' for time-sensitive tests."""
    fixed = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)

    class _Clock:
        @staticmethod
        def now(tz: object = None) -> datetime:
            return fixed if tz is None else fixed.astimezone(tz)  # type: ignore[arg-type]

    monkeypatch.setattr("pycon_cal_scraper.http_cache._now", _Clock.now, raising=False)
    return fixed
