"""Tests for path resolution and environment overrides."""

from __future__ import annotations

from pathlib import Path

from pycon_cal_scraper import paths


def test_paths_default_locations(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PYCON_CAL_DATA_DIR", raising=False)
    monkeypatch.delenv("PYCON_CAL_CACHE_DIR", raising=False)
    monkeypatch.delenv("PYCON_CAL_CONFIG_DIR", raising=False)

    data = paths.data_dir()
    cache = paths.cache_dir()
    config = paths.config_dir()

    assert data.is_dir()
    assert cache.is_dir()
    assert config.is_dir()
    # Each directory should belong to the platform's user space and have our app name in it.
    assert "pycon-cal-scraper" in str(data).lower()


def test_paths_env_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PYCON_CAL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PYCON_CAL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("PYCON_CAL_CONFIG_DIR", str(tmp_path / "config"))

    assert paths.data_dir() == tmp_path / "data"
    assert paths.cache_dir() == tmp_path / "cache"
    assert paths.config_dir() == tmp_path / "config"
    # Directories must be created on access.
    assert paths.data_dir().is_dir()
    assert paths.cache_dir().is_dir()
    assert paths.config_dir().is_dir()


def test_derived_paths(tmp_data_dir: Path) -> None:
    assert paths.events_file().name == "events.json"
    assert paths.saved_file().name == "saved.json"
    assert paths.token_file().name == "token.json"
    assert paths.config_file().name == "config.json"
    assert paths.http_cache_dir().name == "http"
    # All derived paths should sit under the env-overridden roots.
    assert paths.events_file().parent == paths.data_dir()
    assert paths.http_cache_dir().parent == paths.cache_dir()
