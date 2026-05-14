"""Filesystem locations used by pycon-cal-scraper.

Every directory is resolved on demand and created if missing. Each function
honours a corresponding ``PYCON_CAL_*_DIR`` environment variable so tests can
redirect the entire I/O surface to a temporary directory.
"""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_cache_dir, user_config_dir, user_data_dir

APP_NAME = "pycon-cal-scraper"


def _resolve(env_var: str, default: Path) -> Path:
    """Resolve a directory path, honouring an env-var override.

    Args:
        env_var: Name of the environment variable to consult first.
        default: Fallback path (typically a platformdirs default).

    Returns:
        The chosen directory. Created on disk if it didn't already exist.
    """
    raw = os.environ.get(env_var)
    target = Path(raw) if raw else default
    target.mkdir(parents=True, exist_ok=True)
    return target


def data_dir() -> Path:
    """Return the directory for persistent data (scraped events, saved list)."""
    return _resolve("PYCON_CAL_DATA_DIR", Path(user_data_dir(APP_NAME)))


def cache_dir() -> Path:
    """Return the directory for disposable caches (HTTP response cache)."""
    return _resolve("PYCON_CAL_CACHE_DIR", Path(user_cache_dir(APP_NAME)))


def config_dir() -> Path:
    """Return the directory for user configuration and the OAuth token."""
    return _resolve("PYCON_CAL_CONFIG_DIR", Path(user_config_dir(APP_NAME)))


def events_file() -> Path:
    """Return the path to ``events.json`` (scraped event cache)."""
    return data_dir() / "events.json"


def saved_file() -> Path:
    """Return the path to ``saved.json`` (ordered list of saved event ids)."""
    return data_dir() / "saved.json"


def embeddings_file() -> Path:
    """Return the path to the on-disk embedding cache (``embeddings.npz``)."""
    return data_dir() / "embeddings.npz"


def http_cache_dir() -> Path:
    """Return the directory used for cached HTTP response bodies."""
    target = cache_dir() / "http"
    target.mkdir(parents=True, exist_ok=True)
    return target


def token_file() -> Path:
    """Return the path to the cached Google OAuth token."""
    return config_dir() / "token.json"


def config_file() -> Path:
    """Return the path to the user-config JSON file."""
    return config_dir() / "config.json"


def main():
    print(cache_dir())


if __name__ == "__main__":
    main()
