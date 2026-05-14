"""User configuration: a small JSON file under :func:`paths.config_dir`."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from pycon_cal_scraper.conference import (
    CONFERENCE_BASE_URL,
    CONFERENCE_USER_AGENT,
    CONFERENCE_VENUE_DEFAULT,
)
from pycon_cal_scraper.paths import config_file


class UserConfig(BaseModel):
    """Persistent user configuration.

    Every default here is what the corresponding module constant ships
    with — set a field via ``pycon-cal-scraper config set <key> <value>``
    to override it without editing source.

    Attributes:
        calendar_id: The Google Calendar id to sync to. ``"primary"`` is
            the user's default calendar.
        client_secret_path: Filesystem path to the user's
            ``client_secret.json`` (Desktop OAuth client). Required for
            ``gcal login``.
        default_tz: IANA timezone to use when the schedule page doesn't
            carry one explicitly.
        search_results_limit: Maximum number of results the ``search``
            command and the REPL render in a single match table. Must be
            at least 1.
        embedding_model: Voyage AI model name used to embed events for
            semantic search. Changing this invalidates the cached
            embeddings on the next ``embed`` run.
        embedding_batch_size: Number of documents per Voyage embedding
            API call. Tune up for larger calendars, down on rate-limited
            tiers.
        voyage_api_key_env: Name of the environment variable that holds
            the Voyage AI API key. Defaults to ``VOYAGE_API_KEY``.
        scraper_base_url: Root URL the scraper walks. Override only when
            mirroring the conference site for offline testing.
        http_user_agent: ``User-Agent`` header sent on every scrape
            request. Be polite — the default identifies this tool.
        http_cache_ttl_hours: How long a cached HTTP response stays fresh
            before the scraper re-fetches the page.
        http_min_interval_seconds: Minimum delay between two *live* HTTP
            requests, in seconds. ``0`` disables throttling.
        http_concurrency: Maximum number of in-flight live HTTP requests.
        search_weight_title: Lexical search weight for title hits.
        search_weight_speaker: Lexical search weight for speaker hits.
        search_weight_abstract: Lexical search weight for abstract hits.
        semantic_negative_threshold: Cosine similarity at/above which a
            ``!"phrase"`` negative excludes an event.
        venue_address: Postal address of the conference venue. Used as
            the base of every Google Calendar event's ``location`` (with
            the event's room prepended), so taps in Google Calendar open
            Maps at the right building.
    """

    model_config = ConfigDict(extra="forbid")

    calendar_id: str = "primary"
    client_secret_path: str | None = None
    default_tz: str = Field(default="America/Los_Angeles")
    search_results_limit: int = Field(default=20, ge=1)
    embedding_model: str = Field(default="voyage-3-lite")
    embedding_batch_size: int = Field(default=64, ge=1)
    voyage_api_key_env: str = Field(default="VOYAGE_API_KEY")
    scraper_base_url: str = Field(default=CONFERENCE_BASE_URL)
    http_user_agent: str = Field(default=CONFERENCE_USER_AGENT)
    http_cache_ttl_hours: float = Field(default=24.0, gt=0)
    http_min_interval_seconds: float = Field(default=0.25, ge=0)
    http_concurrency: int = Field(default=5, ge=1)
    search_weight_title: int = Field(default=4, ge=0)
    search_weight_speaker: int = Field(default=2, ge=0)
    search_weight_abstract: int = Field(default=1, ge=0)
    semantic_negative_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    venue_address: str = Field(default=CONFERENCE_VENUE_DEFAULT)


def load_config(path: Path | None = None) -> UserConfig:
    """Read the user config, or return defaults if no file exists.

    Args:
        path: Override the default config-file path (used by tests).

    Returns:
        A :class:`UserConfig` instance.
    """
    target = path or config_file()
    if not target.exists():
        return UserConfig()
    return UserConfig.model_validate_json(target.read_text(encoding="utf-8"))


def save_config(cfg: UserConfig, path: Path | None = None) -> None:
    """Persist ``cfg`` to disk as JSON.

    Args:
        cfg: The configuration to persist.
        path: Override the default config-file path.
    """
    target = path or config_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cfg.model_dump(), indent=2), encoding="utf-8")
