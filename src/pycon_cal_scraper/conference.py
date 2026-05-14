"""Conference-specific constants — the one place that needs to change each year.

Every URL, weekday code, and default venue derives from values in this module.
When PyCon US 2027 rolls around, edit :data:`CONFERENCE_YEAR`, refresh
:data:`CONFERENCE_DAYS`, and update :data:`CONFERENCE_VENUE_DEFAULT` if the
venue moves — nothing else should need touching.
"""

from __future__ import annotations

from datetime import date
from zoneinfo import ZoneInfo

#: Year of the conference whose schedule the scraper targets.
CONFERENCE_YEAR: int = 2026

#: Timezone the conference operates in (used for day-boundary and ICS output).
CONFERENCE_TZ: ZoneInfo = ZoneInfo("America/Los_Angeles")

#: Three-letter weekday codes accepted by :func:`filters.parse_day_token`.
#: PyCon US 2026 runs Wednesday May 13 (tutorials) through Tuesday May 19
#: (sprints) — verify against the schedule when bumping :data:`CONFERENCE_YEAR`.
CONFERENCE_DAYS: dict[str, date] = {
    "wed": date(2026, 5, 13),
    "thu": date(2026, 5, 14),
    "fri": date(2026, 5, 15),
    "sat": date(2026, 5, 16),
    "sun": date(2026, 5, 17),
    "mon": date(2026, 5, 18),
    "tue": date(2026, 5, 19),
}

#: Default postal address of the conference venue. Used both as a config
#: default and as a fallback in ICS export.
CONFERENCE_VENUE_DEFAULT: str = (
    "Long Beach Convention Center, 300 East Ocean Boulevard, Long Beach, CA 90802"
)

#: User-Agent string the scraper sends. Identifies the tool and points back
#: at the conference site so curious admins can find this project.
CONFERENCE_USER_AGENT: str = f"pycon-cal-scraper/0.1 (+https://us.pycon.org/{CONFERENCE_YEAR}/)"

#: Root URL of the PyCon US site. Override in :class:`UserConfig` when
#: mirroring for offline testing.
CONFERENCE_BASE_URL: str = "https://us.pycon.org"

#: Schedule list pages, relative to :data:`CONFERENCE_BASE_URL`.
SCHEDULE_PATHS: tuple[str, ...] = (
    f"/{CONFERENCE_YEAR}/schedule/talks/",
    f"/{CONFERENCE_YEAR}/schedule/tutorials/",
    f"/{CONFERENCE_YEAR}/schedule/sponsor-presentations/",
)

#: URL fragment used to build per-event detail URLs.
PRESENTATION_PATH_TEMPLATE: str = f"/{CONFERENCE_YEAR}/schedule/presentation/{{event_id}}/"
