"""Date/time-range filters and overlap detection for event lists."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from pycon_cal_scraper.conference import CONFERENCE_DAYS, CONFERENCE_TZ
from pycon_cal_scraper.models import Event

#: Three-letter weekday codes accepted by :func:`parse_day_token`. The map
#: itself lives in :mod:`pycon_cal_scraper.conference` so the year-rollover
#: edit is centralized.
_DAY_CODES: dict[str, date] = CONFERENCE_DAYS

#: Default IANA timezone used when a date/datetime input has no tz set. Lives
#: under :data:`pycon_cal_scraper.conference.CONFERENCE_TZ` so the year
#: rollover changes only one file.
DEFAULT_TZ = CONFERENCE_TZ


def parse_day_token(token: str) -> date:
    """Resolve a short weekday code to a PyCon US 2026 date.

    Args:
        token: A case-insensitive three-letter code (``mon``..``sun``) or a
            full ISO date (``2026-05-15``).

    Returns:
        The corresponding :class:`date`.

    Raises:
        ValueError: If the token doesn't parse.
    """
    cleaned = token.strip().lower()
    if cleaned in _DAY_CODES:
        return _DAY_CODES[cleaned]
    try:
        return date.fromisoformat(token)
    except ValueError as exc:
        raise ValueError(
            f"can't parse day {token!r}; expected one of {sorted(_DAY_CODES)} or YYYY-MM-DD"
        ) from exc


def parse_when(token: str, *, tz: ZoneInfo = DEFAULT_TZ) -> datetime:
    """Parse a date or datetime string into a timezone-aware :class:`datetime`.

    Args:
        token: ``YYYY-MM-DD`` (midnight in ``tz``) or ``YYYY-MM-DDTHH:MM``
            ISO datetime. Also accepts the same weekday codes as
            :func:`parse_day_token`.
        tz: Timezone to attach when the token has no offset.

    Returns:
        A timezone-aware datetime.
    """
    try:
        d = parse_day_token(token)
        return datetime.combine(d, time.min, tzinfo=tz)
    except ValueError:
        pass
    dt = datetime.fromisoformat(token)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt


def filter_events_by_window(
    events: Iterable[Event],
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[Event]:
    """Return only events that fall within ``[start, end)``.

    A None bound means "no constraint on that side". Events whose interval
    overlaps the window are kept (so a 9-noon talk passes a 10-11am filter).

    Args:
        events: Candidate events.
        start: Earliest acceptable event end. Events ending before this are
            excluded.
        end: Latest acceptable event start. Events starting after this are
            excluded.

    Returns:
        The filtered list, preserving input order.
    """
    filtered: list[Event] = []
    for e in events:
        if start is not None and e.end <= start:
            continue
        if end is not None and e.start >= end:
            continue
        filtered.append(e)
    return filtered


def filter_events_by_day(events: Iterable[Event], day: date) -> list[Event]:
    """Return only events whose start lands on ``day`` (Pacific time)."""
    return [e for e in events if e.start.astimezone(DEFAULT_TZ).date() == day]


def filter_events_by_room(events: Iterable[Event], needle: str) -> list[Event]:
    """Return events whose ``room`` contains ``needle`` (case-insensitive substring).

    Args:
        events: Candidate events.
        needle: Substring to look for in :attr:`Event.room`. Pass ``""`` to
            keep every event regardless of room.

    Returns:
        Matching events, preserving input order. Events with no room are
        always excluded when ``needle`` is non-empty.
    """
    if not needle:
        return list(events)
    needle_l = needle.lower()
    return [e for e in events if e.room and needle_l in e.room.lower()]


def events_overlap(a: Event, b: Event) -> bool:
    """Return ``True`` iff ``a`` and ``b`` share any time.

    A zero-length boundary (e.g. one event ending exactly when the other
    starts) does not count as an overlap.
    """
    return a.start < b.end and b.start < a.end


def event_overlaps_any(event: Event, others: Iterable[Event]) -> bool:
    """Return ``True`` iff ``event`` overlaps with any event in ``others``.

    The check skips ``event`` itself (matched by id) so an event isn't
    flagged as overlapping with itself.
    """
    return any(other.id != event.id and events_overlap(event, other) for other in others)


def conflict_window(
    saved: Iterable[Event], *, padding: timedelta = timedelta(days=1)
) -> tuple[datetime, datetime] | None:
    """Return ``(min(starts) - padding, max(ends) + padding)`` for ``saved``."""
    saved_list = list(saved)
    if not saved_list:
        return None
    return (
        min(e.start for e in saved_list) - padding,
        max(e.end for e in saved_list) + padding,
    )


def events_happening_at(events: Iterable[Event], at: datetime) -> list[Event]:
    """Return events whose interval covers ``at``.

    Ties are not broken — every event with ``start <= at < end`` is returned
    in input order. Callers wanting a single "what should I be in" usually
    pair this with :func:`upcoming_events` to handle the empty case.
    """
    return [e for e in events if e.start <= at < e.end]


def upcoming_events(events: Iterable[Event], at: datetime, *, limit: int = 5) -> list[Event]:
    """Return up to ``limit`` events starting strictly after ``at``.

    Sorted by start time ascending. ``limit=0`` returns an empty list.
    """
    if limit <= 0:
        return []
    future = sorted((e for e in events if e.start > at), key=lambda e: e.start)
    return future[:limit]


def find_conflict_groups(events: Iterable[Event]) -> list[list[Event]]:
    """Group events into connected components of pairwise time-overlap.

    Two events that share any time end up in the same cluster; clusters
    grow transitively, so three events ``A`` 9-10, ``B`` 9:30-10:30,
    ``C`` 10:15-11 all land in one group even though ``A`` and ``C`` don't
    touch directly. Singletons (events with no overlap) are omitted —
    callers want only actual conflicts.

    Args:
        events: Candidate events.

    Returns:
        A list of clusters, each cluster sorted by start time, and the
        clusters themselves sorted by the earliest start within each.
    """
    items = sorted(events, key=lambda ev: ev.start)
    n = len(items)
    if n < 2:
        return []
    parent = list(range(n))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    for i in range(n):
        for j in range(i + 1, n):
            # Sorted by start, so once j's start passes i's end we can stop.
            if items[j].start >= items[i].end:
                break
            if events_overlap(items[i], items[j]):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups: dict[int, list[Event]] = {}
    for idx, ev in enumerate(items):
        groups.setdefault(find(idx), []).append(ev)
    result = [g for g in groups.values() if len(g) >= 2]
    result.sort(key=lambda g: g[0].start)
    return result
