"""Tests for date filters and overlap detection."""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from pycon_cal_scraper.filters import (
    event_overlaps_any,
    events_overlap,
    filter_events_by_day,
    filter_events_by_room,
    filter_events_by_window,
    parse_day_token,
    parse_when,
)
from pycon_cal_scraper.models import Event, EventType

PACIFIC = ZoneInfo("America/Los_Angeles")


def _event(eid: str, hour_start: int, hour_end: int, day: date = date(2026, 5, 15)) -> Event:
    return Event.model_validate(
        {
            "id": eid,
            "url": f"https://us.pycon.org/2026/schedule/presentation/{eid}/",
            "title": f"Talk {eid}",
            "type": EventType.talk,
            "speakers": [],
            "start": datetime.combine(day, datetime.min.time(), tzinfo=PACIFIC).replace(
                hour=hour_start
            ),
            "end": datetime.combine(day, datetime.min.time(), tzinfo=PACIFIC).replace(
                hour=hour_end
            ),
            "room": None,
        }
    )


def test_parse_day_token_resolves_codes() -> None:
    assert parse_day_token("fri") == date(2026, 5, 15)
    assert parse_day_token("MON") == date(2026, 5, 18)


def test_parse_day_token_accepts_iso_date() -> None:
    assert parse_day_token("2026-05-15") == date(2026, 5, 15)


def test_parse_day_token_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="can't parse"):
        parse_day_token("xyz")


def test_parse_when_attaches_pacific_tz() -> None:
    dt = parse_when("2026-05-15T14:00")
    assert dt.tzinfo == PACIFIC
    assert dt.hour == 14


def test_parse_when_accepts_weekday_code() -> None:
    dt = parse_when("fri")
    assert dt.date() == date(2026, 5, 15)
    assert dt.hour == 0


def test_filter_events_by_window_keeps_overlapping() -> None:
    events = [_event("1", 9, 10), _event("2", 10, 11), _event("3", 13, 14)]
    start = datetime(2026, 5, 15, 10, 30, tzinfo=PACIFIC)
    end = datetime(2026, 5, 15, 11, 30, tzinfo=PACIFIC)
    # event "1" ends at 10:00 (before window start) — excluded.
    # event "2" 10-11 overlaps 10:30-11:30 - included.
    # event "3" starts at 13:00 (after window end) — excluded.
    assert [e.id for e in filter_events_by_window(events, start=start, end=end)] == ["2"]


def test_filter_events_by_window_unbounded() -> None:
    events = [_event("1", 9, 10), _event("2", 14, 15)]
    only_morning = filter_events_by_window(events, end=datetime(2026, 5, 15, 12, 0, tzinfo=PACIFIC))
    assert [e.id for e in only_morning] == ["1"]


def test_filter_events_by_day() -> None:
    fri = _event("1", 9, 10, day=date(2026, 5, 15))
    sat = _event("2", 9, 10, day=date(2026, 5, 16))
    assert filter_events_by_day([fri, sat], date(2026, 5, 15)) == [fri]


def test_events_overlap_basic_cases() -> None:
    a = _event("a", 9, 11)
    b = _event("b", 10, 12)
    c = _event("c", 11, 13)
    assert events_overlap(a, b)
    # Touching boundary is NOT an overlap.
    assert not events_overlap(a, c)


def test_event_overlaps_any_ignores_self() -> None:
    a = _event("a", 9, 11)
    assert not event_overlaps_any(a, [a])
    b = _event("b", 10, 12)
    assert event_overlaps_any(b, [a])


def test_event_overlaps_any_empty_others() -> None:
    a = _event("a", 9, 11)
    assert not event_overlaps_any(a, [])


def _event_with_room(eid: str, room: str | None) -> Event:
    return Event.model_validate(
        {
            "id": eid,
            "url": f"https://us.pycon.org/2026/schedule/presentation/{eid}/",
            "title": f"Talk {eid}",
            "type": EventType.talk,
            "speakers": [],
            "start": datetime(2026, 5, 15, 9, 0, tzinfo=PACIFIC),
            "end": datetime(2026, 5, 15, 10, 0, tzinfo=PACIFIC),
            "room": room,
        }
    )


def test_filter_events_by_room_substring_match() -> None:
    events = [
        _event_with_room("1", "Room 103ABC"),
        _event_with_room("2", "Grand Ballroom A"),
        _event_with_room("3", "Room 104AB"),
        _event_with_room("4", None),
    ]
    assert [e.id for e in filter_events_by_room(events, "104")] == ["3"]
    assert [e.id for e in filter_events_by_room(events, "ballroom")] == ["2"]


def test_filter_events_by_room_empty_needle_returns_all() -> None:
    events = [_event_with_room("1", "X"), _event_with_room("2", None)]
    assert filter_events_by_room(events, "") == events


def test_filter_events_by_room_skips_events_without_room() -> None:
    events = [_event_with_room("1", None)]
    assert filter_events_by_room(events, "anything") == []
