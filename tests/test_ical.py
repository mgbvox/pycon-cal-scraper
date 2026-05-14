"""Tests for the lightweight VCALENDAR serializer."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from pycon_cal_scraper.ical import event_to_vevent, events_to_ics
from pycon_cal_scraper.models import Event, EventType

PACIFIC = ZoneInfo("America/Los_Angeles")


def _event(eid: str, title: str = "Async patterns") -> Event:
    return Event.model_validate(
        {
            "id": eid,
            "url": f"https://us.pycon.org/2026/schedule/presentation/{eid}/",
            "title": title,
            "type": EventType.talk,
            "speakers": ["Pat Speaker"],
            "start": datetime(2026, 5, 15, 9, 0, tzinfo=PACIFIC),
            "end": datetime(2026, 5, 15, 9, 30, tzinfo=PACIFIC),
            "room": "Room 1",
            "description": "A talk about async patterns.",
        }
    )


def test_events_to_ics_wraps_in_vcalendar() -> None:
    out = events_to_ics([_event("1")], venue_address="Long Beach")
    assert out.startswith("BEGIN:VCALENDAR")
    assert out.rstrip().endswith("END:VCALENDAR")
    assert "VERSION:2.0" in out


def test_event_to_vevent_includes_tzid_and_pycon_id() -> None:
    block = event_to_vevent(_event("42"), venue_address="Long Beach")
    assert "BEGIN:VEVENT" in block
    assert "DTSTART;TZID=America/Los_Angeles:20260515T090000" in block
    assert "DTEND;TZID=America/Los_Angeles:20260515T093000" in block
    assert "UID:pycon-42@pycon-cal-scraper" in block
    assert "X-PYCON-ID:42" in block
    assert "SUMMARY:Async patterns" in block


def test_event_to_vevent_escapes_special_characters() -> None:
    bumpy = Event.model_validate(
        {
            **_event("1").model_dump(mode="json"),
            "title": "Commas, semis; and \\backslashes",
            "description": "Line one\nLine two",
        }
    )
    block = event_to_vevent(bumpy)
    assert "SUMMARY:Commas\\, semis\\; and \\\\backslashes" in block
    # Newlines collapse to the escaped sequence in TEXT properties.
    assert "Line one\\nLine two" in block


def test_event_to_vevent_uses_combined_location() -> None:
    block = event_to_vevent(_event("1"), venue_address="LBCC, 300 East Ocean")
    assert "LOCATION:Room 1\\, LBCC\\, 300 East Ocean" in block


def test_events_to_ics_folds_long_lines() -> None:
    """RFC 5545 mandates 75-octet line folding with a space continuation."""
    long_title = "A" * 200
    block = events_to_ics([_event("1", long_title)])
    for line in block.split("\r\n"):
        # Continuation lines start with " "; primary lines must stay under 76 chars.
        assert len(line) <= 75 or line.startswith(" ")
