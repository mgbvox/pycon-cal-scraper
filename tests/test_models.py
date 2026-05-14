"""Tests for the Event / EventType / SavedEvent data model."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from pycon_cal_scraper.models import Event, EventType, SavedEvent

PACIFIC = ZoneInfo("America/Los_Angeles")


def _make_event(**overrides: object) -> Event:
    defaults: dict[str, object] = dict(
        id="2",
        url="https://us.pycon.org/2026/schedule/presentation/2/",
        title="Build your first MCP server in Python",
        type=EventType.tutorial,
        speakers=["Pamela Fox"],
        start=datetime(2026, 5, 13, 9, 0, tzinfo=PACIFIC),
        end=datetime(2026, 5, 13, 12, 30, tzinfo=PACIFIC),
        room="Room 101A",
    )
    defaults.update(overrides)
    return Event.model_validate(defaults)


def test_event_minimum_valid() -> None:
    e = _make_event()
    assert e.id == "2"
    assert e.type is EventType.tutorial
    assert e.start.tzinfo is not None
    assert e.duration_minutes == 210


def test_event_rejects_naive_datetimes() -> None:
    with pytest.raises(ValidationError):
        _make_event(start=datetime(2026, 5, 13, 9, 0))  # type: ignore[arg-type]


def test_event_rejects_end_before_start() -> None:
    with pytest.raises(ValidationError):
        _make_event(end=datetime(2026, 5, 13, 8, 0, tzinfo=PACIFIC))


def test_event_serialization_roundtrip() -> None:
    e = _make_event(abstract="MCP is great.", description="Long description.")
    payload = e.model_dump(mode="json")
    restored = Event.model_validate(payload)
    assert restored == e


def test_event_normalizes_speakers_string() -> None:
    e = _make_event(speakers="Jia Chen, Steven Troxler")
    assert e.speakers == ["Jia Chen", "Steven Troxler"]


def test_event_type_from_slot_class() -> None:
    assert EventType.from_slot_class("slot-talk") is EventType.talk
    assert EventType.from_slot_class("slot-tutorial") is EventType.tutorial
    assert EventType.from_slot_class("slot-sponsor") is EventType.sponsor
    assert EventType.from_slot_class("slot-keynote") is EventType.keynote
    assert EventType.from_slot_class("slot-plenary") is EventType.plenary
    assert EventType.from_slot_class("slot-break") is EventType.break_
    assert EventType.from_slot_class("slot-unknown") is None


def test_saved_event_roundtrip() -> None:
    saved = SavedEvent(id="2", saved_at=datetime(2026, 5, 10, 0, 0, tzinfo=UTC))
    payload = saved.model_dump(mode="json")
    assert SavedEvent.model_validate(payload) == saved
