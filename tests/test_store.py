"""Tests for on-disk persistence of events and the saved-list."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from pycon_cal_scraper.models import Event, EventType, SavedEvent
from pycon_cal_scraper.store import EventsStore, SavedStore

PACIFIC = ZoneInfo("America/Los_Angeles")


def _event(eid: str, title: str = "Demo") -> Event:
    return Event.model_validate(
        {
            "id": eid,
            "url": f"https://us.pycon.org/2026/schedule/presentation/{eid}/",
            "title": title,
            "type": EventType.talk,
            "speakers": ["Someone"],
            "start": datetime(2026, 5, 15, 9, 0, tzinfo=PACIFIC),
            "end": datetime(2026, 5, 15, 9, 30, tzinfo=PACIFIC),
            "room": "Room 1",
        }
    )


def test_events_store_save_and_load(tmp_path: Path) -> None:
    store = EventsStore(tmp_path / "events.json")
    assert store.load() == []
    events = [_event("1"), _event("2", title="Other")]
    store.save(events)
    loaded = store.load()
    assert loaded == events


def test_events_store_missing_file_returns_empty(tmp_path: Path) -> None:
    store = EventsStore(tmp_path / "absent.json")
    assert store.load() == []


def test_saved_store_add_is_idempotent(tmp_path: Path) -> None:
    store = SavedStore(tmp_path / "saved.json")
    store.add("1")
    store.add("1")
    store.add("2")
    ids = store.ids()
    assert ids == {"1", "2"}


def test_saved_store_remove_missing_is_noop(tmp_path: Path) -> None:
    store = SavedStore(tmp_path / "saved.json")
    store.add("1")
    store.remove("does-not-exist")
    assert store.ids() == {"1"}


def test_saved_store_iteration_yields_savedevent(tmp_path: Path) -> None:
    store = SavedStore(tmp_path / "saved.json")
    store.add("1")
    items = list(store)
    assert len(items) == 1
    assert isinstance(items[0], SavedEvent)
    assert items[0].id == "1"


def test_saved_store_resolves_against_events(tmp_path: Path) -> None:
    events_store = EventsStore(tmp_path / "events.json")
    saved_store = SavedStore(tmp_path / "saved.json")
    events_store.save([_event("1"), _event("2"), _event("3")])
    saved_store.add("2")
    saved_store.add("1")
    resolved = saved_store.resolve(events_store.load())
    # Resolution preserves insertion order from the saved file.
    assert [e.id for e in resolved] == ["2", "1"]


def test_saved_store_resolve_silently_skips_unknown(tmp_path: Path) -> None:
    events_store = EventsStore(tmp_path / "events.json")
    saved_store = SavedStore(tmp_path / "saved.json")
    events_store.save([_event("1")])
    saved_store.add("1")
    saved_store.add("999")  # not in the events store; e.g. stale schedule
    resolved = saved_store.resolve(events_store.load())
    assert [e.id for e in resolved] == ["1"]


def test_saved_store_clear(tmp_path: Path) -> None:
    store = SavedStore(tmp_path / "saved.json")
    store.add("1")
    store.add("2")
    store.clear()
    assert store.ids() == set()
