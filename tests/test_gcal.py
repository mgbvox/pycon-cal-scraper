"""Tests for the Google Calendar sync diff and event-payload builders."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from pycon_cal_scraper.gcal.sync import (
    SyncPlan,
    build_event_payload,
    diff,
    extract_pycon_id,
)
from pycon_cal_scraper.models import Event, EventType

PACIFIC = ZoneInfo("America/Los_Angeles")


def _event(eid: str, title: str = "T", room: str = "Room 1") -> Event:
    return Event.model_validate(
        {
            "id": eid,
            "url": f"https://us.pycon.org/2026/schedule/presentation/{eid}/",
            "title": title,
            "type": EventType.talk,
            "speakers": ["Sp"],
            "start": datetime(2026, 5, 15, 9, 0, tzinfo=PACIFIC),
            "end": datetime(2026, 5, 15, 9, 30, tzinfo=PACIFIC),
            "room": room,
            "description": "Body",
        }
    )


def _gcal_item(
    pycon_id: str, *, summary: str | None = None, location: str = "Room 1"
) -> dict[str, object]:
    """Build a synthetic GCal item that round-trips with build_event_payload."""
    base = build_event_payload(_event(pycon_id, title=summary or "T", room=location))
    return {"id": f"gcal-{pycon_id}", **base}


def test_extract_pycon_id_handles_missing_extended_props() -> None:
    assert extract_pycon_id({"summary": "Hi"}) is None
    assert extract_pycon_id({"extendedProperties": {"private": {"pycon_id": "42"}}}) == "42"


def test_diff_inserts_new_event_when_absent() -> None:
    saved = [_event("1")]
    plan = diff(saved, existing_gcal_items=[])
    assert [e.id for e in plan.to_insert] == ["1"]
    assert plan.to_patch == []
    assert plan.to_delete == []


def test_diff_no_op_when_event_unchanged() -> None:
    saved = [_event("1")]
    existing = [_gcal_item("1")]
    plan = diff(saved, existing_gcal_items=existing)
    assert plan.to_insert == []
    assert plan.to_patch == []
    assert plan.to_delete == []


def test_diff_patches_when_event_changed() -> None:
    saved = [_event("1", title="New Title")]
    existing = [_gcal_item("1", summary="Old Title")]
    plan = diff(saved, existing_gcal_items=existing)
    assert plan.to_insert == []
    assert [pair[0].id for pair in plan.to_patch] == ["1"]
    assert plan.to_patch[0][1] == "gcal-1"


def test_diff_deletes_only_when_prune() -> None:
    saved: list[Event] = []
    existing = [_gcal_item("1")]
    no_prune = diff(saved, existing_gcal_items=existing, prune=False)
    assert no_prune.to_delete == []

    prune = diff(saved, existing_gcal_items=existing, prune=True)
    assert prune.to_delete == ["gcal-1"]


def test_build_event_payload_round_trip_idempotent() -> None:
    e = _event("1", title="Hi")
    payload = build_event_payload(e)
    assert payload["summary"] == "Hi"
    assert payload["location"] == "Room 1"
    assert payload["start"]["timeZone"] == "America/Los_Angeles"
    assert payload["extendedProperties"]["private"]["pycon_id"] == "1"
    assert "presentation/1/" in payload["description"]


def test_sync_plan_summary_counts() -> None:
    plan = SyncPlan(to_insert=[_event("1")], to_patch=[(_event("2"), "g2")], to_delete=["g3"])
    assert plan.summary() == "1 insert, 1 patch, 1 delete"


def test_list_managed_sync_fires_per_page_callback() -> None:
    """:func:`_list_managed_sync` should call ``on_page`` after each API page."""
    from pycon_cal_scraper.gcal.sync import _list_managed_sync

    pages = [
        {
            "items": [
                _gcal_item("1"),
                {"id": "noise-1", "summary": "Unrelated"},
            ],
            "nextPageToken": "tok2",
        },
        {
            "items": [_gcal_item("2"), _gcal_item("3")],
        },
    ]

    class _FakeRequest:
        def __init__(self, page: dict[str, object]) -> None:
            self._page = page

        def execute(self) -> dict[str, object]:
            return self._page

    class _FakeEvents:
        def __init__(self) -> None:
            self._page_iter = iter(pages)

        def list(self, **kwargs: object) -> _FakeRequest:
            return _FakeRequest(next(self._page_iter))

    class _FakeService:
        def events(self) -> _FakeEvents:
            return self._events

        _events = _FakeEvents()

    seen: list[tuple[int, int, int]] = []

    result = _list_managed_sync(
        _FakeService(),
        calendar_id="primary",
        time_min=None,
        time_max=None,
        on_page=lambda page, scanned, found: seen.append((page, scanned, found)),
    )
    # 3 managed events filtered out of 4 scanned.
    assert [extract_pycon_id(item) for item in result] == ["1", "2", "3"]
    assert seen == [(1, 2, 1), (2, 4, 3)]
