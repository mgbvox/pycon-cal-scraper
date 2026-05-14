"""Tests for the Google Calendar sync diff and event-payload builders."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from pycon_cal_scraper.gcal.sync import (
    SyncPlan,
    build_event_payload,
    build_location,
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


def test_build_location_combines_room_and_venue() -> None:
    """Both room and venue → comma-separated single string."""
    assert (
        build_location("Room 103ABC", "Long Beach Convention Center, 300 East Ocean")
        == "Room 103ABC, Long Beach Convention Center, 300 East Ocean"
    )


def test_build_location_falls_back_to_venue_when_room_missing() -> None:
    """No room → just the venue so Google Maps still resolves the building."""
    assert build_location(None, "300 East Ocean Boulevard") == "300 East Ocean Boulevard"
    assert build_location("", "300 East Ocean Boulevard") == "300 East Ocean Boulevard"


def test_build_location_returns_room_when_no_venue_configured() -> None:
    """No venue → just the room (legacy behavior, useful for tests)."""
    assert build_location("Room 1", None) == "Room 1"
    assert build_location("Room 1", "") == "Room 1"


def test_build_location_returns_none_when_neither_present() -> None:
    assert build_location(None, None) is None
    assert build_location("", "") is None


def test_build_event_payload_includes_venue_in_location() -> None:
    """``build_event_payload`` with a venue should emit the composite location."""
    e = _event("1", title="Hi", room="Room 103ABC")
    payload = build_event_payload(e, venue_address="Long Beach Convention Center")
    assert payload["location"] == "Room 103ABC, Long Beach Convention Center"


def test_sync_plan_summary_counts() -> None:
    plan = SyncPlan(to_insert=[_event("1")], to_patch=[(_event("2"), "g2")], to_delete=["g3"])
    assert plan.summary() == "1 insert, 1 patch, 1 delete"


class _FakeBatch:
    """A drop-in stand-in for ``BatchHttpRequest`` used in apply-path tests.

    Records each ``add()`` call and, on ``execute()``, fires the registered
    callback for every queued sub-request — either with a fake response or
    an exception, driven by ``failure_rounds`` (a per-rid count of
    consecutive failures to inject before letting the request succeed).
    """

    def __init__(self, callback: object, failure_rounds: dict[str, int]) -> None:
        self._callback = callback
        self._failure_rounds = failure_rounds
        self._queued: list[tuple[object, str]] = []

    def add(self, request: object, request_id: str) -> None:
        self._queued.append((request, request_id))

    def execute(self) -> None:
        for _request, rid in self._queued:
            remaining = self._failure_rounds.get(rid, 0)
            if remaining > 0:
                self._failure_rounds[rid] = remaining - 1
                exc = RuntimeError(f"injected failure: {rid}")
                self._callback(rid, None, exc)  # type: ignore[operator]
            else:
                self._callback(rid, {"id": rid, "status": "ok"}, None)  # type: ignore[operator]


class _FakeEventsResource:
    """Returns sentinel objects representing each kind of write request."""

    def insert(self, *, calendarId: str, body: dict[str, object]) -> dict[str, object]:
        return {"action": "insert", "calendarId": calendarId, "body": body}

    def update(
        self, *, calendarId: str, eventId: str, body: dict[str, object]
    ) -> dict[str, object]:
        return {"action": "update", "calendarId": calendarId, "eventId": eventId, "body": body}

    def delete(self, *, calendarId: str, eventId: str) -> dict[str, object]:
        return {"action": "delete", "calendarId": calendarId, "eventId": eventId}


class _FakeService:
    """A fake calendar v3 service exposing ``events()`` and the batch factory."""

    def __init__(self, failure_rounds: dict[str, int] | None = None) -> None:
        self._events = _FakeEventsResource()
        self._failure_rounds = failure_rounds or {}
        self.batches: list[_FakeBatch] = []

    def events(self) -> _FakeEventsResource:
        return self._events

    def new_batch_http_request(self, callback: object) -> _FakeBatch:
        batch = _FakeBatch(callback, self._failure_rounds)
        self.batches.append(batch)
        return batch


def test_apply_sends_all_subrequests_in_one_batch_when_small() -> None:
    """A plan with ≤ batch_size sub-requests should produce exactly one batch."""
    from pycon_cal_scraper.gcal.sync import _apply_sync

    plan = SyncPlan(
        to_insert=[_event("1"), _event("2")],
        to_patch=[(_event("3"), "gcal-3")],
        to_delete=["gcal-9"],
    )
    service = _FakeService()
    progress: list[tuple[str, int, int]] = []

    _apply_sync(
        service,
        plan,
        calendar_id="primary",
        on_progress=lambda action, done, total: progress.append((action, done, total)),
        venue_address=None,
        retries=3,
        batch_size=50,
    )

    assert len(service.batches) == 1
    assert len(service.batches[0]._queued) == 4
    # Progress fires once at start (done=0) plus once per sub-request.
    assert progress[0] == ("start", 0, 4)
    assert progress[-1][1] == 4 and progress[-1][2] == 4


def test_apply_splits_into_multiple_batches_when_over_limit() -> None:
    """With batch_size=2 and 5 inserts the plan should split into 3 batches."""
    from pycon_cal_scraper.gcal.sync import _apply_sync

    plan = SyncPlan(to_insert=[_event(str(i)) for i in range(5)])
    service = _FakeService()

    _apply_sync(
        service,
        plan,
        calendar_id="primary",
        on_progress=None,
        venue_address=None,
        retries=0,
        batch_size=2,
    )

    assert [len(b._queued) for b in service.batches] == [2, 2, 1]


def test_apply_retries_failed_subrequests_and_recovers() -> None:
    """Sub-requests that fail twice should be retried and ultimately succeed."""
    from pycon_cal_scraper.gcal.sync import _apply_sync

    plan = SyncPlan(to_insert=[_event("1"), _event("2")])
    service = _FakeService(failure_rounds={"insert:1": 2})  # rid fails round 1 + 2
    progress: list[tuple[str, int, int]] = []

    _apply_sync(
        service,
        plan,
        calendar_id="primary",
        on_progress=lambda action, done, total: progress.append((action, done, total)),
        venue_address=None,
        retries=3,
        batch_size=50,
    )

    # Initial attempt + 2 retry passes = 3 batches sent. (After the 3rd batch
    # everything has succeeded so no further retries are queued.)
    assert len(service.batches) == 3
    # Progress reaches the full plan total.
    assert progress[-1] == ("insert", 2, 2)


def test_apply_raises_exception_group_after_exhausting_retries() -> None:
    """When sub-requests keep failing, ``apply`` should surface an ExceptionGroup."""
    from pycon_cal_scraper.gcal.sync import _apply_sync

    plan = SyncPlan(
        to_insert=[_event("1")],
        to_delete=["gcal-9"],
    )
    # Each rid is rigged to fail more times than the retry budget allows.
    service = _FakeService(failure_rounds={"insert:1": 99, "delete:gcal-9": 99})

    try:
        _apply_sync(
            service,
            plan,
            calendar_id="primary",
            on_progress=None,
            venue_address=None,
            retries=2,
            batch_size=50,
        )
    except ExceptionGroup as eg:
        assert len(eg.exceptions) == 2
        assert all(isinstance(e, RuntimeError) for e in eg.exceptions)
    else:  # pragma: no cover - test should never reach here
        raise AssertionError("expected ExceptionGroup")

    # 1 initial attempt + 2 retries = 3 rounds (each with 1 batch).
    assert len(service.batches) == 3


def test_apply_zero_retries_disables_retry_loop() -> None:
    """``retries=0`` should mean exactly one attempt; failures raise immediately."""
    from pycon_cal_scraper.gcal.sync import _apply_sync

    plan = SyncPlan(to_insert=[_event("1")])
    service = _FakeService(failure_rounds={"insert:1": 1})

    try:
        _apply_sync(
            service,
            plan,
            calendar_id="primary",
            on_progress=None,
            venue_address=None,
            retries=0,
            batch_size=50,
        )
    except ExceptionGroup as eg:
        assert len(eg.exceptions) == 1
    else:  # pragma: no cover
        raise AssertionError("expected ExceptionGroup")

    assert len(service.batches) == 1


def test_apply_fires_on_batch_with_retry_pass_progression() -> None:
    """``on_batch`` should fire once per batch with monotonically rising retry_pass."""
    from pycon_cal_scraper.gcal.sync import _apply_sync

    plan = SyncPlan(to_insert=[_event("1"), _event("2"), _event("3")])
    # Force rid "insert:1" to fail twice so we observe retry passes 1 and 2.
    service = _FakeService(failure_rounds={"insert:1": 2})
    batch_events: list[tuple[int, int, int]] = []

    _apply_sync(
        service,
        plan,
        calendar_id="primary",
        on_progress=None,
        on_batch=lambda idx, total, retry: batch_events.append((idx, total, retry)),
        venue_address=None,
        retries=3,
        batch_size=2,
    )

    # Round 0: 3 items / batch_size 2 = 2 batches. Round 1: 1 failing item = 1 batch.
    # Round 2: that item finally succeeds (1 batch). Round 3 not entered.
    assert batch_events == [
        (0, 2, 0),
        (1, 2, 0),
        (0, 1, 1),
        (0, 1, 2),
    ]


def test_apply_empty_plan_does_no_batches() -> None:
    """An empty plan shouldn't call ``new_batch_http_request`` at all."""
    from pycon_cal_scraper.gcal.sync import _apply_sync

    service = _FakeService()
    _apply_sync(
        service,
        SyncPlan(),
        calendar_id="primary",
        on_progress=None,
        venue_address=None,
        retries=3,
        batch_size=50,
    )
    assert service.batches == []


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
