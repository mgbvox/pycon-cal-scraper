"""Diff a saved-list against a Google Calendar and apply the result.

The diff is a pure function: it takes a list of :class:`Event` and the
current ``events.list()`` response items (filtered by
``privateExtendedProperty``), and returns a :class:`SyncPlan`. The execute
step calls the calendar API to fulfil the plan; it is kept thin so the pure
diff stays trivially testable.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from pycon_cal_scraper.models import Event

PYCON_ID_KEY = "pycon_id"
DEFAULT_CALENDAR = "primary"

#: Google Calendar caps a single batch at 50 sub-requests.
#: See https://developers.google.com/calendar/api/guides/batch.
BATCH_LIMIT = 50

#: Default number of retry passes for sub-requests that fail inside a batch.
#: After this many additional rounds we surface an ExceptionGroup.
DEFAULT_RETRIES = 3

#: Callback for progress reporting from :func:`apply`. Fired as
#: ``(action, completed, total)`` where ``action`` is ``"insert"``,
#: ``"patch"``, or ``"delete"`` and ``total`` is the total number of API
#: actions in the plan. An initial ``completed=0`` event is emitted before
#: any action runs.
ApplyProgressCallback = Callable[[str, int, int], None]

#: Callback fired by :func:`list_managed_events` after each page of the
#: paginated ``events.list`` call. Arguments are
#: ``(page_number, items_scanned_so_far, managed_found_so_far)``. Useful
#: for streaming progress on large calendars.
ListProgressCallback = Callable[[int, int, int], None]

#: Callback fired by :func:`apply` immediately before each batch HTTP
#: request is dispatched. Arguments are
#: ``(batch_index, batches_in_round, retry_pass)`` where ``batch_index``
#: is 0-based within the current round, ``batches_in_round`` is the total
#: number of batches still pending in this round, and ``retry_pass`` is
#: 0 on the initial attempt and increments for each retry pass. Use this
#: to surface batch/retry context (e.g. in a progress bar label); the
#: per-sub-request :data:`ApplyProgressCallback` continues to fire for
#: bar fill granularity.
BatchProgressCallback = Callable[[int, int, int], None]


def extract_pycon_id(gcal_item: dict[str, Any]) -> str | None:
    """Return the ``pycon_id`` private extended-property from a GCal item.

    Args:
        gcal_item: An ``events.list`` item.

    Returns:
        The id we tagged when inserting, or ``None`` if the item doesn't
        carry one.
    """
    return (
        gcal_item.get("extendedProperties", {}).get("private", {}).get(PYCON_ID_KEY)  # type: ignore[no-any-return]
    )


def build_location(room: str | None, venue_address: str | None) -> str | None:
    """Compose the Google Calendar ``location`` field from room and venue.

    Args:
        room: The room within the venue (e.g. ``"Room 103ABC"``). May be
            ``None`` when the scraper couldn't determine a room.
        venue_address: The full postal address of the conference venue.
            May be ``None`` / empty to skip the venue suffix.

    Returns:
        A single string Google Maps can parse. Examples:

        * ``"Room 103ABC, Long Beach Convention Center, 300 East Ocean
          Boulevard, Long Beach, CA 90802"`` (both present)
        * ``"Long Beach Convention Center, 300 East Ocean Boulevard, Long
          Beach, CA 90802"`` (no room — still opens the right building)
        * ``"Room 103ABC"`` (no venue configured)
        * ``None`` (neither available)
    """
    room_clean = (room or "").strip() or None
    venue_clean = (venue_address or "").strip() or None
    if room_clean and venue_clean:
        return f"{room_clean}, {venue_clean}"
    return room_clean or venue_clean


def build_event_payload(event: Event, *, venue_address: str | None = None) -> dict[str, Any]:
    """Build the request body for ``events.insert`` / ``events.update``.

    Args:
        event: The PyCon event to translate.
        venue_address: Optional full postal address of the conference
            venue. When provided, it's appended to ``event.room`` to form
            the calendar ``location`` field, so Google Maps resolves to
            the building. When ``None`` (the default), only the room is
            used — useful for tests that don't care about the venue.

    Returns:
        A dict suitable as the ``body`` argument to the Calendar API. It
        embeds the PyCon id and URL as private extended properties so we
        can find these events later.
    """
    tz = event.start.tzinfo
    tz_name = getattr(tz, "key", "America/Los_Angeles")
    summary = event.title
    description_lines: list[str] = []
    if event.speakers:
        description_lines.append("Speakers: " + ", ".join(event.speakers))
    if event.audience_level:
        description_lines.append(f"Audience: {event.audience_level}")
    if event.track:
        description_lines.append(f"Track: {event.track}")
    if event.description or event.abstract:
        description_lines.append("")
        description_lines.append(event.description or event.abstract or "")
    description_lines.append("")
    description_lines.append(str(event.url))
    return {
        "summary": summary,
        "location": build_location(event.room, venue_address),
        "description": "\n".join(description_lines),
        "start": {
            "dateTime": event.start.isoformat(),
            "timeZone": tz_name,
        },
        "end": {
            "dateTime": event.end.isoformat(),
            "timeZone": tz_name,
        },
        "extendedProperties": {
            "private": {
                PYCON_ID_KEY: event.id,
                "pycon_url": str(event.url),
            }
        },
    }


def _has_drifted(
    event: Event, gcal_item: dict[str, Any], *, venue_address: str | None = None
) -> bool:
    """Return ``True`` iff the GCal item differs from ``event`` in a meaningful way."""
    desired = build_event_payload(event, venue_address=venue_address)
    for key in ("summary", "location", "description"):
        if (gcal_item.get(key) or None) != (desired[key] or None):
            return True
    for key in ("start", "end"):
        if gcal_item.get(key, {}).get("dateTime") != desired[key]["dateTime"]:
            return True
    return False


@dataclass
class SyncPlan:
    """The set of mutations that bring a Google Calendar into sync.

    Attributes:
        to_insert: Saved events that don't yet exist on the calendar.
        to_patch: ``(event, gcal_event_id)`` pairs that exist but have drifted.
        to_delete: Google Calendar event ids to remove (only populated when
            ``prune=True``).
    """

    to_insert: list[Event] = field(default_factory=list)
    to_patch: list[tuple[Event, str]] = field(default_factory=list)
    to_delete: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Return ``True`` if the plan has no actions."""
        return not (self.to_insert or self.to_patch or self.to_delete)

    def total_actions(self) -> int:
        """Return the total number of API calls this plan represents."""
        return len(self.to_insert) + len(self.to_patch) + len(self.to_delete)

    def summary(self) -> str:
        """Return a short one-line description of the plan."""
        return (
            f"{len(self.to_insert)} insert, "
            f"{len(self.to_patch)} patch, "
            f"{len(self.to_delete)} delete"
        )


def diff(
    saved: Sequence[Event],
    existing_gcal_items: Sequence[dict[str, Any]],
    *,
    prune: bool = False,
    venue_address: str | None = None,
) -> SyncPlan:
    """Compute the actions needed to make the calendar match ``saved``.

    Args:
        saved: The saved-list events that should be on the calendar.
        existing_gcal_items: The current calendar items (filtered to those
            we previously created).
        prune: When ``True``, also delete GCal items whose ``pycon_id`` is
            no longer in ``saved``. Defaults to ``False`` so accidental
            unsaves don't silently destroy calendar entries.
        venue_address: Optional venue postal address; threaded through to
            :func:`build_event_payload` so the drift check compares the
            *full* location string (room + venue) rather than the room
            alone.

    Returns:
        A :class:`SyncPlan` describing the inserts, patches, and deletes
        required.
    """
    plan = SyncPlan()
    existing_by_pid: dict[str, dict[str, Any]] = {}
    for item in existing_gcal_items:
        pid = extract_pycon_id(item)
        if pid is not None:
            existing_by_pid[pid] = item

    saved_ids: set[str] = set()
    for event in saved:
        saved_ids.add(event.id)
        existing = existing_by_pid.get(event.id)
        if existing is None:
            plan.to_insert.append(event)
        elif _has_drifted(event, existing, venue_address=venue_address):
            plan.to_patch.append((event, str(existing["id"])))

    if prune:
        for pid, item in existing_by_pid.items():
            if pid not in saved_ids:
                plan.to_delete.append(str(item["id"]))

    return plan


@dataclass(frozen=True)
class _SubRequest:
    """A single sub-request inside a calendar batch.

    Carries everything needed to *rebuild* the underlying
    ``googleapiclient`` request on each retry, since ``BatchHttpRequest``
    consumes its inputs.
    """

    rid: str
    action: str  # "insert" | "patch" | "delete"
    event: Event | None  # populated for insert/patch, None for delete
    gcal_id: str | None  # populated for patch/delete, None for insert

    def build_request(
        self, events_resource: Any, calendar_id: str, venue_address: str | None
    ) -> Any:
        """Construct a fresh googleapiclient request for this sub-request."""
        if self.action == "insert":
            assert self.event is not None
            return events_resource.insert(
                calendarId=calendar_id,
                body=build_event_payload(self.event, venue_address=venue_address),
            )
        if self.action == "patch":
            assert self.event is not None and self.gcal_id is not None
            return events_resource.update(
                calendarId=calendar_id,
                eventId=self.gcal_id,
                body=build_event_payload(self.event, venue_address=venue_address),
            )
        assert self.action == "delete" and self.gcal_id is not None
        return events_resource.delete(calendarId=calendar_id, eventId=self.gcal_id)


def _plan_to_subrequests(plan: SyncPlan) -> list[_SubRequest]:
    """Flatten a :class:`SyncPlan` into a list of sub-requests with stable rids.

    Rids are stable so retries can be matched back to the same sub-request
    across batches.
    """
    out: list[_SubRequest] = []
    for event in plan.to_insert:
        out.append(
            _SubRequest(rid=f"insert:{event.id}", action="insert", event=event, gcal_id=None)
        )
    for event, gcal_id in plan.to_patch:
        out.append(
            _SubRequest(rid=f"patch:{event.id}", action="patch", event=event, gcal_id=gcal_id)
        )
    for gcal_id in plan.to_delete:
        out.append(
            _SubRequest(rid=f"delete:{gcal_id}", action="delete", event=None, gcal_id=gcal_id)
        )
    return out


def _run_single_batch(
    service: Any,
    events_resource: Any,
    calendar_id: str,
    venue_address: str | None,
    chunk: Sequence[_SubRequest],
) -> tuple[set[str], dict[str, Exception]]:
    """Send ``chunk`` (≤ 50 requests) as one Google Calendar batch.

    Returns ``(successful_rids, {failed_rid: exception})``.
    """
    successes: set[str] = set()
    failures: dict[str, Exception] = {}

    def _callback(request_id: str, response: Any, exception: Exception | None) -> None:
        if exception is not None:
            failures[request_id] = exception
        else:
            successes.add(request_id)

    batch = service.new_batch_http_request(callback=_callback)
    for sub in chunk:
        batch.add(
            sub.build_request(events_resource, calendar_id, venue_address), request_id=sub.rid
        )
    batch.execute()
    return successes, failures


def _apply_sync(
    service: Any,
    plan: SyncPlan,
    *,
    calendar_id: str,
    on_progress: ApplyProgressCallback | None,
    venue_address: str | None,
    retries: int,
    batch_size: int,
    on_batch: BatchProgressCallback | None = None,
) -> None:
    """Synchronous worker that drives the calendar API via batched calls.

    Splits the plan into chunks of at most ``batch_size`` sub-requests and
    sends each chunk as one batch HTTP request. Sub-requests that fail are
    retried up to ``retries`` additional times across the whole plan; any
    that still fail after the final pass are aggregated into an
    :class:`ExceptionGroup` raised at the end of the call.

    Args:
        service: The discovery-built ``calendar`` v3 service.
        plan: The plan to apply.
        calendar_id: The calendar to mutate.
        on_progress: Optional progress callback fired before/after each action.
        on_batch: Optional callback fired before each batch is sent — useful
            for surfacing batch/retry context in a progress label. See
            :data:`BatchProgressCallback`.
        venue_address: Forwarded to :func:`build_event_payload`.
        retries: Number of retry passes after the initial attempt.
        batch_size: Maximum sub-requests per batch (Google's hard cap is 50).
    """
    sub_requests = _plan_to_subrequests(plan)
    total = len(sub_requests)
    if on_progress is not None:
        on_progress("start", 0, total)
    if not sub_requests:
        return

    events_resource = service.events()
    pending: list[_SubRequest] = list(sub_requests)
    final_errors: dict[str, Exception] = {}
    done = 0

    for retry_pass in range(retries + 1):
        if not pending:
            break
        next_pending: list[_SubRequest] = []
        round_errors: dict[str, Exception] = {}
        batches_in_round = (len(pending) + batch_size - 1) // batch_size
        for batch_idx, start in enumerate(range(0, len(pending), batch_size)):
            chunk = pending[start : start + batch_size]
            if on_batch is not None:
                on_batch(batch_idx, batches_in_round, retry_pass)
            successes, failures = _run_single_batch(
                service, events_resource, calendar_id, venue_address, chunk
            )
            for sub in chunk:
                if sub.rid in successes:
                    done += 1
                    if on_progress is not None:
                        on_progress(sub.action, done, total)
                else:
                    next_pending.append(sub)
                    round_errors[sub.rid] = failures.get(
                        sub.rid, RuntimeError(f"no response for {sub.rid}")
                    )
        pending = next_pending
        final_errors = round_errors

    if pending:
        raise ExceptionGroup(
            f"{len(pending)} calendar sync operation(s) failed after {retries} retries",
            [final_errors[sub.rid] for sub in pending],
        )


async def apply(
    service: Any,
    plan: SyncPlan,
    *,
    calendar_id: str = DEFAULT_CALENDAR,
    on_progress: ApplyProgressCallback | None = None,
    on_batch: BatchProgressCallback | None = None,
    venue_address: str | None = None,
    retries: int = DEFAULT_RETRIES,
    batch_size: int = BATCH_LIMIT,
) -> None:
    """Apply a :class:`SyncPlan` to a Google Calendar.

    Sub-requests are bundled into Google Calendar batch HTTP requests
    (max ``batch_size`` ≤ 50 per Google's limit), so each batch is a single
    round-trip regardless of how many inserts/patches/deletes it contains.
    Failed sub-requests are retried up to ``retries`` more times; any that
    still fail are raised as an :class:`ExceptionGroup` at the end.

    ``googleapiclient`` is synchronous, so the actual HTTP work runs in a
    worker thread to keep this function awaitable alongside the rest of the
    pipeline.

    Args:
        service: The discovery-built ``calendar`` v3 service.
        plan: The plan to apply.
        calendar_id: The calendar to mutate. Defaults to ``"primary"``.
        on_progress: Optional progress callback. See :data:`ApplyProgressCallback`.
        on_batch: Optional batch-level callback (see
            :data:`BatchProgressCallback`) fired before each batch HTTP
            request is dispatched. Use it to annotate a progress bar with
            batch/retry context.
        venue_address: Forwarded to :func:`build_event_payload`.
        retries: Extra attempts (default 3) for sub-requests that fail inside
            a batch. Set to 0 to disable retries.
        batch_size: Maximum sub-requests per batch (Google's cap is 50).

    Raises:
        ExceptionGroup: When one or more sub-requests still fail after all
            retries; the group's ``exceptions`` carry every individual
            Google API error in the order the sub-requests were attempted.
    """
    await asyncio.to_thread(
        _apply_sync,
        service,
        plan,
        calendar_id=calendar_id,
        on_progress=on_progress,
        on_batch=on_batch,
        venue_address=venue_address,
        retries=retries,
        batch_size=batch_size,
    )


def _list_managed_sync(
    service: Any,
    *,
    calendar_id: str,
    time_min: datetime | None,
    time_max: datetime | None,
    on_page: ListProgressCallback | None,
) -> list[dict[str, Any]]:
    """Sync helper for :func:`list_managed_events`; paginates the API."""
    events_resource = service.events()
    items: list[dict[str, Any]] = []
    page_token: str | None = None
    page_number = 0
    items_scanned = 0
    kwargs: dict[str, Any] = {
        "calendarId": calendar_id,
        "singleEvents": True,
        "maxResults": 2500,
        "showDeleted": False,
    }
    if time_min is not None:
        kwargs["timeMin"] = time_min.isoformat()
    if time_max is not None:
        kwargs["timeMax"] = time_max.isoformat()
    while True:
        response = events_resource.list(pageToken=page_token, **kwargs).execute()
        page_number += 1
        for item in response.get("items", []):
            items_scanned += 1
            # Google's API has no "key exists" filter for privateExtendedProperty —
            # `pycon_id=*` matches the literal value '*', not "any value". So we
            # query an unfiltered window and keep only the events we own.
            if extract_pycon_id(item) is not None:
                items.append(item)
        if on_page is not None:
            on_page(page_number, items_scanned, len(items))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return items


async def list_managed_events(
    service: Any,
    *,
    calendar_id: str = DEFAULT_CALENDAR,
    time_min: datetime | None = None,
    time_max: datetime | None = None,
    on_page: ListProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """List every event on ``calendar_id`` that pycon-cal-scraper previously created.

    Args:
        service: The discovery-built ``calendar`` v3 service.
        calendar_id: The calendar to read. Defaults to ``"primary"``.
        time_min: Optional lower bound — exclude events ending before this
            datetime. Callers should typically pass a small window around
            the saved events so we don't scan the user's entire calendar.
        time_max: Optional upper bound — exclude events starting after this
            datetime.
        on_page: Optional callback fired after each paginated response.
            See :data:`ListProgressCallback`.

    Returns:
        Every event in the window whose ``extendedProperties.private``
        contains ``pycon_id``. The blocking ``googleapiclient`` paging
        runs in a worker thread.
    """
    return await asyncio.to_thread(
        _list_managed_sync,
        service,
        calendar_id=calendar_id,
        time_min=time_min,
        time_max=time_max,
        on_page=on_page,
    )


def conference_window(
    events: Iterable[Event], *, padding_days: int = 1
) -> tuple[datetime, datetime] | None:
    """Return a ``(time_min, time_max)`` bracket around ``events`` for the API.

    Args:
        events: The events whose times define the window.
        padding_days: Pad each side by this many days so events on the edges
            aren't accidentally clipped by timezone offsets.

    Returns:
        A ``(time_min, time_max)`` tuple, or ``None`` if ``events`` is empty.
    """
    starts: list[datetime] = []
    ends: list[datetime] = []
    for e in events:
        starts.append(e.start)
        ends.append(e.end)
    if not starts:
        return None
    pad = timedelta(days=padding_days)
    return min(starts) - pad, max(ends) + pad
