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


def build_event_payload(event: Event) -> dict[str, Any]:
    """Build the request body for ``events.insert`` / ``events.update``.

    Args:
        event: The PyCon event to translate.

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
        "location": event.room,
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


def _has_drifted(event: Event, gcal_item: dict[str, Any]) -> bool:
    """Return ``True`` iff the GCal item differs from ``event`` in a meaningful way."""
    desired = build_event_payload(event)
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
) -> SyncPlan:
    """Compute the actions needed to make the calendar match ``saved``.

    Args:
        saved: The saved-list events that should be on the calendar.
        existing_gcal_items: The current calendar items (filtered to those
            we previously created).
        prune: When ``True``, also delete GCal items whose ``pycon_id`` is
            no longer in ``saved``. Defaults to ``False`` so accidental
            unsaves don't silently destroy calendar entries.

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
        elif _has_drifted(event, existing):
            plan.to_patch.append((event, str(existing["id"])))

    if prune:
        for pid, item in existing_by_pid.items():
            if pid not in saved_ids:
                plan.to_delete.append(str(item["id"]))

    return plan


def _apply_sync(
    service: Any,
    plan: SyncPlan,
    *,
    calendar_id: str,
    on_progress: ApplyProgressCallback | None,
) -> None:
    """Synchronous worker that actually drives the googleapiclient calls.

    Args:
        service: The discovery-built ``calendar`` v3 service.
        plan: The plan to apply.
        calendar_id: The calendar to mutate.
        on_progress: Optional progress callback fired before/after each action.
    """
    events_resource = service.events()
    total = plan.total_actions()
    done = 0
    if on_progress is not None:
        on_progress("start", 0, total)
    for event in plan.to_insert:
        events_resource.insert(calendarId=calendar_id, body=build_event_payload(event)).execute()
        done += 1
        if on_progress is not None:
            on_progress("insert", done, total)
    for event, gcal_id in plan.to_patch:
        events_resource.update(
            calendarId=calendar_id, eventId=gcal_id, body=build_event_payload(event)
        ).execute()
        done += 1
        if on_progress is not None:
            on_progress("patch", done, total)
    for gcal_id in plan.to_delete:
        events_resource.delete(calendarId=calendar_id, eventId=gcal_id).execute()
        done += 1
        if on_progress is not None:
            on_progress("delete", done, total)


async def apply(
    service: Any,
    plan: SyncPlan,
    *,
    calendar_id: str = DEFAULT_CALENDAR,
    on_progress: ApplyProgressCallback | None = None,
) -> None:
    """Apply a :class:`SyncPlan` to a Google Calendar.

    ``googleapiclient`` is synchronous, so the actual HTTP calls run in a
    worker thread to keep this function awaitable alongside the rest of the
    pipeline.

    Args:
        service: The discovery-built ``calendar`` v3 service.
        plan: The plan to apply.
        calendar_id: The calendar to mutate. Defaults to ``"primary"``.
        on_progress: Optional progress callback. See :data:`ApplyProgressCallback`.
    """
    await asyncio.to_thread(
        _apply_sync, service, plan, calendar_id=calendar_id, on_progress=on_progress
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
