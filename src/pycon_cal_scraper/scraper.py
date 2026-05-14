"""Parse the PyCon US schedule pages into Event objects.

Scraping strategy:
- Each schedule "list page" (talks/tutorials/sponsor-presentations) contains one
  ``<div class="calendar">`` per day, parented by ``<div id="panel-YYYY-MM-DD">``.
- Each calendar is a CSS grid: column 1 holds ``<time class="calendar-time">`` rows in
  DOM order; columns 2..N hold ``<div class="calendar-room">`` header cells. Slots
  ``<section class="slot slot-*">`` carry ``style="grid-row-start: N; grid-row-end: M;"``,
  which we resolve back to start/end ``datetime`` via the DOM order of the time elements.
- For events that link to ``/<year>/schedule/presentation/<id>/`` we follow up with a
  detail fetch to populate speakers, audience level, and description.
"""

from __future__ import annotations

import asyncio
import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from datetime import date, datetime, time
from typing import Protocol
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

from pycon_cal_scraper.conference import (
    CONFERENCE_BASE_URL,
    CONFERENCE_TZ,
    PRESENTATION_PATH_TEMPLATE,
    SCHEDULE_PATHS,
)
from pycon_cal_scraper.models import Event, EventType

#: Callback for progress reporting. ``(stage, completed, total)`` is fired
#: at least twice per stage: once with ``completed=0`` when the stage begins
#: (so callers can size their progress bar) and once after each item finishes.
ProgressCallback = Callable[[str, int, int], None]

BASE_URL = CONFERENCE_BASE_URL
PACIFIC = CONFERENCE_TZ

SCHEDULE_PAGES: tuple[str, ...] = tuple(BASE_URL + path for path in SCHEDULE_PATHS)


def schedule_pages_for(base_url: str) -> tuple[str, ...]:
    """Return the schedule list-page URLs rooted at ``base_url``."""
    root = base_url.rstrip("/")
    return tuple(root + path for path in SCHEDULE_PATHS)


_GRID_RE = re.compile(r"grid-row-(start|end)\s*:\s*(\d+)")
_COL_RE = re.compile(r"grid-column-(start|end)\s*:\s*(\d+)")
_PANEL_DATE_RE = re.compile(r"panel-(\d{4}-\d{2}-\d{2})$")
_PRESENTATION_HREF_RE = re.compile(r"/schedule/presentation/(\d+)/")


class _Client(Protocol):
    async def get_text(self, url: str, *, force_refresh: bool = False) -> str: ...


def _slug(text: str) -> str:
    """Slug-ify a string for synthetic event ids (e.g. ``keynote:lin-qiao``)."""
    norm = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    norm = re.sub(r"[^A-Za-z0-9]+", "-", norm).strip("-").lower()
    return norm or "untitled"


def _style_of(tag: Tag) -> str:
    """Return the ``style`` attribute of ``tag`` as a plain string."""
    raw = tag.get("style", "")
    if isinstance(raw, list):
        return " ".join(raw)
    return raw or ""


def _parse_grid_row(style: str) -> tuple[int | None, int | None]:
    """Extract ``(grid-row-start, grid-row-end)`` from a CSS style string."""
    start = end = None
    for kind, value in _GRID_RE.findall(style or ""):
        if kind == "start":
            start = int(value)
        else:
            end = int(value)
    return start, end


def _parse_grid_col(style: str) -> tuple[int | None, int | None]:
    """Extract ``(grid-column-start, grid-column-end)`` from a CSS style string."""
    start = end = None
    for kind, value in _COL_RE.findall(style or ""):
        if kind == "start":
            start = int(value)
        else:
            end = int(value)
    return start, end


def _build_time_table(times: Iterable[str]) -> list[time]:
    """Translate the 12-hour clock-faces in column 1 into ``time`` objects.

    The page starts in the morning, then crosses noon. We bump into PM when
    the hour drops vs. the previous entry; once we're in PM the hour stays
    PM until the day ends.

    Args:
        times: The text content of each ``<time class="calendar-time">`` in
            DOM order (e.g. ``["8:00", "9:00", "12:15", "1:00", ...]``).

    Returns:
        A list of :class:`datetime.time` instances, same length as ``times``.
    """
    result: list[time] = []
    pm = False
    last_hour: int | None = None
    for txt in times:
        m = re.match(r"^\s*(\d{1,2})(?::(\d{2}))?\s*$", txt)
        if not m:
            continue
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        # 12:xx in the morning is genuinely noon; we treat 12 in AM-mode as PM (lunch).
        if (hour == 12 and not pm) or (last_hour is not None and hour < last_hour and not pm):
            pm = True
        last_hour = hour
        h24 = hour if hour == 12 else (hour + 12 if pm else hour)
        result.append(time(hour=h24 % 24, minute=minute))
    return result


def _calendar_day(calendar: Tag) -> date | None:
    """Walk up from a ``<div class="calendar">`` to find the day it belongs to.

    Args:
        calendar: A BeautifulSoup ``<div class="calendar">`` element.

    Returns:
        The date encoded in an ancestor ``id="panel-YYYY-MM-DD"``, or
        ``None`` if no such ancestor exists.
    """
    parent = calendar.parent
    while parent is not None:
        pid_raw = parent.get("id") if isinstance(parent, Tag) else None
        pid = " ".join(pid_raw) if isinstance(pid_raw, list) else pid_raw
        if pid:
            m = _PANEL_DATE_RE.search(pid)
            if m:
                return date.fromisoformat(m.group(1))
        parent = parent.parent if isinstance(parent, Tag) else None
    return None


def _calendar_rooms(calendar: Tag) -> dict[int, str]:
    """Return the calendar's column-index -> room-name map."""
    rooms: dict[int, str] = {}
    for idx, room_div in enumerate(calendar.find_all("div", class_="calendar-room"), start=2):
        rooms[idx] = room_div.get_text(strip=True)
    return rooms


def _slot_room(slot: Tag, rooms: dict[int, str]) -> str | None:
    """Resolve the room for a slot, via ``room-override`` or grid column."""
    override = slot.find("div", class_="room-override")
    if override:
        a = override.find("a") if isinstance(override, Tag) else None
        if a:
            return a.get_text(strip=True)
        if isinstance(override, Tag):
            return override.get_text(strip=True)
    col_start, _ = _parse_grid_col(_style_of(slot))
    if col_start is not None and col_start in rooms:
        return rooms[col_start]
    return None


def _slot_kind(slot: Tag) -> tuple[EventType | None, str | None]:
    """Return the slot's ``(EventType, track)``."""
    classes = slot.get("class") or []
    kind: EventType | None = None
    track: str | None = None
    for c in classes:
        if c == "slot":
            continue
        et = EventType.from_slot_class(c)
        if et is not None and kind is None:
            kind = et
        tr = EventType.track_from_slot_class(c)
        if tr is not None and track is None:
            track = tr
    return kind, track


def _row_to_dt(row: int, day: date, time_table: list[time]) -> datetime | None:
    """Resolve a CSS-grid row index into an absolute :class:`datetime`.

    Args:
        row: 1-based row index from ``grid-row-start`` / ``grid-row-end``.
            Row 1 is the header; first time element sits at row 2.
        day: The calendar day this row belongs to.
        time_table: Time-of-day for each row, in row order.

    Returns:
        A timezone-aware datetime, or ``None`` if ``row`` is out of bounds.
    """
    idx = row - 2
    if idx < 0 or idx >= len(time_table):
        return None
    t = time_table[idx]
    return datetime.combine(day, t, tzinfo=PACIFIC)


def _slot_title(slot: Tag) -> tuple[str, Tag | None]:
    """Return ``(title_text, link_or_None)`` for a slot's title block."""
    title_div = slot.find("div", class_="title")
    if not title_div:
        return "", None
    link = title_div.find("a") if isinstance(title_div, Tag) else None
    text = title_div.get_text(" ", strip=True)
    return text, link if isinstance(link, Tag) else None


def parse_list_page(html: str, *, page_url: str) -> list[dict[str, object]]:
    """Parse a schedule list page.

    Args:
        html: The raw HTML of a talks/tutorials/sponsor-presentations page.
        page_url: Absolute URL of the page (used as a fallback ``url`` for
            events that don't link to a ``/presentation/<id>/`` detail).

    Returns:
        One dict per non-break slot inside the calendar grid, with keys
        ``id``, ``url``, ``title``, ``type``, ``track``, ``speakers``,
        ``start``, ``end``, and ``room``.
    """
    soup = BeautifulSoup(html, "lxml")
    out: list[dict[str, object]] = []
    for calendar in soup.find_all("div", class_="calendar"):
        day = _calendar_day(calendar)
        if day is None:
            continue
        time_texts = [
            t.get_text(strip=True) for t in calendar.find_all("time", class_="calendar-time")
        ]
        time_table = _build_time_table(time_texts)
        rooms = _calendar_rooms(calendar)
        for slot in calendar.find_all("section", class_="slot"):
            kind, track = _slot_kind(slot)
            if kind is None or kind is EventType.break_:
                continue
            row_start, row_end = _parse_grid_row(_style_of(slot))
            if row_start is None or row_end is None:
                continue
            start = _row_to_dt(row_start, day, time_table)
            end = _row_to_dt(row_end, day, time_table)
            if start is None or end is None or end <= start:
                continue
            title, link = _slot_title(slot)
            if not title:
                continue
            speakers_div = slot.find("div", class_="speaker")
            speakers_raw = speakers_div.get_text(" ", strip=True) if speakers_div else ""
            speakers_list = [s.strip() for s in re.split(r",|\band\b", speakers_raw) if s.strip()]
            room = _slot_room(slot, rooms)

            event_url = page_url
            event_id: str
            if link is not None and link.has_attr("href"):
                href = link["href"]
                # bs4 may give a list for multi-valued attributes; href is single-valued.
                if isinstance(href, list):
                    href = href[0]
                m = _PRESENTATION_HREF_RE.search(href)
                if m:
                    event_id = m.group(1)
                    event_url = urljoin(
                        BASE_URL, PRESENTATION_PATH_TEMPLATE.format(event_id=event_id)
                    )
                else:
                    event_id = f"{kind.value}:{_slug(title)}"
                    event_url = urljoin(BASE_URL, href)
            else:
                event_id = f"{kind.value}:{_slug(title)}"
                event_url = page_url

            out.append(
                {
                    "id": event_id,
                    "url": event_url,
                    "title": title,
                    "type": kind,
                    "track": track,
                    "speakers": speakers_list,
                    "start": start,
                    "end": end,
                    "room": room,
                }
            )
    return out


def parse_presentation_detail(html: str) -> dict[str, str | list[str] | None]:
    """Extract speakers, audience level, and description from a detail page.

    Args:
        html: HTML body of a ``/<year>/schedule/presentation/<id>/`` page.

    Returns:
        A dict with keys ``title``, ``speakers`` (list), ``audience_level``,
        and ``description``. Missing fields are ``None`` (or ``[]`` for
        ``speakers``).
    """
    soup = BeautifulSoup(html, "lxml")
    result: dict[str, str | list[str] | None] = {
        "title": None,
        "speakers": [],
        "audience_level": None,
        "description": None,
    }
    h1 = soup.find("h1")
    if h1:
        result["title"] = h1.get_text(" ", strip=True)

    def section_text(label: str) -> str | None:
        for h2 in soup.find_all("h2"):
            if h2.get_text(strip=True).rstrip(":").lower() == label.lower():
                parts: list[str] = []
                for sib in h2.next_siblings:
                    name = getattr(sib, "name", None)
                    if name == "h2":
                        break
                    if hasattr(sib, "get_text"):
                        t = sib.get_text(" ", strip=True)
                        if t:
                            parts.append(t)
                return "\n\n".join(parts) or None
        return None

    presented_by = section_text("Presented by")
    if presented_by:
        result["speakers"] = [s.strip() for s in re.split(r",|\band\b", presented_by) if s.strip()]
    audience = section_text("Experience Level")
    if audience:
        result["audience_level"] = audience
    description = section_text("Description")
    if description:
        result["description"] = description
    return result


class Scraper:
    """Drives an async HTTP client through the configured list/detail pages.

    The list pages are fetched in parallel; for each event with a numeric
    presentation id, the detail page is fetched in parallel as well.
    Concurrency is bounded by the underlying :class:`CachedClient`'s
    semaphore, so this method scales without overwhelming the conference's
    server.

    Attributes:
        client: An async HTTP client implementing ``get_text(url, *,
            force_refresh)``. Typically a :class:`CachedClient`.
        pages: The schedule list pages this scraper will pull.
    """

    def __init__(self, client: _Client, pages: Sequence[str] = SCHEDULE_PAGES) -> None:
        """Build a new scraper.

        Args:
            client: The async HTTP client to use.
            pages: List-page URLs to scrape. Defaults to :data:`SCHEDULE_PAGES`.
        """
        self.client = client
        self.pages = tuple(pages)

    async def fetch_all(
        self,
        *,
        force_refresh: bool = False,
        on_progress: ProgressCallback | None = None,
    ) -> list[Event]:
        """Fetch every event from the configured pages and return them.

        Args:
            force_refresh: When ``True``, ignore the on-disk HTTP cache and
                re-fetch every page.
            on_progress: Optional callback invoked as
                ``on_progress(stage, completed, total)``. Stages are
                ``"list"`` (one item per list page) and ``"detail"`` (one
                item per presentation detail page). Each stage emits an
                initial ``completed=0`` event so callers can size their
                progress bar before work starts.

        Returns:
            A de-duplicated list of :class:`Event` objects, one per slot
            across all configured pages.
        """
        list_htmls = await self._fetch_stage(
            "list",
            self.pages,
            force_refresh=force_refresh,
            on_progress=on_progress,
        )
        rows: list[dict[str, object]] = []
        for page, html in zip(self.pages, list_htmls, strict=True):
            rows.extend(parse_list_page(html, page_url=page))

        # Fetch all detail pages concurrently, deduped by URL.
        detail_urls = sorted({str(r["url"]) for r in rows if str(r["id"]).isdigit()})
        detail_htmls = await self._fetch_stage(
            "detail",
            detail_urls,
            force_refresh=force_refresh,
            on_progress=on_progress,
        )
        details: dict[str, dict[str, str | list[str] | None]] = {
            u: parse_presentation_detail(h) for u, h in zip(detail_urls, detail_htmls, strict=True)
        }

        events: dict[str, Event] = {}
        for row in rows:
            event_id = str(row["id"])
            if event_id.isdigit():
                d = details[str(row["url"])]
                speakers = d["speakers"] or row["speakers"]
                abstract = d["description"]
                description = d["description"]
                audience_level = d["audience_level"]
            else:
                speakers = row["speakers"]
                abstract = None
                description = None
                audience_level = None
            event = Event.model_validate(
                {
                    "id": event_id,
                    "url": row["url"],
                    "title": row["title"],
                    "type": row["type"],
                    "track": row.get("track"),
                    "speakers": speakers,
                    "start": row["start"],
                    "end": row["end"],
                    "room": row["room"],
                    "audience_level": audience_level,
                    "abstract": abstract,
                    "description": description,
                }
            )
            events[event_id] = event
        return list(events.values())

    async def _fetch_stage(
        self,
        stage: str,
        urls: Sequence[str],
        *,
        force_refresh: bool,
        on_progress: ProgressCallback | None,
    ) -> list[str]:
        """Fetch ``urls`` concurrently while reporting progress.

        Args:
            stage: Label passed to ``on_progress`` (``"list"`` / ``"detail"``).
            urls: URLs to fetch, in the order their results should be returned.
            force_refresh: Forwarded to :meth:`CachedClient.get_text`.
            on_progress: Optional progress callback.

        Returns:
            The response bodies, in the same order as ``urls``.
        """
        total = len(urls)
        if on_progress is not None:
            on_progress(stage, 0, total)
        if total == 0:
            return []
        completed = 0
        completed_lock = asyncio.Lock()

        async def _one(u: str) -> str:
            nonlocal completed
            body = await self.client.get_text(u, force_refresh=force_refresh)
            if on_progress is not None:
                async with completed_lock:
                    completed += 1
                    on_progress(stage, completed, total)
            return body

        return await asyncio.gather(*(_one(u) for u in urls))
