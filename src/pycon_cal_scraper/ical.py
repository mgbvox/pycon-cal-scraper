"""Render Event objects as an RFC 5545 VCALENDAR feed.

This is a tiny, dependency-free serializer — just enough to import a saved
list into Apple Calendar, Outlook, Fantastical, etc. without depending on
``icalendar`` for one feature.

The output is intentionally minimal:

* ``BEGIN:VCALENDAR`` ... ``END:VCALENDAR`` wrapper
* one ``VEVENT`` per :class:`Event`, with stable UIDs derived from
  :attr:`Event.id` so importing twice updates rather than duplicates
* ``DTSTART`` / ``DTEND`` with the IANA timezone name attached as ``TZID``
* ``SUMMARY``, ``LOCATION`` (room + venue), and a ``DESCRIPTION`` matching
  the gcal-sync description format
* ``X-PYCON-ID`` extension property so a future reverse-importer can match
  rows back to the saved-list

The serializer does *not* emit ``VTIMEZONE`` blocks — modern clients
resolve IANA TZIDs themselves and Apple/Google/Outlook all do this fine for
``America/Los_Angeles``. If we ever need bullet-proof Olson definitions,
swap to the ``icalendar`` package.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from pycon_cal_scraper.gcal.sync import build_location
from pycon_cal_scraper.models import Event

#: Producer string written into the ``PRODID`` header.
PRODID = "-//pycon-cal-scraper//EN"

#: RFC 5545 mandates CRLF line endings between properties.
CRLF = "\r\n"


def _ics_escape(text: str) -> str:
    """Escape a string for an iCalendar TEXT-typed property.

    RFC 5545 §3.3.11 — escape ``\\``, ``;``, ``,`` and replace literal newlines
    with the two-character sequence ``\\n``.
    """
    return (
        text.replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\r\n", "\\n")
        .replace("\n", "\\n")
        .replace("\r", "\\n")
    )


def _format_dt(value: datetime) -> tuple[str, str]:
    """Return ``(tzid, value)`` for a tz-aware datetime as ICS prefers it."""
    tzinfo = value.tzinfo
    tz_name = getattr(tzinfo, "key", None) or "UTC"
    return tz_name, value.strftime("%Y%m%dT%H%M%S")


def _fold_line(line: str) -> str:
    """Apply RFC 5545 line folding (max 75 octets per line).

    Continuation lines start with a single space; the fold happens at byte
    boundaries — counted as ASCII characters here, which is fine because we
    escape away all the special control bytes upstream.
    """
    if len(line) <= 75:
        return line
    chunks = [line[:75]]
    rest = line[75:]
    while rest:
        chunks.append(" " + rest[:74])
        rest = rest[74:]
    return CRLF.join(chunks)


def _description_lines(event: Event) -> list[str]:
    """Build the human-readable description body for an :class:`Event`."""
    lines: list[str] = []
    if event.speakers:
        lines.append("Speakers: " + ", ".join(event.speakers))
    if event.audience_level:
        lines.append(f"Audience: {event.audience_level}")
    if event.track:
        lines.append(f"Track: {event.track}")
    body = event.description or event.abstract
    if body:
        lines.append("")
        lines.append(body)
    lines.append("")
    lines.append(str(event.url))
    return lines


def event_to_vevent(event: Event, *, venue_address: str | None = None) -> str:
    """Render one :class:`Event` as a ``VEVENT`` block.

    Args:
        event: The event to render.
        venue_address: Optional venue postal address; combined with
            :attr:`Event.room` via :func:`gcal.sync.build_location` so a
            calendar tap opens Maps at the right building.

    Returns:
        A multi-line CRLF string containing the ``BEGIN:VEVENT`` ...
        ``END:VEVENT`` block, with property lines folded per RFC 5545.
    """
    start_tz, start_value = _format_dt(event.start)
    end_tz, end_value = _format_dt(event.end)
    uid = f"pycon-{event.id}@pycon-cal-scraper"
    dtstamp = datetime.now(tz=event.start.tzinfo).strftime("%Y%m%dT%H%M%SZ")
    location = build_location(event.room, venue_address)
    description = _ics_escape("\n".join(_description_lines(event)))
    summary = _ics_escape(event.title)
    lines: list[str] = [
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{dtstamp}",
        f"DTSTART;TZID={start_tz}:{start_value}",
        f"DTEND;TZID={end_tz}:{end_value}",
        f"SUMMARY:{summary}",
    ]
    if location:
        lines.append(f"LOCATION:{_ics_escape(location)}")
    lines.append(f"DESCRIPTION:{description}")
    lines.append(f"URL:{event.url}")
    lines.append(f"X-PYCON-ID:{event.id}")
    lines.append("END:VEVENT")
    return CRLF.join(_fold_line(line) for line in lines)


def events_to_ics(
    events: Iterable[Event],
    *,
    venue_address: str | None = None,
    calendar_name: str = "PyCon US — saved",
) -> str:
    """Render an iterable of :class:`Event` as a complete VCALENDAR feed.

    Args:
        events: The events to include — typically the resolved saved-list.
        venue_address: Optional venue postal address; threaded into each
            event's ``LOCATION``.
        calendar_name: Value of the ``X-WR-CALNAME`` extension, which most
            clients display as the calendar's display name on import.

    Returns:
        The full ICS payload as a single string, CRLF-terminated and ready
        to write to a ``.ics`` file.
    """
    lines: list[str] = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        f"PRODID:{PRODID}",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        f"X-WR-CALNAME:{_ics_escape(calendar_name)}",
    ]
    body = [event_to_vevent(e, venue_address=venue_address) for e in events]
    lines.extend(body)
    lines.append("END:VCALENDAR")
    return CRLF.join(lines) + CRLF
