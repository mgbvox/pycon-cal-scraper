"""Tests for the schedule scraper, using captured HTML fixtures."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from pycon_cal_scraper.models import EventType
from pycon_cal_scraper.scraper import (
    SCHEDULE_PAGES,
    Scraper,
    parse_list_page,
    parse_presentation_detail,
)

PACIFIC = ZoneInfo("America/Los_Angeles")


@pytest.fixture
def talks_events(read_fixture: Callable[[str], str]) -> list[dict[str, object]]:
    html = read_fixture("talks_list.html")
    return parse_list_page(html, page_url="https://us.pycon.org/2026/schedule/talks/")


def test_parse_talks_list_extracts_events(talks_events: list[dict[str, object]]) -> None:
    # PyCon's talks page renders each event in both a grid view and a list view; we parse
    # the grid view only. Across Fri-Sun there are ~60 slot-talk + a handful of security/ai
    # /lightning-talks variants. >=70 is a generous floor that catches major regressions.
    talk_entries = [e for e in talks_events if e["type"] is EventType.talk]
    assert len(talk_entries) >= 70
    sample = talk_entries[0]
    assert sample["id"].isdigit()
    assert sample["title"]
    assert sample["url"].startswith("https://us.pycon.org/2026/schedule/presentation/")
    assert isinstance(sample["start"], datetime)
    assert isinstance(sample["end"], datetime)
    assert sample["start"].tzinfo is not None
    assert sample["end"] > sample["start"]


def test_parse_talks_list_includes_keynotes(talks_events: list[dict[str, object]]) -> None:
    keynotes = [e for e in talks_events if e["type"] is EventType.keynote]
    assert keynotes, "expected at least one keynote in the talks page"
    k = keynotes[0]
    assert k["id"].startswith("keynote:")
    assert "Keynote" in k["title"]


def test_parse_talks_list_skips_breaks(talks_events: list[dict[str, object]]) -> None:
    assert all(e["type"] is not EventType.break_ for e in talks_events)


def test_parse_talks_list_resolves_times(talks_events: list[dict[str, object]]) -> None:
    # First talk should land on 2026-05-15 (Friday) and be in Pacific time.
    first_talk = next(e for e in talks_events if e["type"] is EventType.talk)
    assert first_talk["start"].date().isoformat() == "2026-05-15"
    assert first_talk["start"].tzinfo == PACIFIC


def test_parse_talks_list_extracts_room_and_speakers(talks_events: list[dict[str, object]]) -> None:
    first_talk = next(e for e in talks_events if e["type"] is EventType.talk and e["speakers"])
    assert isinstance(first_talk["speakers"], list)
    assert all(isinstance(s, str) for s in first_talk["speakers"])
    assert first_talk["room"]


def test_parse_tutorials_list(read_fixture: Callable[[str], str]) -> None:
    html = read_fixture("tutorials_list.html")
    events = parse_list_page(html, page_url="https://us.pycon.org/2026/schedule/tutorials/")
    tutorials = [e for e in events if e["type"] is EventType.tutorial]
    assert len(tutorials) >= 20
    # Tutorials run on May 13-14, 2026.
    dates = {e["start"].date().isoformat() for e in tutorials}
    assert dates <= {"2026-05-13", "2026-05-14"}


def test_parse_sponsor_list(read_fixture: Callable[[str], str]) -> None:
    html = read_fixture("sponsor_list.html")
    events = parse_list_page(
        html, page_url="https://us.pycon.org/2026/schedule/sponsor-presentations/"
    )
    sponsors = [e for e in events if e["type"] is EventType.sponsor]
    assert len(sponsors) >= 1


def test_parse_presentation_detail(read_fixture: Callable[[str], str]) -> None:
    html = read_fixture("presentation_2.html")
    detail = parse_presentation_detail(html)
    assert detail["title"] == "Build your first MCP server in Python"
    assert detail["speakers"] == ["Pamela Fox"]
    assert detail["audience_level"] == "Some experience"
    assert detail["description"]
    assert "MCP" in (detail["description"] or "")


def test_parse_presentation_detail_handles_multi_speaker(
    read_fixture: Callable[[str], str],
) -> None:
    html = read_fixture("presentation_19.html")
    detail = parse_presentation_detail(html)
    assert detail["speakers"]
    assert detail["title"]


async def test_scraper_end_to_end_with_fixtures(read_fixture: Callable[[str], str]) -> None:
    """Stub the http client to serve fixtures, then run the full Scraper."""

    pages = {
        "https://us.pycon.org/2026/schedule/talks/": read_fixture("talks_list.html"),
        "https://us.pycon.org/2026/schedule/tutorials/": read_fixture("tutorials_list.html"),
        "https://us.pycon.org/2026/schedule/sponsor-presentations/": read_fixture(
            "sponsor_list.html"
        ),
        "https://us.pycon.org/2026/schedule/presentation/2/": read_fixture("presentation_2.html"),
        "https://us.pycon.org/2026/schedule/presentation/19/": read_fixture("presentation_19.html"),
        "https://us.pycon.org/2026/schedule/presentation/24/": read_fixture("presentation_24.html"),
    }

    class StubClient:
        async def get_text(self, url: str, *, force_refresh: bool = False) -> str:
            if url in pages:
                return pages[url]
            return "<html><body><h1>Stub</h1></body></html>"

    scraper = Scraper(client=StubClient(), pages=SCHEDULE_PAGES[:3])
    events = await scraper.fetch_all()
    assert len(events) >= 70
    # Detail-page enrichment should populate description for events we have fixtures for.
    by_id = {e.id: e for e in events}
    assert "2" in by_id
    assert by_id["2"].description
    assert by_id["2"].audience_level == "Some experience"


async def test_scraper_fires_progress_callbacks(read_fixture: Callable[[str], str]) -> None:
    """Each stage should emit an initial 0/N event and a final N/N event."""

    pages = {
        "https://us.pycon.org/2026/schedule/talks/": read_fixture("talks_list.html"),
        "https://us.pycon.org/2026/schedule/tutorials/": read_fixture("tutorials_list.html"),
        "https://us.pycon.org/2026/schedule/sponsor-presentations/": read_fixture(
            "sponsor_list.html"
        ),
    }

    class StubClient:
        async def get_text(self, url: str, *, force_refresh: bool = False) -> str:
            return pages.get(url, "<html><body><h1>Stub</h1></body></html>")

    events: list[tuple[str, int, int]] = []

    def on_progress(stage: str, completed: int, total: int) -> None:
        events.append((stage, completed, total))

    scraper = Scraper(client=StubClient(), pages=SCHEDULE_PAGES[:3])
    await scraper.fetch_all(on_progress=on_progress)

    stages = {stage for stage, _, _ in events}
    assert {"list", "detail"} <= stages

    # Each stage starts with 0/N and ends with N/N where N > 0.
    for stage in ("list", "detail"):
        stage_events = [(c, t) for s, c, t in events if s == stage]
        assert stage_events[0][0] == 0
        final_c, final_t = stage_events[-1]
        assert final_t > 0
        assert final_c == final_t
