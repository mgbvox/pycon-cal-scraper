"""Tests for the typer CLI surface."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from typer.testing import CliRunner

from pycon_cal_scraper import cli
from pycon_cal_scraper.models import Event, EventType
from pycon_cal_scraper.paths import events_file, saved_file
from pycon_cal_scraper.store import EventsStore

PACIFIC = ZoneInfo("America/Los_Angeles")


def _seed(eid: str, title: str = "Demo") -> Event:
    return Event.model_validate(
        {
            "id": eid,
            "url": f"https://us.pycon.org/2026/schedule/presentation/{eid}/",
            "title": title,
            "type": EventType.talk,
            "speakers": ["Pat"],
            "start": datetime(2026, 5, 15, 9, 0, tzinfo=PACIFIC),
            "end": datetime(2026, 5, 15, 9, 30, tzinfo=PACIFIC),
            "room": "Room 1",
        }
    )


@pytest.fixture
def runner(tmp_data_dir: Path) -> CliRunner:
    return CliRunner()


def test_search_without_cached_events_fails_helpfully(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["search", "anything"])
    assert result.exit_code == 1
    assert "sync" in result.stdout


def test_save_and_saved_roundtrip(runner: CliRunner) -> None:
    EventsStore(events_file()).save([_seed("1", "Async"), _seed("2", "Logging")])
    save = runner.invoke(cli.app, ["save", "1"])
    assert save.exit_code == 0
    assert "saved" in save.stdout

    listed = runner.invoke(cli.app, ["saved"])
    assert listed.exit_code == 0
    assert "Async" in listed.stdout
    assert "Logging" not in listed.stdout


def test_save_warns_on_unknown_id(runner: CliRunner) -> None:
    EventsStore(events_file()).save([_seed("1")])
    result = runner.invoke(cli.app, ["save", "999"])
    assert result.exit_code == 0
    assert "unknown event id" in result.stdout


def test_unsave_removes_id(runner: CliRunner) -> None:
    EventsStore(events_file()).save([_seed("1"), _seed("2")])
    runner.invoke(cli.app, ["save", "1"])
    runner.invoke(cli.app, ["save", "2"])
    result = runner.invoke(cli.app, ["unsave", "1"])
    assert result.exit_code == 0
    assert "removed" in result.stdout
    listed = runner.invoke(cli.app, ["saved"])
    assert "presentation/1/" not in listed.stdout  # ID 1 should be gone


def test_search_query_returns_matches(runner: CliRunner) -> None:
    EventsStore(events_file()).save([_seed("1", "Async patterns"), _seed("2", "Rust")])
    result = runner.invoke(cli.app, ["search", "async"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "Async patterns" in result.stdout
    assert "Rust" not in result.stdout


def test_search_query_no_matches(runner: CliRunner) -> None:
    EventsStore(events_file()).save([_seed("1", "Async")])
    result = runner.invoke(cli.app, ["search", "rust"])
    assert result.exit_code == 0
    assert "No matches" in result.stdout


def test_config_show_and_set(runner: CliRunner, tmp_data_dir: Path) -> None:
    show = runner.invoke(cli.app, ["config", "show"])
    assert show.exit_code == 0
    assert "calendar_id" in show.stdout

    setted = runner.invoke(
        cli.app, ["config", "set", "calendar_id", "test@group.calendar.google.com"]
    )
    assert setted.exit_code == 0

    show2 = runner.invoke(cli.app, ["config", "show"])
    assert "test@group.calendar.google.com" in show2.stdout


def test_config_set_rejects_unknown_key(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["config", "set", "nope", "x"])
    assert result.exit_code == 1
    assert "Unknown config key" in result.stdout


def test_config_set_search_results_limit_coerces_to_int(runner: CliRunner) -> None:
    setted = runner.invoke(cli.app, ["config", "set", "search_results_limit", "5"])
    assert setted.exit_code == 0

    show = runner.invoke(cli.app, ["config", "show"])
    assert show.exit_code == 0
    assert "search_results_limit = 5" in show.stdout


def test_config_set_search_results_limit_rejects_zero(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["config", "set", "search_results_limit", "0"])
    assert result.exit_code != 0


def test_search_respects_config_search_results_limit(runner: CliRunner) -> None:
    EventsStore(events_file()).save([_seed(str(i), f"Async event {i}") for i in range(10)])
    runner.invoke(cli.app, ["config", "set", "search_results_limit", "3"])

    result = runner.invoke(cli.app, ["search", "async"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    rendered_ids = sum(1 for line in result.stdout.splitlines() if "Async event" in line)
    assert rendered_ids == 3


def test_search_limit_flag_overrides_config(runner: CliRunner) -> None:
    EventsStore(events_file()).save([_seed(str(i), f"Async event {i}") for i in range(10)])
    runner.invoke(cli.app, ["config", "set", "search_results_limit", "10"])

    result = runner.invoke(cli.app, ["search", "--limit", "2", "async"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    rendered_ids = sum(1 for line in result.stdout.splitlines() if "Async event" in line)
    assert rendered_ids == 2


class _ReplOutCapture:
    """Replacement for ``cli.console`` that records ``console.print`` calls."""

    def __init__(self) -> None:
        self.buffer: list[str] = []

    def print(self, *args: object, **kwargs: object) -> None:
        self.buffer.append(" ".join(str(a) for a in args))

    @property
    def text(self) -> str:
        return "\n".join(self.buffer)


def _scripted_session(lines: list[str]) -> type:
    iterator = iter(lines)

    class _FakeSession:
        def __init__(self, *args: object, **kwargs: object) -> None: ...

        def prompt(self, *args: object, **kwargs: object) -> str:
            try:
                return next(iterator)
            except StopIteration as exc:
                raise EOFError from exc

    return _FakeSession


def test_repl_limit_command_changes_in_session_limit(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The REPL ``/limit N`` command updates the in-session match cap."""
    import pycon_cal_scraper.cli as cli_mod

    EventsStore(events_file()).save([_seed(str(i), f"Async event {i}") for i in range(10)])
    events = EventsStore(events_file()).load()

    capture = _ReplOutCapture()
    monkeypatch.setattr(cli_mod, "console", capture)
    monkeypatch.setattr(
        "prompt_toolkit.PromptSession", _scripted_session(["async", "/limit 2", "async", "/quit"])
    )

    cli_mod._run_repl(events, results_limit=5)

    assert "showing up to 5" in capture.text
    assert "showing up to 2" in capture.text


def test_embed_command_requires_events(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["embed"])
    assert result.exit_code == 1
    assert "sync" in result.stdout


def test_repl_warns_when_no_embedding_cache(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At REPL start, the user should be told if no embedding cache exists."""
    import pycon_cal_scraper.cli as cli_mod

    EventsStore(events_file()).save([_seed("1", "Async")])
    events = EventsStore(events_file()).load()

    capture = _ReplOutCapture()
    monkeypatch.setattr(cli_mod, "console", capture)
    monkeypatch.setattr("prompt_toolkit.PromptSession", _scripted_session(["/quit"]))

    cli_mod._run_repl(events, results_limit=5)

    assert "No embeddings cached" in capture.text


def test_repl_semantic_toggle_blocks_without_cache(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``/semantic`` should refuse to enable when no cache exists."""
    import pycon_cal_scraper.cli as cli_mod

    EventsStore(events_file()).save([_seed("1", "Async")])
    events = EventsStore(events_file()).load()

    capture = _ReplOutCapture()
    monkeypatch.setattr(cli_mod, "console", capture)
    monkeypatch.setattr(
        "prompt_toolkit.PromptSession",
        _scripted_session(["/semantic", "/quit"]),
    )

    cli_mod._run_repl(events, results_limit=5)

    # The toggle should NOT have flipped — no "semantic search enabled" message.
    assert "semantic search enabled" not in capture.text
    assert "No embeddings cached" in capture.text


def test_search_keyword_flag_uses_exact_token_count(runner: CliRunner) -> None:
    """``--keyword`` should rank by hit count and skip fuzzy near-misses."""
    EventsStore(events_file()).save(
        [
            _seed("1", "Rust patterns"),
            _seed("2", "Asynk patterns"),  # fuzzy near-miss for `async`
            _seed("3", "Other"),
        ]
    )
    # `--keyword rust` finds event 1 (exact), not 2 or 3.
    result = runner.invoke(cli.app, ["search", "--keyword", "rust"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.stdout
    assert " 1 " in result.stdout
    assert " 2 " not in result.stdout
    assert " 3 " not in result.stdout


def test_search_room_filter_keeps_only_matching_rooms(runner: CliRunner) -> None:
    """``--room`` should restrict matches by substring of the event's room."""
    EventsStore(events_file()).save(
        [
            _seed("1", "Async patterns"),  # default room "Room 1"
            _seed("2", "Async lessons"),
        ]
    )
    result = runner.invoke(cli.app, ["search", "--room", "Room 1", "async"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.stdout
    assert " 1 " in result.stdout
    assert " 2 " in result.stdout
    # A bogus room should match nothing.
    none = runner.invoke(cli.app, ["search", "--room", "ZZZ", "async"])
    assert "No matches" in none.stdout


def test_search_lexical_negative_excludes_keyword(runner: CliRunner) -> None:
    """``!python`` should drop events containing the word 'python'."""
    EventsStore(events_file()).save(
        [
            _seed("1", "Python at scale"),
            _seed("2", "Rust patterns"),
        ]
    )
    result = runner.invoke(cli.app, ["search", "patterns", "!python"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.stdout
    assert "Rust patterns" in result.stdout
    assert "Python at scale" not in result.stdout


def test_search_keyword_and_semantic_are_exclusive(runner: CliRunner) -> None:
    EventsStore(events_file()).save([_seed("1", "x")])
    result = runner.invoke(cli.app, ["search", "--keyword", "--semantic", "x"])
    assert result.exit_code == 1
    assert "Pick at most one" in result.stdout


def test_repl_keyword_mode_command(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    """``/keyword`` should switch the REPL into exact-token mode."""
    import pycon_cal_scraper.cli as cli_mod

    EventsStore(events_file()).save(
        [
            _seed("1", "Rust patterns"),
            _seed("2", "Asynk patterns"),
        ]
    )
    events = EventsStore(events_file()).load()

    capture = _ReplOutCapture()
    monkeypatch.setattr(cli_mod, "console", capture)
    monkeypatch.setattr(
        "prompt_toolkit.PromptSession",
        _scripted_session(["/keyword", "rust", "/quit"]),
    )

    cli_mod._run_repl(events, results_limit=5)

    assert "search mode: keyword" in capture.text
    # _ReplOutCapture stringifies the rendered Table to its repr; that's enough to
    # confirm matches were found rather than the REPL printing "no matches".
    assert "no matches" not in capture.text
    assert "<rich.table.Table" in capture.text


def test_gcal_clean_streams_scan_progress(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``gcal clean`` must announce scan state up-front and report when done."""

    class _FakeCreds:
        valid = True

    monkeypatch.setattr("pycon_cal_scraper.gcal.auth.load_cached_credentials", lambda: _FakeCreds())
    monkeypatch.setattr(
        "pycon_cal_scraper.gcal.auth.build_calendar_service", lambda creds: object()
    )

    async def _fake_list(*args: object, **kwargs: object) -> list[dict[str, object]]:
        # The CLI must pass an on_page callable; verify it.
        on_page = kwargs.get("on_page")
        assert callable(on_page)
        on_page(1, 100, 2)
        return []

    monkeypatch.setattr("pycon_cal_scraper.gcal.sync.list_managed_events", _fake_list)

    result = runner.invoke(cli.app, ["gcal", "clean", "--yes"])
    assert result.exit_code == 0
    assert "Scanning calendar" in result.stdout
    assert "Scan complete" in result.stdout


def test_embed_command_writes_cache(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    EventsStore(events_file()).save([_seed("1"), _seed("2")])

    class _StubVoyage:
        def __init__(self, *args: object, **kwargs: object) -> None: ...

        async def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

        async def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0]

    monkeypatch.setattr("pycon_cal_scraper.semantic.VoyageEmbedder", _StubVoyage)

    result = runner.invoke(cli.app, ["embed"])
    assert result.exit_code == 0, result.stdout
    assert "Embedded 2" in result.stdout

    from pycon_cal_scraper import paths as paths_mod
    from pycon_cal_scraper.semantic import EmbeddingCache

    cache = EmbeddingCache.load(paths_mod.embeddings_file(), model="voyage-3-lite")
    assert {"1", "2"} == cache.known_ids()


def test_embed_command_reports_missing_api_key(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    EventsStore(events_file()).save([_seed("1")])
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    result = runner.invoke(cli.app, ["embed"])
    assert result.exit_code == 1
    assert "VOYAGE_API_KEY" in result.stdout


def test_semantic_search_requires_embeddings(runner: CliRunner) -> None:
    EventsStore(events_file()).save([_seed("1")])
    result = runner.invoke(cli.app, ["search", "--semantic", "async"])
    assert result.exit_code == 1
    assert "No embeddings cached" in result.stdout


def test_semantic_search_uses_voyage(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    EventsStore(events_file()).save([_seed("1", "Async"), _seed("2", "Rust")])

    import numpy as np

    from pycon_cal_scraper import paths as paths_mod
    from pycon_cal_scraper.semantic import EmbeddingCache

    cache = EmbeddingCache(paths_mod.embeddings_file(), model="voyage-3-lite")
    cache.put("1", np.array([1.0, 0.0], dtype=np.float32))
    cache.put("2", np.array([0.0, 1.0], dtype=np.float32))
    cache.save()

    class _StubVoyage:
        def __init__(self, *args: object, **kwargs: object) -> None: ...

        async def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.0, 0.0] for _ in texts]

        async def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0]  # closest to event "1"

    monkeypatch.setattr("pycon_cal_scraper.semantic.VoyageEmbedder", _StubVoyage)

    result = runner.invoke(
        cli.app, ["search", "--semantic", "asynchronous"], env={"COLUMNS": "200"}
    )
    assert result.exit_code == 0, result.stdout
    # Event "1" (Async) should appear before "2" (Rust).
    pos_async = result.stdout.find(" 1 ")
    pos_rust = result.stdout.find(" 2 ")
    assert pos_async != -1 and (pos_rust == -1 or pos_async < pos_rust)


def test_sync_warns_when_saved_event_drifts(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If a saved event's fields change between scrapes, sync should hint."""
    old = _seed("1", "Old title")
    EventsStore(events_file()).save([old])
    saved_file().write_text(
        json.dumps([{"id": "1", "saved_at": "2026-05-10T00:00:00+00:00"}]), encoding="utf-8"
    )
    new = _seed("1", "New title")

    class _FakeScraper:
        def __init__(self, *args: object, **kwargs: object) -> None: ...

        async def fetch_all(self, **kwargs: object) -> list[Event]:
            return [new]

    monkeypatch.setattr(cli, "Scraper", _FakeScraper)

    class _FakeClient:
        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(self, *exc: object) -> None: ...

    monkeypatch.setattr(cli, "CachedClient", lambda *a, **k: _FakeClient())

    result = runner.invoke(cli.app, ["sync"])
    assert result.exit_code == 0, result.stdout
    assert "saved event(s) changed" in result.stdout
    assert "gcal sync" in result.stdout


def test_sync_does_not_warn_when_no_drift(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    event = _seed("1", "Stable")
    EventsStore(events_file()).save([event])
    saved_file().write_text(
        json.dumps([{"id": "1", "saved_at": "2026-05-10T00:00:00+00:00"}]), encoding="utf-8"
    )

    class _FakeScraper:
        def __init__(self, *args: object, **kwargs: object) -> None: ...

        async def fetch_all(self, **kwargs: object) -> list[Event]:
            return [event]

    monkeypatch.setattr(cli, "Scraper", _FakeScraper)

    class _FakeClient:
        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(self, *exc: object) -> None: ...

    monkeypatch.setattr(cli, "CachedClient", lambda *a, **k: _FakeClient())

    result = runner.invoke(cli.app, ["sync"])
    assert result.exit_code == 0, result.stdout
    assert "saved event(s) changed" not in result.stdout


def test_search_day_filter_restricts_results(runner: CliRunner) -> None:
    fri = _seed("1", "Fri talk")
    sat = Event.model_validate(
        {
            **fri.model_dump(mode="json"),
            "id": "2",
            "title": "Sat talk",
            "start": datetime(2026, 5, 16, 9, 0, tzinfo=PACIFIC).isoformat(),
            "end": datetime(2026, 5, 16, 9, 30, tzinfo=PACIFIC).isoformat(),
        }
    )
    EventsStore(events_file()).save([fri, sat])

    result = runner.invoke(cli.app, ["search", "--day", "fri", "talk"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "Fri talk" in result.stdout
    assert "Sat talk" not in result.stdout


def test_search_from_to_filters_window(runner: CliRunner) -> None:
    morning = _seed("1", "Morning talk")
    afternoon = Event.model_validate(
        {
            **morning.model_dump(mode="json"),
            "id": "2",
            "title": "Afternoon talk",
            "start": datetime(2026, 5, 15, 14, 0, tzinfo=PACIFIC).isoformat(),
            "end": datetime(2026, 5, 15, 15, 0, tzinfo=PACIFIC).isoformat(),
        }
    )
    EventsStore(events_file()).save([morning, afternoon])

    result = runner.invoke(
        cli.app,
        ["search", "--from", "2026-05-15T13:00", "--to", "2026-05-15T16:00", "talk"],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0
    assert "Afternoon talk" in result.stdout
    assert "Morning talk" not in result.stdout


def test_search_rejects_invalid_day(runner: CliRunner) -> None:
    EventsStore(events_file()).save([_seed("1")])
    result = runner.invoke(cli.app, ["search", "--day", "xyz", "anything"])
    assert result.exit_code == 1
    assert "can't parse" in result.stdout


def test_search_marks_overlap_with_saved(runner: CliRunner) -> None:
    """Events overlapping a saved event should render with the ⚠ marker."""
    saved_event = _seed("1", "Already saved")
    overlapping = Event.model_validate(
        {
            **saved_event.model_dump(mode="json"),
            "id": "2",
            "title": "Overlapper",
        }
    )
    EventsStore(events_file()).save([saved_event, overlapping])
    saved_file().write_text(
        json.dumps([{"id": "1", "saved_at": "2026-05-10T00:00:00+00:00"}]), encoding="utf-8"
    )

    result = runner.invoke(cli.app, ["search", "overlapper"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    overlap_lines = [line for line in result.stdout.splitlines() if "Overlapper" in line]
    assert overlap_lines, result.stdout
    assert "⚠" in overlap_lines[0]


def test_gcal_clean_aborts_when_user_declines(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeCreds:
        valid = True

    monkeypatch.setattr("pycon_cal_scraper.gcal.auth.load_cached_credentials", lambda: _FakeCreds())
    monkeypatch.setattr(
        "pycon_cal_scraper.gcal.auth.build_calendar_service", lambda creds: object()
    )

    async def _fake_list(*args: object, **kwargs: object) -> list[dict[str, object]]:
        return [
            {
                "id": "gcal-1",
                "summary": "Hi",
                "start": {"dateTime": "2026-05-15T09:00:00-07:00"},
                "extendedProperties": {"private": {"pycon_id": "1"}},
            }
        ]

    monkeypatch.setattr("pycon_cal_scraper.gcal.sync.list_managed_events", _fake_list)

    apply_called = False

    async def _fake_apply(*args: object, **kwargs: object) -> None:
        nonlocal apply_called
        apply_called = True

    monkeypatch.setattr("pycon_cal_scraper.gcal.sync.apply", _fake_apply)

    result = runner.invoke(cli.app, ["gcal", "clean"], input="n\n")
    assert result.exit_code == 0
    assert "Aborted" in result.stdout
    assert apply_called is False


def test_gcal_clean_deletes_when_confirmed(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeCreds:
        valid = True

    monkeypatch.setattr("pycon_cal_scraper.gcal.auth.load_cached_credentials", lambda: _FakeCreds())
    monkeypatch.setattr(
        "pycon_cal_scraper.gcal.auth.build_calendar_service", lambda creds: object()
    )

    async def _fake_list(*args: object, **kwargs: object) -> list[dict[str, object]]:
        return [
            {
                "id": "gcal-1",
                "summary": "Hi",
                "start": {"dateTime": "2026-05-15T09:00:00-07:00"},
                "extendedProperties": {"private": {"pycon_id": "1"}},
            },
            {
                "id": "gcal-2",
                "summary": "There",
                "start": {"dateTime": "2026-05-15T10:00:00-07:00"},
                "extendedProperties": {"private": {"pycon_id": "2"}},
            },
        ]

    monkeypatch.setattr("pycon_cal_scraper.gcal.sync.list_managed_events", _fake_list)

    captured: list[object] = []

    async def _fake_apply(service: object, plan: object, **kwargs: object) -> None:
        captured.append(plan)

    monkeypatch.setattr("pycon_cal_scraper.gcal.sync.apply", _fake_apply)

    result = runner.invoke(cli.app, ["gcal", "clean", "--yes"])
    assert result.exit_code == 0, result.stdout
    assert "Removed 2" in result.stdout
    assert len(captured) == 1
    plan = captured[0]
    assert sorted(plan.to_delete) == ["gcal-1", "gcal-2"]


def test_gcal_clean_no_managed_events(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeCreds:
        valid = True

    monkeypatch.setattr("pycon_cal_scraper.gcal.auth.load_cached_credentials", lambda: _FakeCreds())
    monkeypatch.setattr(
        "pycon_cal_scraper.gcal.auth.build_calendar_service", lambda creds: object()
    )

    async def _fake_list(*args: object, **kwargs: object) -> list[dict[str, object]]:
        return []

    monkeypatch.setattr("pycon_cal_scraper.gcal.sync.list_managed_events", _fake_list)

    result = runner.invoke(cli.app, ["gcal", "clean", "--yes"])
    assert result.exit_code == 0
    assert "No managed events" in result.stdout


def test_repl_limit_command_rejects_garbage(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``/limit foo`` should print a usage hint and not change the limit."""
    import pycon_cal_scraper.cli as cli_mod

    EventsStore(events_file()).save([_seed("1")])
    events = EventsStore(events_file()).load()

    capture = _ReplOutCapture()
    monkeypatch.setattr(cli_mod, "console", capture)
    monkeypatch.setattr(
        "prompt_toolkit.PromptSession",
        _scripted_session(["/limit foo", "/limit 0", "/limit", "/quit"]),
    )

    cli_mod._run_repl(events, results_limit=5)

    # The "usage" message appears for each of the three bad invocations.
    assert capture.text.count("usage: /limit") == 3


def test_sync_command_writes_events(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    """The sync command pulls events via the scraper and writes them to disk."""

    fake_events = [_seed("1", "Hello")]

    class _FakeScraper:
        def __init__(self, *args: object, **kwargs: object) -> None: ...

        async def fetch_all(self, **kwargs: object) -> list[Event]:
            return fake_events

    monkeypatch.setattr(cli, "Scraper", _FakeScraper)

    class _FakeClient:
        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(self, *exc: object) -> None: ...

    monkeypatch.setattr(cli, "CachedClient", lambda *a, **k: _FakeClient())

    result = runner.invoke(cli.app, ["sync"])
    assert result.exit_code == 0, result.stdout
    assert "Cached 1 events" in result.stdout

    loaded = json.loads(events_file().read_text())
    assert loaded[0]["id"] == "1"


def test_gcal_login_requires_client_secret(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["gcal", "login"])
    assert result.exit_code == 1
    assert "client_secret_path" in result.stdout


def test_gcal_sync_requires_token(runner: CliRunner) -> None:
    EventsStore(events_file()).save([_seed("1")])
    saved_file().write_text(
        json.dumps([{"id": "1", "saved_at": "2026-05-10T00:00:00+00:00"}]), encoding="utf-8"
    )
    result = runner.invoke(cli.app, ["gcal", "sync"])
    assert result.exit_code == 1
    assert "gcal login" in result.stdout


def test_gcal_sync_no_saved_events(runner: CliRunner) -> None:
    EventsStore(events_file()).save([_seed("1")])
    result = runner.invoke(cli.app, ["gcal", "sync"])
    assert result.exit_code == 0
    assert "nothing to sync" in result.stdout
