"""Typer-based CLI for pycon-cal-scraper.

Sub-apps are mounted at ``gcal`` (Google Calendar integration) and
``config`` (user configuration); the rest of the commands live at the top
level.

Long-running, network-bounded commands (``sync``, ``gcal sync``) render a
rich progress bar driven by the progress callbacks exposed by
:mod:`pycon_cal_scraper.scraper` and :mod:`pycon_cal_scraper.gcal.sync`.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from pycon_cal_scraper import paths
from pycon_cal_scraper.conference import CONFERENCE_TZ
from pycon_cal_scraper.config import UserConfig, load_config, save_config
from pycon_cal_scraper.filters import (
    event_overlaps_any,
    events_happening_at,
    filter_events_by_day,
    filter_events_by_room,
    filter_events_by_window,
    find_conflict_groups,
    parse_day_token,
    parse_when,
    upcoming_events,
)
from pycon_cal_scraper.http_cache import CachedClient
from pycon_cal_scraper.models import Event
from pycon_cal_scraper.scraper import Scraper, schedule_pages_for
from pycon_cal_scraper.search import (
    ParsedQuery,
    SearchWeights,
    apply_lexical_negatives,
    keyword_search,
    parse_query,
    search,
)
from pycon_cal_scraper.store import EventsStore, SavedStore

app = typer.Typer(
    help="Scrape the PyCon US schedule, search, save, and sync to Google Calendar.",
    no_args_is_help=True,
)
gcal_app = typer.Typer(help="Google Calendar integration.", no_args_is_help=True)
config_app = typer.Typer(help="Show/edit user configuration.", no_args_is_help=True)
app.add_typer(gcal_app, name="gcal")
app.add_typer(config_app, name="config")

console = Console()


def _events_store() -> EventsStore:
    """Return the configured :class:`EventsStore`."""
    return EventsStore(paths.events_file())


def _saved_store() -> SavedStore:
    """Return the configured :class:`SavedStore`."""
    return SavedStore(paths.saved_file())


def _format_dt(event: Event) -> str:
    """Format an event's start as a compact day/time label."""
    return event.start.strftime("%a %m-%d %H:%M")


_FIELD_TO_LETTER = {"title": "T", "speakers": "S", "abstract": "A"}


def _format_match_fields(fields: frozenset[str]) -> str:
    """Render a matched-fields set as a stable T/S/A string for the ``Where`` column."""
    return "".join(_FIELD_TO_LETTER[f] for f in ("title", "speakers", "abstract") if f in fields)


def _render_events_table(
    events: list[Event],
    *,
    saved_ids: set[str] | None = None,
    saved_events: list[Event] | None = None,
    scores: dict[str, float] | None = None,
    matches: dict[str, frozenset[str]] | None = None,
) -> Table:
    """Build a rich :class:`Table` view over a list of events.

    Args:
        events: The events to render, in display order.
        saved_ids: Ids currently in the saved-list — used to render a star
            in the first column.
        saved_events: The saved events themselves, used to mark rows that
            overlap with any saved event. Pass ``None`` to skip the check.
        scores: Optional id -> similarity-score map. When supplied, an
            extra ``Score`` column is rendered.
        matches: Optional id -> matched-fields map. When supplied, an
            extra ``Where`` column flags which fields contained the
            query — ``T``, ``S``, ``A`` for title, speakers, abstract.

    Returns:
        A configured :class:`rich.table.Table`.
    """
    table = Table(show_header=True, header_style="bold")
    table.add_column("★", width=1)
    table.add_column("⚠", width=1)
    table.add_column("ID", width=8)
    if scores is not None:
        table.add_column("Score", width=6, justify="right")
    if matches is not None:
        table.add_column("Where", width=5)
    table.add_column("When", width=14)
    table.add_column("Type", width=10)
    table.add_column("Title")
    table.add_column("Speakers")
    table.add_column("Room")
    saved_set = list(saved_events) if saved_events else []
    for e in events:
        saved_flag = "[yellow]★[/yellow]" if saved_ids and e.id in saved_ids else ""
        # Only flag conflicts for events the user *hasn't* already saved.
        overlap = (
            "[red]⚠[/red]"
            if saved_set
            and (not saved_ids or e.id not in saved_ids)
            and event_overlaps_any(e, saved_set)
            else ""
        )
        row = [saved_flag, overlap, e.id]
        if scores is not None:
            row.append(f"{scores.get(e.id, 0.0):.3f}")
        if matches is not None:
            row.append(_format_match_fields(matches.get(e.id, frozenset())))
        row.extend(
            [
                _format_dt(e),
                e.type.value,
                e.title,
                ", ".join(e.speakers),
                e.room or "",
            ]
        )
        table.add_row(*row)
    return table


@contextmanager
def _progress() -> Iterator[Progress]:
    """A consistent ``rich.progress.Progress`` style for the whole CLI.

    Yields:
        A configured progress instance that the caller can ``add_task`` to.
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
        transient=True,  # Progress bar disappears once the work completes.
    )
    with progress:
        yield progress


# --- sync ---------------------------------------------------------------------


_STAGE_LABELS = {
    "list": "[cyan]Schedule pages[/cyan]",
    "detail": "[cyan]Talk details   [/cyan]",
}


def _search_weights(cfg: UserConfig) -> SearchWeights:
    """Build a :class:`SearchWeights` bundle from a :class:`UserConfig`."""
    return SearchWeights(
        title=cfg.search_weight_title,
        speaker=cfg.search_weight_speaker,
        abstract=cfg.search_weight_abstract,
    )


async def _do_sync(refresh: bool, progress: Progress) -> list[Event]:
    """Run a full scrape, driving ``progress`` along the way.

    Args:
        refresh: Forwarded to :meth:`Scraper.fetch_all` as ``force_refresh``.
        progress: The progress instance to render into.

    Returns:
        The freshly-fetched events. Persisting is the caller's job.
    """
    from datetime import timedelta

    cfg = load_config()
    tasks: dict[str, TaskID] = {}

    def on_progress(stage: str, completed: int, total: int) -> None:
        task_id = tasks.get(stage)
        if task_id is None:
            label = _STAGE_LABELS.get(stage, stage)
            tasks[stage] = progress.add_task(label, total=total)
        else:
            progress.update(task_id, completed=completed, total=total)

    async with CachedClient(
        cache_dir=paths.http_cache_dir(),
        ttl=timedelta(hours=cfg.http_cache_ttl_hours),
        min_interval=cfg.http_min_interval_seconds,
        concurrency=cfg.http_concurrency,
        user_agent=cfg.http_user_agent,
    ) as client:
        scraper = Scraper(client=client, pages=schedule_pages_for(cfg.scraper_base_url))
        return await scraper.fetch_all(force_refresh=refresh, on_progress=on_progress)


def _saved_payload(event: Event) -> dict[str, object]:
    """Return the fields whose drift should trigger a gcal-sync reminder.

    Only fields that map onto a Google Calendar event body are compared, so
    cosmetic changes to scraper internals don't trigger spurious prompts.
    """
    return {
        "title": event.title,
        "start": event.start.isoformat(),
        "end": event.end.isoformat(),
        "room": event.room,
        "speakers": list(event.speakers),
        "description": event.description,
        "abstract": event.abstract,
    }


def _drifted_saved_ids(before: list[Event], after: list[Event], saved_ids: set[str]) -> set[str]:
    """Return saved ids whose payload changed between ``before`` and ``after``."""
    before_map = {e.id: _saved_payload(e) for e in before if e.id in saved_ids}
    after_map = {e.id: _saved_payload(e) for e in after if e.id in saved_ids}
    drifted: set[str] = set()
    for eid, payload in after_map.items():
        if eid in before_map and before_map[eid] != payload:
            drifted.add(eid)
    return drifted


def _cancelled_saved_ids(after: list[Event], saved_ids: set[str]) -> set[str]:
    """Return saved ids that no longer appear in the freshly scraped events.

    A "cancelled" event is one the user previously saved that has dropped off
    the schedule entirely — typically because the conference removed it. The
    caller is responsible for telling the user; this helper is purely set
    arithmetic.
    """
    return saved_ids - {e.id for e in after}


@app.command()
def sync(
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Ignore the HTTP cache and re-fetch everything.")
    ] = False,
) -> None:
    """Fetch the PyCon schedule and update the on-disk event cache."""
    console.print("[cyan]Pulling schedule from us.pycon.org...[/cyan]")
    store = _events_store()
    before = store.load()
    saved_ids = _saved_store().ids()
    with _progress() as progress:
        events = asyncio.run(_do_sync(refresh, progress))
    store.save(events)
    console.print(f"[green]Cached {len(events)} events to {paths.events_file()}[/green]")
    if saved_ids:
        cancelled = _cancelled_saved_ids(events, saved_ids)
        if cancelled:
            console.print(
                f"[red]⚠ {len(cancelled)} saved event(s) no longer appear on the "
                f"schedule (likely cancelled): {', '.join(sorted(cancelled))}.[/red] "
                f"Use [bold]pycon-cal-scraper unsave <id>[/bold] to clean them up, "
                f"then [bold]gcal sync --prune[/bold] to remove the calendar entries."
            )
        drifted = _drifted_saved_ids(before, events, saved_ids)
        if drifted:
            console.print(
                f"[yellow]⚠ {len(drifted)} saved event(s) changed since last scrape "
                f"({', '.join(sorted(drifted))}). Run [bold]gcal sync[/bold] to push the "
                f"updates to your calendar.[/yellow]"
            )


# --- embed --------------------------------------------------------------------


async def _do_embed(progress: Progress, rebuild: bool) -> tuple[int, int]:
    """Embed any uncached events using Voyage AI.

    Args:
        progress: Active rich progress instance.
        rebuild: When ``True``, start from an empty cache (re-embeds every
            event).

    Returns:
        ``(newly_embedded, total_in_cache)``.
    """
    from pycon_cal_scraper.semantic import (
        EmbeddingCache,
        VoyageEmbedder,
        embed_events,
    )

    cfg = load_config()
    events = _events_store().load()
    cache_path = paths.embeddings_file()
    if rebuild and cache_path.exists():
        cache_path.unlink()
    cache = EmbeddingCache.load(cache_path, model=cfg.embedding_model)
    embedder = VoyageEmbedder(model=cfg.embedding_model, api_key_env=cfg.voyage_api_key_env)

    task_id: TaskID | None = None

    def on_progress(completed: int, total: int) -> None:
        nonlocal task_id
        if task_id is None:
            task_id = progress.add_task("[cyan]Embedding events[/cyan]", total=total)
        progress.update(task_id, completed=completed, total=total)

    before = len(cache)
    await embed_events(
        events, embedder, cache, batch_size=cfg.embedding_batch_size, on_progress=on_progress
    )
    after = len(cache)
    return after - before, after


@app.command()
def embed(
    rebuild: Annotated[
        bool,
        typer.Option("--rebuild", help="Discard the existing cache and re-embed every event."),
    ] = False,
) -> None:
    """Embed scraped events with Voyage AI for semantic search.

    Requires the ``VOYAGE_API_KEY`` environment variable to be set.
    """
    events = _events_store().load()
    if not events:
        console.print("[yellow]No events cached yet. Run `pycon-cal-scraper sync`.[/yellow]")
        raise typer.Exit(code=1)
    try:
        with _progress() as progress:
            new_count, total = asyncio.run(_do_embed(progress, rebuild=rebuild))
    except RuntimeError as exc:
        # Raised by VoyageEmbedder when the API key is missing.
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    if new_count == 0:
        console.print(f"[green]Embeddings up to date.[/green] ({total} cached)")
    else:
        console.print(
            f"[green]Embedded {new_count} new event(s).[/green] ({total} cached in total)"
        )


# --- search & list ------------------------------------------------------------


@app.command()
def search_cmd(
    query: Annotated[
        list[str] | None,
        typer.Argument(help="Search terms; omit to enter an interactive REPL."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            "-n",
            help="Override the configured search_results_limit for this invocation.",
            min=1,
        ),
    ] = None,
    semantic: Annotated[
        bool,
        typer.Option(
            "--semantic",
            "-s",
            help="Use Voyage AI semantic embeddings instead of lexical search.",
        ),
    ] = False,
    keyword: Annotated[
        bool,
        typer.Option(
            "--keyword",
            "-k",
            help="Exact-token keyword search: rank by total weighted hit count, no fuzz.",
        ),
    ] = False,
    day: Annotated[
        str | None,
        typer.Option("--day", help="Restrict to one day (mon-sun or YYYY-MM-DD)."),
    ] = None,
    from_: Annotated[
        str | None,
        typer.Option("--from", help="Only events ending on or after this datetime."),
    ] = None,
    to: Annotated[
        str | None,
        typer.Option("--to", help="Only events starting on or before this datetime."),
    ] = None,
    room: Annotated[
        str | None,
        typer.Option("--room", help="Restrict to events whose room contains this substring."),
    ] = None,
) -> None:
    """Search scraped events. With no QUERY, opens an interactive REPL.

    Query syntax: bare words / phrases are positive matches; words prefixed
    with ``!`` are excluded (``!python``); ``!"some phrase"`` semantically
    excludes events similar to that phrase (or substring-excludes when no
    embedding cache is available).
    """
    if semantic and keyword:
        console.print("[red]Pick at most one of --semantic / --keyword.[/red]")
        raise typer.Exit(code=1)
    events = _events_store().load()
    if not events:
        console.print("[yellow]No events cached yet. Run `pycon-cal-scraper sync`.[/yellow]")
        raise typer.Exit(code=1)

    saved_store = _saved_store()
    saved_ids = saved_store.ids()
    saved_events = saved_store.resolve(events)
    effective_limit = limit if limit is not None else load_config().search_results_limit

    try:
        candidates = _apply_time_filters(events, day=day, from_=from_, to=to)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    if room:
        candidates = filter_events_by_room(candidates, room)

    mode = "semantic" if semantic else ("keyword" if keyword else "lexical")
    if not query:
        _run_repl(events, results_limit=effective_limit, mode=mode, room=room)
        return

    raw_query = " ".join(query)
    parsed = parse_query(raw_query)
    if not parsed.positive and not parsed.lexical_negatives and not parsed.semantic_negatives:
        console.print("[yellow]Empty query.[/yellow]")
        return
    matches, score_map, match_map = _execute_search(
        candidates, parsed, mode=mode, limit=effective_limit
    )
    if matches is None:  # semantic path failure already loud
        raise typer.Exit(code=1)
    if not matches:
        console.print("[yellow]No matches.[/yellow]")
        return
    console.print(
        _render_events_table(
            matches,
            saved_ids=saved_ids,
            saved_events=saved_events,
            scores=score_map,
            matches=match_map,
        )
    )


def _execute_search(
    candidates: list[Event],
    parsed: ParsedQuery,
    *,
    mode: str,
    limit: int,
) -> tuple[list[Event] | None, dict[str, float] | None, dict[str, frozenset[str]] | None]:
    """Run the configured positive search, then apply negatives.

    Returns ``(matches, scores, match_fields)``. ``matches=None`` signals
    that the underlying semantic search failed loudly; callers should
    propagate as a non-zero exit / REPL-skip.
    """
    weights = _search_weights(load_config())
    # Apply lexical negatives BEFORE the positive search to shrink the candidate pool.
    candidates = apply_lexical_negatives(candidates, parsed.lexical_negatives)
    score_map: dict[str, float] | None = None
    match_map: dict[str, frozenset[str]] | None = None
    positive = parsed.positive.strip()
    if mode == "semantic":
        if not positive:
            # Pure-negative semantic searches don't make sense — every uncached
            # event would survive. Just return everything that survived neg filtering.
            matches: list[Event] = candidates[:limit]
        else:
            scored = _try_semantic_search(candidates, positive, limit)
            if scored is None:
                return None, None, None
            matches = [e for e, _ in scored]
            score_map = {e.id: s for e, s in scored}
    elif mode == "keyword":
        if not positive:
            matches = candidates[:limit]
        else:
            kw_results = keyword_search(candidates, positive, weights=weights)[:limit]
            matches = [e for e, _, _ in kw_results]
            score_map = {e.id: float(s) for e, s, _ in kw_results}
            match_map = {e.id: fields for e, _, fields in kw_results}
    else:
        if not positive:
            matches = candidates[:limit]
        else:
            matches = search(candidates, positive, weights=weights)[:limit]
    matches = _apply_semantic_negatives(matches, parsed.semantic_negatives)
    return matches, score_map, match_map


def _apply_time_filters(
    events: list[Event], *, day: str | None, from_: str | None, to: str | None
) -> list[Event]:
    """Apply the optional ``--day`` / ``--from`` / ``--to`` filters in order."""
    out = events
    if day is not None:
        out = filter_events_by_day(out, parse_day_token(day))
    start = parse_when(from_) if from_ else None
    end = parse_when(to) if to else None
    if start is not None or end is not None:
        out = filter_events_by_window(out, start=start, end=end)
    return out


def _try_semantic_search(
    events: list[Event], query: str, limit: int
) -> list[tuple[Event, float]] | None:
    """Run semantic search; print a loud, specific error and return ``None`` on failure.

    The REPL uses this so a misconfigured cache or missing API key doesn't
    kill the interactive session. The one-shot CLI wraps this with
    :func:`_semantic_search_or_exit` to turn a ``None`` result into a
    non-zero exit.

    Args:
        events: Candidate events.
        query: User query string.
        limit: Maximum number of matches to return.

    Returns:
        ``(event, similarity)`` pairs ranked best-first, or ``None`` if
        the search could not run (no cache, no API key, etc.). When the
        cache is *partially* populated, a warning is printed but the
        search still runs against the cached subset.
    """
    from pycon_cal_scraper.semantic import (
        EmbeddingCache,
        VoyageEmbedder,
        coverage,
        semantic_search_with_scores,
    )

    cfg = load_config()
    cache_path = paths.embeddings_file()
    cache = EmbeddingCache.load(cache_path, model=cfg.embedding_model)
    if len(cache) == 0:
        console.print(
            "[red]No embeddings cached.[/red] Semantic search is unavailable until "
            "you run [bold]pycon-cal-scraper embed[/bold] "
            f"(expected at {cache_path})."
        )
        return None

    cached, total = coverage(cache, events)
    if cached < total:
        console.print(
            f"[yellow]⚠ Embedding cache covers {cached}/{total} candidate events.[/yellow] "
            f"Re-run [bold]pycon-cal-scraper embed[/bold] to include the rest."
        )
    if cached == 0:
        console.print(
            "[red]None of the candidate events have embeddings.[/red] Run "
            "[bold]pycon-cal-scraper embed[/bold] first."
        )
        return None

    try:
        embedder = VoyageEmbedder(model=cfg.embedding_model)
    except RuntimeError as exc:
        console.print(f"[red]Semantic search failed: {exc}[/red]")
        return None

    async def _run() -> list[float]:
        return await embedder.embed_query(query)

    try:
        query_vec = asyncio.run(_run())
    except Exception as exc:  # network / auth errors from Voyage
        console.print(f"[red]Semantic search failed embedding the query: {exc}[/red]")
        return None
    return semantic_search_with_scores(events, query_vec, cache, top_k=limit)


def _semantic_search_or_exit(
    events: list[Event], query: str, limit: int
) -> list[tuple[Event, float]]:
    """One-shot variant of :func:`_try_semantic_search` that exits on failure."""
    result = _try_semantic_search(events, query, limit)
    if result is None:
        raise typer.Exit(code=1)
    return result


def _apply_semantic_negatives(events: list[Event], phrases: Sequence[str]) -> list[Event]:
    """Filter ``events`` by semantic-negation phrases, falling back to substring.

    When the embedding cache is present, embeds each phrase via Voyage and
    drops events whose cosine similarity with any phrase exceeds
    :attr:`UserConfig.semantic_negative_threshold`. When the cache is
    missing or the API key isn't set, prints a warning and falls back to
    substring exclusion via :func:`apply_lexical_negatives`.
    """
    if not phrases:
        return events
    from pycon_cal_scraper.semantic import (
        EmbeddingCache,
        VoyageEmbedder,
        filter_by_negative_phrases,
    )

    cfg = load_config()
    cache = EmbeddingCache.load(paths.embeddings_file(), model=cfg.embedding_model)
    if len(cache) == 0:
        console.print(
            "[yellow]No embeddings cache; treating semantic negatives as substring "
            "excludes. Run [bold]pycon-cal-scraper embed[/bold] for true semantic "
            "negation.[/yellow]"
        )
        return apply_lexical_negatives(events, phrases)
    try:
        embedder = VoyageEmbedder(model=cfg.embedding_model, api_key_env=cfg.voyage_api_key_env)
    except RuntimeError as exc:
        console.print(
            f"[yellow]{exc} — falling back to substring excludes for semantic negatives.[/yellow]"
        )
        return apply_lexical_negatives(events, phrases)

    async def _embed_all() -> list[list[float]]:
        return [await embedder.embed_query(p) for p in phrases]

    try:
        vectors = asyncio.run(_embed_all())
    except Exception as exc:  # network / auth errors
        console.print(
            f"[yellow]Voyage call failed for negative phrases ({exc}); falling back "
            f"to substring excludes.[/yellow]"
        )
        return apply_lexical_negatives(events, phrases)
    return filter_by_negative_phrases(
        events, vectors, cache, threshold=cfg.semantic_negative_threshold
    )


app.command(name="search")(search_cmd)


@app.command()
def saved() -> None:
    """List events in the saved-list."""
    store = _saved_store()
    events = _events_store().load()
    resolved = store.resolve(events)
    if not resolved:
        console.print("[yellow]No saved events yet.[/yellow]")
        return
    # In the saved view itself, the overlap column flags internal conflicts.
    console.print(_render_events_table(resolved, saved_ids=store.ids(), saved_events=resolved))


@app.command()
def save(
    ids: Annotated[list[str], typer.Argument(help="Event IDs to mark as saved.")],
) -> None:
    """Add one or more events to the saved-list."""
    store = _saved_store()
    events_by_id = {e.id: e for e in _events_store().load()}
    for eid in ids:
        if eid not in events_by_id:
            console.print(f"[yellow]warning:[/yellow] unknown event id {eid!r}")
        added = store.add(eid)
        if added:
            console.print(f"[green]saved[/green] {eid}")
        else:
            console.print(f"[dim]already saved[/dim] {eid}")


@app.command()
def unsave(
    ids: Annotated[list[str], typer.Argument(help="Event IDs to remove from the saved-list.")],
) -> None:
    """Remove one or more events from the saved-list."""
    store = _saved_store()
    for eid in ids:
        if store.remove(eid):
            console.print(f"[green]removed[/green] {eid}")
        else:
            console.print(f"[dim]not in saved-list[/dim] {eid}")


# --- now ----------------------------------------------------------------------


def _render_now(events: list[Event], saved_ids: set[str], at: datetime) -> None:
    """Print the "happening now" + "next up" tables for ``at``."""
    saved_set = [e for e in events if e.id in saved_ids]
    now_events = events_happening_at(events, at)
    soon = upcoming_events(events, at, limit=5)
    pretty_at = at.strftime("%a %b %d %H:%M %Z")
    console.print(f"[bold]As of {pretty_at}[/bold]")
    if not now_events and not soon:
        console.print(
            "[yellow]No events covering this moment — the conference may not "
            "be running, or `sync` is out of date.[/yellow]"
        )
        return
    if now_events:
        console.print("[bold cyan]Happening now[/bold cyan]")
        console.print(_render_events_table(now_events, saved_ids=saved_ids, saved_events=saved_set))
    else:
        console.print("[dim]Nothing happening at this exact moment.[/dim]")
    if soon:
        console.print("[bold cyan]Up next[/bold cyan]")
        console.print(_render_events_table(soon, saved_ids=saved_ids, saved_events=saved_set))


@app.command()
def now() -> None:
    """Show events happening right now plus the next five up.

    Uses the local time on this machine, interpreted in the conference
    timezone. Useful day-of: ``pycon-cal-scraper now`` tells you whether to
    walk into the room you're in front of or look for the next thing.
    """
    events = _events_store().load()
    if not events:
        console.print("[yellow]No events cached yet. Run `pycon-cal-scraper sync`.[/yellow]")
        raise typer.Exit(code=1)
    saved_ids = _saved_store().ids()
    _render_now(events, saved_ids, datetime.now(tz=CONFERENCE_TZ))


# --- export -------------------------------------------------------------------


@app.command()
def export(
    output: Annotated[
        Path,
        typer.Argument(
            help="Where to write the .ics file (use '-' to stream to stdout).",
        ),
    ],
    all_events: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Export every scraped event instead of just the saved-list.",
        ),
    ] = False,
) -> None:
    """Export saved events (or every scraped event) as an iCalendar feed.

    The output is RFC 5545 VCALENDAR text suitable for Apple Calendar,
    Outlook, Fantastical, etc. Each VEVENT carries a stable
    ``pycon-<id>@pycon-cal-scraper`` UID, so re-importing updates rather
    than duplicates.
    """
    from pycon_cal_scraper.ical import events_to_ics

    events = _events_store().load()
    if not events:
        console.print("[yellow]No events cached yet. Run `pycon-cal-scraper sync`.[/yellow]")
        raise typer.Exit(code=1)
    if all_events:
        to_export = events
        label = "every scraped event"
    else:
        to_export = _saved_store().resolve(events)
        if not to_export:
            console.print(
                "[yellow]Saved-list is empty. Save some events first, or pass "
                "[bold]--all[/bold] to export the full schedule.[/yellow]"
            )
            raise typer.Exit(code=1)
        label = f"{len(to_export)} saved event(s)"
    cfg = load_config()
    payload = events_to_ics(to_export, venue_address=cfg.venue_address)
    if str(output) == "-":
        sys.stdout.write(payload)
        return
    target = Path(output).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")
    console.print(f"[green]Exported {label} to {target}[/green]")


# --- REPL ---------------------------------------------------------------------


def _warn_if_no_embedding_cache() -> bool:
    """Print a warning to ``console`` if no embedding cache exists.

    Returns:
        ``True`` when a cache is present, ``False`` when it's empty/missing.
    """
    from pycon_cal_scraper.semantic import EmbeddingCache

    cfg = load_config()
    cache_path = paths.embeddings_file()
    cache = EmbeddingCache.load(cache_path, model=cfg.embedding_model)
    if len(cache) == 0:
        console.print(
            "[yellow]⚠ No embeddings cached.[/yellow] Semantic search is unavailable "
            "until you run [bold]pycon-cal-scraper embed[/bold]."
        )
        return False
    return True


_REPL_HELP = (
    "[bold]Interactive search.[/bold] Type a query, or use /save <id>, /unsave <id>, "
    "/saved, /conflicts, /now, /day <code>, /from <when>, /to <when>, /limit <n>, "
    "/lexical, /keyword, /semantic, /room <name>, /quit. Ctrl-D exits.\n"
    "[dim]Query syntax: bare words = positive; !word = exclude; "
    '!"phrase" = semantic exclude.[/dim]\n'
    "[dim]Results/saved table: ↑/↓ move, enter toggles save, o opens URL, "
    "? toggles abstract, esc or q exits.[/dim]\n"
    "[dim]/conflicts: ↑/↓ row, ←/→ group, enter toggles save, esc or q exits.[/dim]"
)


def _picker_active() -> bool:
    """Return ``True`` iff the REPL should use the interactive arrow-key picker.

    Gated on stdin being a real terminal so that CliRunner-based tests (and
    other piped inputs) fall back to the static rich-table renderer.
    """
    return sys.stdin.isatty()


def _truncate(text: str, width: int) -> str:
    """Truncate ``text`` to ``width`` columns, padding with spaces if shorter."""
    if len(text) > width:
        return text[: max(0, width - 1)] + "…"
    return f"{text:<{width}}"


def _picker_header(*, has_scores: bool, has_matches: bool, title_w: int, speakers_w: int) -> str:
    """Build the static header line for the interactive picker."""
    parts = ["  ", "★", "⚠", f"{'ID':<8}"]
    if has_scores:
        parts.append(f"{'Score':>6}")
    if has_matches:
        parts.append(f"{'Where':<5}")
    parts.append(f"{'When':<14}")
    parts.append(f"{'Type':<10}")
    parts.append(_truncate("Title", title_w))
    parts.append(_truncate("Speakers", speakers_w))
    parts.append("Room")
    return "  ".join(parts)


def _picker_row(
    event: Event,
    *,
    is_cursor: bool,
    saved_ids: set[str],
    saved_events: list[Event],
    scores: dict[str, float] | None,
    matches: dict[str, frozenset[str]] | None,
    title_w: int,
    speakers_w: int,
) -> str:
    """Render one event as a fixed-width text row for the picker."""
    cursor_mark = "▶ " if is_cursor else "  "
    star = "★" if event.id in saved_ids else " "
    conflict = "⚠" if event.id not in saved_ids and event_overlaps_any(event, saved_events) else " "
    parts = [cursor_mark, star, conflict, f"{event.id:<8}"]
    if scores is not None:
        parts.append(f"{scores.get(event.id, 0.0):>6.3f}")
    if matches is not None:
        parts.append(f"{_format_match_fields(matches.get(event.id, frozenset())):<5}")
    parts.append(f"{_format_dt(event):<14}")
    parts.append(f"{event.type.value:<10}")
    parts.append(_truncate(event.title, title_w))
    parts.append(_truncate(", ".join(event.speakers), speakers_w))
    parts.append(event.room or "")
    return "  ".join(parts)


def _open_url(url: str) -> bool:
    """Open ``url`` in the user's default browser via :mod:`webbrowser`.

    Returns:
        ``True`` if the browser module reported success, ``False`` otherwise.
        Isolated as a helper so tests can monkeypatch the side effect.
    """
    import webbrowser

    try:
        return webbrowser.open(url)
    except Exception:  # pragma: no cover — webbrowser failures are platform-dependent
        return False


def _run_event_picker(
    rows: list[Event],
    *,
    saved_store: SavedStore,
    all_events: list[Event],
    scores: dict[str, float] | None = None,
    matches: dict[str, frozenset[str]] | None = None,
) -> None:
    """Open an arrow-key-navigable picker over ``rows``.

    Up/Down moves the cursor (wraps at the ends). Enter toggles the saved
    state for the highlighted event. ``?`` toggles an inline abstract panel
    for the cursor row; ``o`` opens that row's URL in the system browser.
    Esc or ``q`` returns control to the REPL prompt. The cursor row is
    rendered in reverse video; ★ and ⚠ markers update on every keypress
    to reflect freshly toggled saves.
    """
    from prompt_toolkit.application import Application
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style

    if not rows:
        return

    title_w = 40
    speakers_w = 25
    cursor = [0]
    show_abstract = [False]
    status = [""]

    def _abstract_block(event: Event) -> list[tuple[str, str]]:
        body = event.description or event.abstract or "(no abstract available)"
        return [
            ("class:abstract-title", f"\n{event.title}\n"),
            ("class:abstract", f"{body}\n"),
            ("class:abstract", f"{event.url}\n"),
        ]

    def get_content() -> FormattedText:
        saved_ids = saved_store.ids()
        saved_events = saved_store.resolve(all_events)
        lines: list[tuple[str, str]] = [
            (
                "class:header",
                _picker_header(
                    has_scores=scores is not None,
                    has_matches=matches is not None,
                    title_w=title_w,
                    speakers_w=speakers_w,
                )
                + "\n",
            )
        ]
        for i, event in enumerate(rows):
            text = _picker_row(
                event,
                is_cursor=(i == cursor[0]),
                saved_ids=saved_ids,
                saved_events=saved_events,
                scores=scores,
                matches=matches,
                title_w=title_w,
                speakers_w=speakers_w,
            )
            style = "class:cursor" if i == cursor[0] else ""
            lines.append((style, text + "\n"))
        if show_abstract[0]:
            lines.extend(_abstract_block(rows[cursor[0]]))
        if status[0]:
            lines.append(("class:status", f"{status[0]}\n"))
        lines.append(
            (
                "class:hint",
                "[↑/↓ move • enter toggle save • ? abstract • o open URL • esc/q exit]\n",
            )
        )
        return FormattedText(lines)

    kb = KeyBindings()

    @kb.add("up")
    def _up(event: Any) -> None:
        cursor[0] = (cursor[0] - 1) % len(rows)
        status[0] = ""

    @kb.add("down")
    def _down(event: Any) -> None:
        cursor[0] = (cursor[0] + 1) % len(rows)
        status[0] = ""

    @kb.add("enter")
    def _toggle(event: Any) -> None:
        target = rows[cursor[0]]
        if target.id in saved_store.ids():
            saved_store.remove(target.id)
        else:
            saved_store.add(target.id)

    @kb.add("?")
    def _toggle_abstract(event: Any) -> None:
        show_abstract[0] = not show_abstract[0]

    @kb.add("o")
    def _open(event: Any) -> None:
        target = rows[cursor[0]]
        opened = _open_url(str(target.url))
        status[0] = f"opened {target.url}" if opened else f"could not open {target.url}"

    @kb.add("escape")
    @kb.add("q")
    @kb.add("c-c")
    @kb.add("c-d")
    def _exit(event: Any) -> None:
        event.app.exit()

    style = Style.from_dict(
        {
            "cursor": "reverse",
            "header": "bold",
            "hint": "italic ansibrightblack",
            "abstract-title": "bold",
            "abstract": "",
            "status": "italic ansiyellow",
        }
    )
    layout = Layout(HSplit([Window(content=FormattedTextControl(get_content))]))
    app: Application[None] = Application(
        layout=layout,
        key_bindings=kb,
        full_screen=False,
        style=style,
        mouse_support=False,
    )
    app.run()


def _run_conflicts_picker(
    groups: list[list[Event]],
    *,
    saved_store: SavedStore,
    all_events: list[Event],
) -> None:
    """Open an arrow-key picker showing one conflict group at a time.

    A banner-style header announces the current position
    (``═══ Conflict 2 of 3 — 4 events ═══``) so users always see how many
    groups exist. Up/Down moves the row cursor inside the current group.
    With more than one group, Left/Right cycle to the previous/next group
    (wrapping at the ends) and reset the row cursor; with a single group,
    the ``←/→`` hint is suppressed so the UI doesn't promise navigation
    that wouldn't visibly do anything. Enter toggles the saved state of
    the highlighted event, ``?`` shows the abstract, ``o`` opens the URL,
    Esc/``q`` exits.

    Group composition is captured at the moment ``/conflicts`` is invoked:
    toggling saves updates ★ markers live but doesn't restructure the
    clusters — re-run ``/conflicts`` to see the new layout.
    """
    from prompt_toolkit.application import Application
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style

    if not groups:
        return

    title_w = 40
    speakers_w = 25
    group_idx = [0]
    row_idx = [0]
    show_abstract = [False]
    status = [""]
    multi_group = len(groups) > 1

    def _abstract_block(event: Event) -> list[tuple[str, str]]:
        body = event.description or event.abstract or "(no abstract available)"
        return [
            ("class:abstract-title", f"\n{event.title}\n"),
            ("class:abstract", f"{body}\n"),
            ("class:abstract", f"{event.url}\n"),
        ]

    def _banner() -> str:
        if multi_group:
            label = (
                f"Conflict {group_idx[0] + 1} of {len(groups)} — {len(groups[group_idx[0]])} events"
            )
        else:
            label = f"Only conflict — {len(groups[0])} events"
        rule = "═" * 3
        return f"{rule} {label} {rule}\n"

    def _hint() -> str:
        nav = "↑/↓ row" + (" • ←/→ group" if multi_group else "")
        return f"[{nav} • enter toggle save • ? abstract • o open URL • esc/q exit]\n"

    def get_content() -> FormattedText:
        group = groups[group_idx[0]]
        saved_ids = saved_store.ids()
        saved_events = saved_store.resolve(all_events)
        lines: list[tuple[str, str]] = [
            ("class:title", _banner()),
            (
                "class:header",
                _picker_header(
                    has_scores=False,
                    has_matches=False,
                    title_w=title_w,
                    speakers_w=speakers_w,
                )
                + "\n",
            ),
        ]
        for i, event in enumerate(group):
            text = _picker_row(
                event,
                is_cursor=(i == row_idx[0]),
                saved_ids=saved_ids,
                saved_events=saved_events,
                scores=None,
                matches=None,
                title_w=title_w,
                speakers_w=speakers_w,
            )
            style = "class:cursor" if i == row_idx[0] else ""
            lines.append((style, text + "\n"))
        if show_abstract[0]:
            lines.extend(_abstract_block(group[row_idx[0]]))
        if status[0]:
            lines.append(("class:status", f"{status[0]}\n"))
        lines.append(("class:hint", _hint()))
        return FormattedText(lines)

    kb = KeyBindings()

    @kb.add("up")
    def _up(event: Any) -> None:
        row_idx[0] = (row_idx[0] - 1) % len(groups[group_idx[0]])
        status[0] = ""

    @kb.add("down")
    def _down(event: Any) -> None:
        row_idx[0] = (row_idx[0] + 1) % len(groups[group_idx[0]])
        status[0] = ""

    @kb.add("left")
    def _prev_group(event: Any) -> None:
        if not multi_group:
            status[0] = "only one conflict group"
            return
        group_idx[0] = (group_idx[0] - 1) % len(groups)
        row_idx[0] = 0
        show_abstract[0] = False
        status[0] = ""

    @kb.add("right")
    def _next_group(event: Any) -> None:
        if not multi_group:
            status[0] = "only one conflict group"
            return
        group_idx[0] = (group_idx[0] + 1) % len(groups)
        row_idx[0] = 0
        show_abstract[0] = False
        status[0] = ""

    @kb.add("enter")
    def _toggle(event: Any) -> None:
        target = groups[group_idx[0]][row_idx[0]]
        if target.id in saved_store.ids():
            saved_store.remove(target.id)
        else:
            saved_store.add(target.id)

    @kb.add("?")
    def _toggle_abstract(event: Any) -> None:
        show_abstract[0] = not show_abstract[0]

    @kb.add("o")
    def _open(event: Any) -> None:
        target = groups[group_idx[0]][row_idx[0]]
        opened = _open_url(str(target.url))
        status[0] = f"opened {target.url}" if opened else f"could not open {target.url}"

    @kb.add("escape")
    @kb.add("q")
    @kb.add("c-c")
    @kb.add("c-d")
    def _exit(event: Any) -> None:
        event.app.exit()

    style = Style.from_dict(
        {
            "cursor": "reverse",
            "header": "bold",
            "title": "bold ansired",
            "hint": "italic ansibrightblack",
            "abstract-title": "bold",
            "abstract": "",
            "status": "italic ansiyellow",
        }
    )
    layout = Layout(HSplit([Window(content=FormattedTextControl(get_content))]))
    app: Application[None] = Application(
        layout=layout,
        key_bindings=kb,
        full_screen=False,
        style=style,
        mouse_support=False,
    )
    app.run()


def _run_repl(
    events: list[Event],
    *,
    results_limit: int,
    mode: str = "lexical",
    room: str | None = None,
) -> None:
    """Run the interactive search REPL until the user quits.

    Args:
        events: The cached event list to search against.
        results_limit: Maximum number of matches to render per query. The
            user can override this for the rest of the session with
            ``/limit <n>``.
        mode: Initial search mode — one of ``"lexical"`` (fuzzy
            Levenshtein), ``"keyword"`` (exact-token hit count), or
            ``"semantic"`` (Voyage embeddings). Switch in-session with
            ``/lexical``, ``/keyword``, ``/semantic``.
    """
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory

    saved_store = _saved_store()
    session: PromptSession[str] = PromptSession(history=InMemoryHistory())
    day: str | None = None
    from_ts: str | None = None
    to_ts: str | None = None
    console.print(_REPL_HELP)

    def _print_filter_state() -> None:
        console.print(
            f"[dim]filters: mode={mode}, room={room or 'any'}, day={day or 'any'}, "
            f"from={from_ts or '—'}, to={to_ts or '—'}, limit={results_limit}[/dim]"
        )

    _print_filter_state()
    # Always surface embedding-cache state at REPL start so the user
    # knows whether /semantic will work before they try it.
    has_cache = _warn_if_no_embedding_cache()
    if mode == "semantic" and not has_cache:
        console.print("[yellow]Falling back to lexical search until embeddings exist.[/yellow]")
        mode = "lexical"
    while True:
        try:
            line = session.prompt("pycon> ").strip()
        except (EOFError, KeyboardInterrupt):  # fmt: skip
            console.print()
            return
        if not line:
            continue
        if line in {"/quit", ":q", "exit"}:
            return
        if line == "/saved":
            saved_resolved = saved_store.resolve(events)
            if _picker_active() and saved_resolved:
                _run_event_picker(saved_resolved, saved_store=saved_store, all_events=events)
            else:
                console.print(
                    _render_events_table(
                        saved_resolved,
                        saved_ids=saved_store.ids(),
                        saved_events=saved_resolved,
                    )
                )
            continue
        if line == "/conflicts":
            saved_resolved = saved_store.resolve(events)
            groups = find_conflict_groups(saved_resolved)
            if not groups:
                console.print("[green]No conflicts in your saved list.[/green]")
                continue
            if _picker_active():
                _run_conflicts_picker(groups, saved_store=saved_store, all_events=events)
            else:
                for n, group in enumerate(groups, start=1):
                    console.print(f"[bold red]Conflict {n} ({len(group)} events):[/bold red]")
                    console.print(
                        _render_events_table(
                            group,
                            saved_ids=saved_store.ids(),
                            saved_events=saved_resolved,
                        )
                    )
            continue
        if line == "/limit" or line.startswith("/limit "):
            parts = line.split(maxsplit=1)
            try:
                new_limit = int(parts[1])
                if new_limit < 1:
                    raise ValueError
            except (ValueError, IndexError):  # fmt: skip
                console.print("[yellow]usage: /limit <positive integer>[/yellow]")
                continue
            results_limit = new_limit
            console.print(f"[dim]showing up to {results_limit} matches per query[/dim]")
            continue
        if line in {"/lexical", "/keyword", "/semantic"}:
            new_mode = line[1:]
            if new_mode == "semantic" and not _warn_if_no_embedding_cache():
                # Don't flip into a mode that's guaranteed to fail.
                continue
            mode = new_mode
            console.print(f"[dim]search mode: {mode}[/dim]")
            continue
        if line == "/room" or line.startswith("/room "):
            parts = line.split(maxsplit=1)
            room = parts[1].strip() if len(parts) == 2 else None
            console.print(f"[dim]room filter: {room or 'any'}[/dim]")
            continue
        if line == "/day" or line.startswith("/day "):
            parts = line.split(maxsplit=1)
            new_day = parts[1].strip() if len(parts) == 2 else None
            if new_day:
                try:
                    parse_day_token(new_day)
                except ValueError as exc:
                    console.print(f"[yellow]usage: /day <mon|tue|...|YYYY-MM-DD> ({exc})[/yellow]")
                    continue
            day = new_day
            console.print(f"[dim]day filter: {day or 'any'}[/dim]")
            continue
        if line == "/from" or line.startswith("/from "):
            parts = line.split(maxsplit=1)
            new_from = parts[1].strip() if len(parts) == 2 else None
            if new_from:
                try:
                    parse_when(new_from)
                except ValueError as exc:
                    console.print(f"[yellow]usage: /from <YYYY-MM-DDTHH:MM> ({exc})[/yellow]")
                    continue
            from_ts = new_from
            console.print(f"[dim]from: {from_ts or '—'}[/dim]")
            continue
        if line == "/to" or line.startswith("/to "):
            parts = line.split(maxsplit=1)
            new_to = parts[1].strip() if len(parts) == 2 else None
            if new_to:
                try:
                    parse_when(new_to)
                except ValueError as exc:
                    console.print(f"[yellow]usage: /to <YYYY-MM-DDTHH:MM> ({exc})[/yellow]")
                    continue
            to_ts = new_to
            console.print(f"[dim]to: {to_ts or '—'}[/dim]")
            continue
        if line == "/now":
            _render_now(events, saved_store.ids(), datetime.now(tz=CONFERENCE_TZ))
            continue
        if line == "/filters":
            _print_filter_state()
            continue
        if line.startswith("/save "):
            for eid in line.split()[1:]:
                if saved_store.add(eid):
                    console.print(f"[green]saved[/green] {eid}")
            continue
        if line.startswith("/unsave "):
            for eid in line.split()[1:]:
                if saved_store.remove(eid):
                    console.print(f"[green]removed[/green] {eid}")
            continue
        parsed = parse_query(line)
        try:
            candidates = _apply_time_filters(events, day=day, from_=from_ts, to=to_ts)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            continue
        if room:
            candidates = filter_events_by_room(candidates, room)
        matches, score_map, match_map = _execute_search(
            candidates, parsed, mode=mode, limit=results_limit
        )
        if matches is None:
            # _execute_search / _try_semantic_search already printed a loud red message.
            continue
        if not matches:
            console.print("[yellow]no matches[/yellow]")
            continue
        if _picker_active():
            _run_event_picker(
                matches,
                saved_store=saved_store,
                all_events=events,
                scores=score_map,
                matches=match_map,
            )
        else:
            console.print(
                _render_events_table(
                    matches,
                    saved_ids=saved_store.ids(),
                    saved_events=saved_store.resolve(events),
                    scores=score_map,
                    matches=match_map,
                )
            )


# --- gcal ---------------------------------------------------------------------


@gcal_app.command("login")
def gcal_login() -> None:
    """Run the Google OAuth desktop flow and cache the resulting token."""
    from pycon_cal_scraper.gcal import auth

    cfg = load_config()
    if not cfg.client_secret_path:
        console.print(
            "[red]Set client_secret_path first:[/red]\n"
            "  pycon-cal-scraper config set client_secret_path /path/to/client_secret.json"
        )
        raise typer.Exit(code=1)
    secret = Path(cfg.client_secret_path).expanduser()
    if not secret.exists():
        console.print(f"[red]client_secret.json not found at {secret}[/red]")
        raise typer.Exit(code=1)
    auth.login(secret)
    console.print(f"[green]OAuth token cached at {paths.token_file()}[/green]")


@gcal_app.command("sync")
def gcal_sync(
    calendar: Annotated[
        str | None, typer.Option("--calendar", help="Override the configured calendar id.")
    ] = None,
    prune: Annotated[
        bool,
        typer.Option("--prune", help="Delete GCal events whose pycon_id is no longer saved."),
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the plan without changing anything.")
    ] = False,
    retries: Annotated[
        int,
        typer.Option(
            "--retries",
            help="Retry passes for sub-requests that fail inside a batch (default 3).",
            min=0,
        ),
    ] = 3,
) -> None:
    """Push the saved-list to the configured Google Calendar."""
    from pycon_cal_scraper.gcal import auth as gauth
    from pycon_cal_scraper.gcal import sync as gsync

    cfg = load_config()
    calendar_id = calendar or cfg.calendar_id
    saved = _saved_store().resolve(_events_store().load())
    if not saved:
        console.print("[yellow]No saved events; nothing to sync.[/yellow]")
        return

    creds = gauth.load_cached_credentials()
    if creds is None or not creds.valid:
        console.print("[red]No valid token; run `pycon-cal-scraper gcal login` first.[/red]")
        raise typer.Exit(code=1)

    service = gauth.build_calendar_service(creds)

    # Only inspect the slice of the calendar that could possibly contain our
    # events. This both narrows the API surface and prevents the previous
    # filter bug (privateExtendedProperty=*) from re-emerging.
    window = gsync.conference_window(saved)
    time_min, time_max = window if window is not None else (None, None)

    async def _run() -> None:
        console.print(f"[cyan]Reading calendar events from {calendar_id}...[/cyan]")
        with _progress() as progress:
            list_task = progress.add_task("[cyan]Reading calendar[/cyan]", total=None)

            def on_page(page: int, scanned: int, found: int) -> None:
                progress.update(
                    list_task,
                    description=(
                        f"[cyan]Reading calendar[/cyan] (page {page}, "
                        f"{scanned} scanned, {found} managed)"
                    ),
                )

            existing = await gsync.list_managed_events(
                service,
                calendar_id=calendar_id,
                time_min=time_min,
                time_max=time_max,
                on_page=on_page,
            )
            progress.update(list_task, total=1, completed=1)
        console.print(f"[dim]Read {len(existing)} managed event(s) from calendar.[/dim]")
        plan = gsync.diff(saved, existing, prune=prune, venue_address=cfg.venue_address)
        console.print(f"[bold]Plan:[/bold] {plan.summary()}")
        if dry_run or plan.is_empty():
            return
        with _progress() as progress:
            apply_task: TaskID | None = None

            def _ensure_task(total: int) -> TaskID:
                nonlocal apply_task
                if apply_task is None:
                    apply_task = progress.add_task("[cyan]Syncing events[/cyan]", total=total)
                return apply_task

            def on_apply(action: str, completed: int, total: int) -> None:
                progress.update(_ensure_task(total), completed=completed, total=total)

            def on_batch(batch_idx: int, batches_in_round: int, retry_pass: int) -> None:
                if retry_pass == 0:
                    desc = f"[cyan]Syncing events[/cyan] (batch {batch_idx + 1}/{batches_in_round})"
                else:
                    desc = (
                        f"[cyan]Syncing events[/cyan] "
                        f"(retry {retry_pass}/{retries} • "
                        f"batch {batch_idx + 1}/{batches_in_round})"
                    )
                progress.update(_ensure_task(plan.total_actions()), description=desc)

            try:
                await gsync.apply(
                    service,
                    plan,
                    calendar_id=calendar_id,
                    on_progress=on_apply,
                    on_batch=on_batch,
                    venue_address=cfg.venue_address,
                    retries=retries,
                )
            except ExceptionGroup as eg:
                console.print(
                    f"[red]Sync finished with {len(eg.exceptions)} failure(s) "
                    f"after {retries} retries:[/red]"
                )
                for exc in eg.exceptions:
                    console.print(f"  [red]•[/red] {exc!r}")
                raise typer.Exit(code=1) from eg
        console.print("[green]Sync complete.[/green]")

    asyncio.run(_run())


@gcal_app.command("clean")
def gcal_clean(
    calendar: Annotated[
        str | None, typer.Option("--calendar", help="Override the configured calendar id.")
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")] = False,
) -> None:
    """Delete every event on the calendar that this tool created.

    Lists the candidate events first, asks for confirmation, and only then
    deletes. Use this when you want a clean slate before re-syncing.
    """
    from pycon_cal_scraper.gcal import auth as gauth
    from pycon_cal_scraper.gcal import sync as gsync

    cfg = load_config()
    calendar_id = calendar or cfg.calendar_id

    creds = gauth.load_cached_credentials()
    if creds is None or not creds.valid:
        console.print("[red]No valid token; run `pycon-cal-scraper gcal login` first.[/red]")
        raise typer.Exit(code=1)

    service = gauth.build_calendar_service(creds)

    console.print(
        f"[cyan]Scanning calendar {calendar_id} for events created by pycon-cal-scraper...[/cyan]"
    )
    console.print(
        "[dim]This walks every event on the calendar (no time bound), so it can take "
        "a while on large calendars.[/dim]"
    )

    async def _list_all() -> list[dict[str, Any]]:
        with _progress() as progress:
            list_task = progress.add_task("[cyan]Scanning calendar[/cyan]", total=None)

            def on_page(page: int, scanned: int, found: int) -> None:
                progress.update(
                    list_task,
                    description=(
                        f"[cyan]Scanning calendar[/cyan] (page {page}, "
                        f"{scanned} events scanned, {found} managed found)"
                    ),
                )

            return await gsync.list_managed_events(
                service, calendar_id=calendar_id, on_page=on_page
            )

    managed = asyncio.run(_list_all())
    console.print(f"[dim]Scan complete: {len(managed)} managed event(s) found.[/dim]")
    if not managed:
        console.print("[yellow]No managed events on this calendar.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("pycon_id", width=12)
    table.add_column("gcal id", width=24)
    table.add_column("When", width=20)
    table.add_column("Summary")
    for item in managed:
        pid = gsync.extract_pycon_id(item) or "?"
        gid = str(item.get("id", "?"))
        start_info = item.get("start") or {}
        start = start_info.get("dateTime") or start_info.get("date") or ""
        table.add_row(pid, gid, str(start), str(item.get("summary", "")))
    console.print(table)
    console.print(f"[bold]About to delete {len(managed)} event(s) from {calendar_id}.[/bold]")

    if not yes and not typer.confirm("Proceed?", default=False):
        console.print("[yellow]Aborted.[/yellow]")
        return

    plan = gsync.SyncPlan(to_delete=[str(item["id"]) for item in managed])

    async def _run_delete() -> None:
        with _progress() as progress:
            apply_task: TaskID | None = None

            def on_apply(action: str, completed: int, total: int) -> None:
                nonlocal apply_task
                if apply_task is None:
                    apply_task = progress.add_task("[cyan]Deleting events[/cyan]", total=total)
                progress.update(apply_task, completed=completed, total=total)

            try:
                await gsync.apply(service, plan, calendar_id=calendar_id, on_progress=on_apply)
            except ExceptionGroup as eg:
                console.print(f"[red]Clean finished with {len(eg.exceptions)} failure(s):[/red]")
                for exc in eg.exceptions:
                    console.print(f"  [red]•[/red] {exc!r}")
                raise typer.Exit(code=1) from eg

    asyncio.run(_run_delete())
    console.print(f"[green]Removed {len(managed)} event(s).[/green]")


# --- config -------------------------------------------------------------------


@config_app.command("show")
def config_show() -> None:
    """Print the current user configuration."""
    cfg = load_config()
    for k, v in cfg.model_dump().items():
        console.print(f"  {k} = {v}")


@config_app.command("keys")
def config_keys() -> None:
    """List every supported config key with its current value."""
    cfg = load_config()
    table = Table(show_header=True, header_style="bold")
    table.add_column("Key")
    table.add_column("Current value")
    for k, v in cfg.model_dump().items():
        table.add_row(k, str(v))
    console.print(table)


@config_app.command("set")
def config_set(
    key: Annotated[
        str,
        typer.Argument(
            help="Config key to update. Run `pycon-cal-scraper config keys` for the full list."
        ),
    ],
    value: Annotated[str, typer.Argument(help="New value.")],
) -> None:
    """Set a single configuration key."""
    cfg = load_config()
    data = cfg.model_dump()
    if key not in data:
        console.print(f"[red]Unknown config key {key!r}. Valid: {sorted(data)}[/red]")
        raise typer.Exit(code=1)
    data[key] = value
    try:
        updated = UserConfig.model_validate(data)
    except ValueError as exc:
        console.print(f"[red]Invalid value for {key!r}: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    save_config(updated)
    console.print(f"[green]config[/green] {key} = {value}")


def main() -> None:
    """Console-script entry point for ``pycon-cal-scraper``."""
    app()


if __name__ == "__main__":
    sys.exit(app())
