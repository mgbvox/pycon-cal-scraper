# TODO

Backlog drawn from the 2026-05-14 audit. Items marked **[skipped from P0/P1]**
were deferred during the P0+P1 sweep because a concurrent session was actively
editing the surrounding files (`gcal/sync.py`). Pick those up first when the
gcal session lands.

The audit grouping is preserved so it's easy to see what tier each item came
from. Cross out items as they ship; new items can be appended at the bottom
of each section.

---

## Carry-over from the P0/P1 sweep (touched `gcal/sync.py` — left to avoid step-on)

### P0-5 — Stop overwriting user calendar customizations
- **Why:** `_has_drifted` in `gcal/sync.py` currently considers any difference
  "drift" and patches the event back to `build_event_payload`'s output. Users
  who add reminders, attach files, change color, or invite collaborators on a
  pycon event lose those edits on every sync.
- **Shape:** before `events_resource.update(...)`, fetch the existing event and
  merge user-mutable fields (`reminders`, `colorId`, `attendees`,
  `attachments`, `transparency`, `visibility`) from the existing item into the
  payload we send. Restrict the drift check to fields we own (summary,
  start, end, location, description).
- **Test:** patch path simulates an existing event with `reminders.overrides`
  set; after sync the override should still be present in the body submitted.

### P1-11 partial — Retries on the gcal side
- HTTP retries with backoff already landed in `http_cache.py` (P1-11 scraper
  half). The gcal side still needs:
  - Map `googleapiclient.errors.HttpError` with 429/5xx to the same retry
    policy (the recent batch work already retries `BatchHttpRequest` sub-
    requests — confirm the retry helper covers single-event `events.insert`
    / `update` / `delete` too).
  - Surface a `--retries 0` opt-out for users debugging API errors.

---

## P2 — Architecture / ergonomics (medium)

### Split `cli.py` into a `cli/` package
- Currently >1,400 LOC mixing Typer wiring, rendering, picker UI, REPL loop.
- Suggested layout:
  - `cli/app.py` — Typer apps + entry point
  - `cli/render.py` — rich tables + picker primitives
  - `cli/repl.py` — `_run_repl` and dispatch
  - `cli/commands/` — one file per top-level command
- Keep the module-level `app`, `gcal_app`, `config_app` re-exports so the
  console script entry point doesn't change.

### Strongly type the scraper's intermediate dicts
- `parse_list_page` returns `list[dict[str, object]]`; `Scraper.fetch_all`
  re-casts every field. PyCharm flags ~5 `str(object)` smells.
- Replace with a `ScrapedRow` `TypedDict` (or dataclass) — same for
  `parse_presentation_detail`'s `dict[str, str | list[str] | None]`.

### Drop `Event.abstract` (collapse to `description`)
- `scraper.py:423-424` writes both to the same string; every consumer
  spells out `event.abstract or event.description`.
- Migration path: keep `abstract` as a property aliasing `description` for
  one release, then remove.

### `SavedEvent` schema versioning
- Add `version: int = 1` to `SavedEvent` and a `_migrate` step in
  `SavedStore._load_from_disk`. Front-loads the work before notes / priority
  fields appear.

### Auto-conflict warning on `save`
- When `pycon-cal-scraper save 102` would create an overlap with an existing
  saved event, print a confirmation prompt by default; `--force` to skip.

### `reminder_minutes` config + per-event override
- New `UserConfig.reminder_minutes: int | None`; wired into
  `build_event_payload` as `reminders.overrides`.

### `gcal status` — read-only diff
- Same flow as `gcal sync --dry-run` but always exits 0 and gives a richer
  per-event status (saved/in-cal/up-to-date/drifted/missing).

### Sprints + posters scraping
- Fixtures already in `tests/fixtures/sprints_list.html` /
  `posters_list.html`. Add the URLs to `conference.SCHEDULE_PATHS` and a
  parser branch for the non-grid layout.

### Drop the unused `paths.main()` shim
- `paths.py:81-86` is a debug-print leftover.

### Live integration test behind a marker
- `pytest -m live` that hits us.pycon.org and asserts the parser produces a
  non-empty `Event` list with sensible day distribution. Skip in default CI.

### Refactor REPL local state into a `ReplState` dataclass
- `mode`, `room`, `day`, `from_ts`, `to_ts`, `results_limit` are currently
  loop-local. A small dataclass cleans up the dispatch table and prepares
  for command-handler extraction.

### Move `creds/` out of the repo path
- The directory is gitignored, but having `credentials.json` / `.env` in the
  working tree is a footgun. `config_dir()` already points at platformdirs;
  document that path as the secrets location.

### Structured logging behind `--debug`
- Use rich's `RichHandler` + `logging.getLogger("pycon_cal_scraper")`. Wire
  to the Typer root callback (`@app.callback`) so flag works for every cmd.

---

## P3 — Future-proofing / nice-to-haves

- Speaker / track favorites (`pycon-cal-scraper follow @Pat` ⇒ auto-tag
  every Pat talk in search).
- Notes per saved event (carry into the gcal description).
- "Ask Claude about this talk" using the Anthropic SDK with prompt caching.
- Headless / device-flow OAuth (`gcal login --no-browser`).
- Picker multi-select with `space`.
- ASCII-fallback markers when the terminal isn't UTF-8 capable.
- Speaker / room / track-only filters on the one-shot `search` CLI
  (`--track`, `--audience`).
- Persist REPL history to disk so up-arrow remembers across sessions.
- `pycon-cal-scraper status` summarizing cache freshness, embed coverage,
  and saved-list size in one glance.

---

## Audit-derived assumptions still worth revisiting

- **CSS-grid parser is brittle.** A redesign of us.pycon.org returns zero
  events silently. The "live" integration test above is the cheapest
  insurance.
- **Synthetic event IDs (`keynote:slug-from-title`) churn with title edits.**
  Consider hashing speaker + start datetime instead.
- **Single-machine state.** No cross-device sync of `saved.json`.
  Two-machine attendees diverge. Lowest-effort fix: optional `--state-file`
  pointed at a Dropbox/iCloud path.
- **Embedding cache silently runs on partial coverage.** The yellow warning
  is easy to miss in a busy REPL — consider auto-running `embed` after
  `sync` adds new events, or refusing semantic search until coverage hits
  100%.
