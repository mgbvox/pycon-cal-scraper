# pycon-cal-scraper

Scrape the [PyCon US 2026 schedule](https://us.pycon.org/2026/schedule/), search it
interactively, build a personal "saved" list, and sync that list to a Google
Calendar.

## Install

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
```

For day-to-day use, install the CLI globally:

```bash
uv tool install . --reinstall   # rerun after pulling changes
```

## Quick start

```bash
# 1. Pull all events from the PyCon site and cache them locally
pycon-cal-scraper sync

# 2. Search the cached schedule (one-shot)
pycon-cal-scraper search "static typing"

# 3. Or open the interactive REPL
pycon-cal-scraper search
#  pycon> async
#  pycon> /save 15
#  pycon> /saved
#  pycon> /limit 50
#  pycon> /room "Grand Ballroom"
#  pycon> /keyword            # switch search mode
#  pycon> /quit

# 4. Add or remove events from your saved-list directly
pycon-cal-scraper save 15 102
pycon-cal-scraper unsave 102
pycon-cal-scraper saved
```

## Search modes

Three orthogonal modes — pick whichever matches your intent:

| Mode | CLI flag | REPL command | Behavior |
| --- | --- | --- | --- |
| **Lexical** (default) | (none) | `/lexical` | Levenshtein-fuzzy match across title/speakers/abstract. Tolerates typos. |
| **Keyword** | `--keyword` / `-k` | `/keyword` | Exact-token hit count. No fuzz. Adds a `Where` column showing T/S/A. |
| **Semantic** | `--semantic` / `-s` | `/semantic` | Voyage AI cosine similarity. Adds a `Score` column. Requires `embed`. |

```bash
pycon-cal-scraper search --keyword rust
pycon-cal-scraper search --semantic "talks about scaling LLM serving"
```

### Filters

Stack any combination of these with any mode:

```bash
# Only one day
pycon-cal-scraper search --day fri "tutorial"

# Time window
pycon-cal-scraper search --from 2026-05-15T13:00 --to 2026-05-15T17:00

# Specific room (substring match)
pycon-cal-scraper search --room "Grand Ballroom" "keynote"

# Override the per-query result cap
pycon-cal-scraper search --limit 5 async
```

In the REPL, `/room <substring>` sets the room filter for the rest of the
session; `/room` (no arg) clears it.

### Negative search

Drop unwanted matches inline using `!` prefixes:

```bash
# Drop events whose title/speakers/abstract contains the word "python"
pycon-cal-scraper search "patterns" "!python"

# Quoted phrases use semantic exclusion (cosine similarity above the
# configured threshold) when an embedding cache exists. Falls back to
# substring exclusion when no cache is available.
pycon-cal-scraper search "talks about reliability" '!"machine learning"'
```

### Conflict markers

Every search/saved table renders a `⚠` column flagging events whose time
overlaps with one already on your saved list — quick visual conflict check
before saving more.

## Semantic search

Anthropic doesn't ship a first-party embedding endpoint; their docs recommend
[Voyage AI](https://docs.voyageai.com/) for Claude-based projects. This tool
uses Voyage embeddings to power optional semantic search and semantic
negation.

```bash
export VOYAGE_API_KEY=your-voyage-key

# One-time: embed every cached event. ~150 events fits in one API call.
pycon-cal-scraper embed

# Now you can search semantically:
pycon-cal-scraper search --semantic "talks about scaling LLM serving"

# Or toggle inside the REPL:
pycon-cal-scraper search
#  pycon> /semantic
#  pycon> talks about scaling LLM serving
```

Embeddings live next to `events.json` as `embeddings.npz` (numpy archive
keyed by event id). The default model is `voyage-3-lite`; override with
`config set embedding_model voyage-3`. Changing the model invalidates the
existing cache on the next `embed` run. Use `embed --rebuild` to force a
re-embed of every event.

If `sync` adds new events after the last `embed` run, the next semantic
search prints a `⚠ Embedding cache covers N/M events` warning so you know
to re-embed.

## Configuration

```bash
pycon-cal-scraper config show         # current values
pycon-cal-scraper config keys         # every supported key + current value
pycon-cal-scraper config set <key> <value>
```

Every default in source can be overridden:

| Key | Default | Notes |
| --- | --- | --- |
| `calendar_id` | `primary` | Google Calendar to sync to. |
| `client_secret_path` | `null` | Path to the OAuth desktop client JSON. Required for `gcal`. |
| `default_tz` | `America/Los_Angeles` | Fallback timezone. |
| `search_results_limit` | `20` | Per-query match cap. |
| `embedding_model` | `voyage-3-lite` | Voyage model name. |
| `embedding_batch_size` | `64` | Documents per Voyage API call. |
| `voyage_api_key_env` | `VOYAGE_API_KEY` | Env-var name to read for the API key. |
| `scraper_base_url` | `https://us.pycon.org` | Override only for offline mirrors. |
| `http_user_agent` | `pycon-cal-scraper/0.1 (...)` | Sent on every scrape request. |
| `http_cache_ttl_hours` | `24.0` | How long a cached page stays fresh. |
| `http_min_interval_seconds` | `0.25` | Minimum gap between live HTTP requests. |
| `http_concurrency` | `5` | Maximum in-flight live HTTP requests. |
| `search_weight_title` | `4` | Lexical/keyword weight for title hits. |
| `search_weight_speaker` | `2` | Lexical/keyword weight for speaker hits. |
| `search_weight_abstract` | `1` | Lexical/keyword weight for abstract hits. |
| `semantic_negative_threshold` | `0.5` | Cosine similarity at/above which `!"phrase"` excludes an event. |

## Google Calendar sync

1. Create an OAuth 2.0 **Desktop** client in
   [Google Cloud Console](https://console.cloud.google.com/apis/credentials) and
   download the `client_secret.json`.
2. Point the tool at it:

   ```bash
   pycon-cal-scraper config set client_secret_path ~/Downloads/client_secret.json
   ```

3. (Optional) Pick a target calendar — defaults to `primary`:

   ```bash
   pycon-cal-scraper config set calendar_id <calendar-id>@group.calendar.google.com
   ```

4. Authorize the app (opens your browser; token is cached afterwards):

   ```bash
   pycon-cal-scraper gcal login
   ```

5. Push your saved-list to the calendar:

   ```bash
   pycon-cal-scraper gcal sync           # add + update only
   pycon-cal-scraper gcal sync --dry-run # preview, no writes
   pycon-cal-scraper gcal sync --prune   # also remove events you've unsaved
   ```

   Sync only touches events that carry a `pycon_id` in their private extended
   properties — your other calendar entries are never modified. After
   `pycon-cal-scraper sync`, if any of your *saved* events changed (title,
   time, room, etc.), you'll see a hint suggesting you re-run `gcal sync`.

6. Wipe the slate clean (deletes every event this tool created on the
   calendar, after confirmation):

   ```bash
   pycon-cal-scraper gcal clean
   ```

   `gcal clean` walks the entire calendar (no time bound) and streams
   per-page progress so you can see what it's doing on large calendars.

## How it works

- **Scrape** — `sync` fetches the `talks`, `tutorials`, and `sponsor-presentations`
  schedule pages, parses each calendar grid into events, then visits each
  `/presentation/<id>/` detail page for the speaker, audience level, and abstract.
  All HTTP traffic is async, throttled, concurrency-limited, and cached on disk.
- **Search** — Three modes (lexical / keyword / semantic) with stackable filters
  (`--day`, `--from`/`--to`, `--room`) and inline negatives (`!word`, `!"phrase"`).
- **Save** — A JSON file (`saved.json`) holds the ordered list of saved event IDs.
- **Sync** — `gcal sync` diffs your saved-list against the calendar (matched by
  `pycon_id` in extended properties) and applies inserts / patches / deletes.

## Where things live

Paths are platform-aware via `platformdirs` and overridable through the
`PYCON_CAL_DATA_DIR`, `PYCON_CAL_CACHE_DIR`, and `PYCON_CAL_CONFIG_DIR`
environment variables. On macOS the defaults sit under
`~/Library/Application Support/pycon-cal-scraper/` (`events.json`, `saved.json`,
`embeddings.npz`, `config.json`, `token.json`) and
`~/Library/Caches/pycon-cal-scraper/http/` for HTTP responses. Linux uses
`~/.local/share`, `~/.cache`, and `~/.config`.

## Development

```bash
uv run pytest                    # tests + coverage (gate: 80%)
uv run ruff check src tests
uv run ruff format src tests
uv run ty check src
```

## Security & privacy

- The OAuth `client_secret.json`, the cached `token.json`, and the
  `VOYAGE_API_KEY` environment variable never leave your machine.
- `creds/`, `*.env`, `client_secret*.json`, `credentials*.json`, and
  `token*.json` are gitignored — keep them outside the repo or under
  `creds/`.
- `gcal sync` only touches events that carry the tool's `pycon_id`
  extended property — it cannot read or modify your other calendar items.
