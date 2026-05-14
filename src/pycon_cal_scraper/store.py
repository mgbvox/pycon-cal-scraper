"""JSON-backed persistence of scraped events and the saved-list.

Two small classes:

* :class:`EventsStore` — round-trips the full event list.
* :class:`SavedStore` — an ordered, idempotent set of saved event ids.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from pydantic import TypeAdapter

from pycon_cal_scraper.models import Event, SavedEvent

_EVENTS_ADAPTER: TypeAdapter[list[Event]] = TypeAdapter(list[Event])
_SAVED_ADAPTER: TypeAdapter[list[SavedEvent]] = TypeAdapter(list[SavedEvent])


def _atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` via a ``.tmp`` + rename, so a crash can't corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


class EventsStore:
    """Reads and writes the scraped events list as a single JSON file.

    Attributes:
        path: The on-disk JSON file. Missing parent directories are created
            on save.
    """

    def __init__(self, path: Path) -> None:
        """Build a store rooted at ``path``."""
        self.path = Path(path)

    def load(self) -> list[Event]:
        """Return the persisted events, or an empty list if the file is missing."""
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return _EVENTS_ADAPTER.validate_python(raw)

    def save(self, events: Iterable[Event]) -> None:
        """Persist ``events`` to disk, replacing any existing file atomically."""
        payload = _EVENTS_ADAPTER.dump_python(list(events), mode="json")
        _atomic_write_text(self.path, json.dumps(payload, indent=2, sort_keys=False))


class SavedStore:
    """An ordered, idempotent set of saved event ids persisted as JSON.

    Insertion order is preserved on disk; lookups via :meth:`ids` use a set.

    Attributes:
        path: The on-disk JSON file.
    """

    def __init__(self, path: Path) -> None:
        """Build a store rooted at ``path`` and load any existing state."""
        self.path = Path(path)
        self._items: list[SavedEvent] = self._load_from_disk()

    def _load_from_disk(self) -> list[SavedEvent]:
        """Read ``self.path``; return ``[]`` if the file does not exist."""
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return _SAVED_ADAPTER.validate_python(raw)

    def _flush(self) -> None:
        """Atomically persist the in-memory state back to disk."""
        payload = _SAVED_ADAPTER.dump_python(self._items, mode="json")
        _atomic_write_text(self.path, json.dumps(payload, indent=2, sort_keys=False))

    def add(self, event_id: str) -> bool:
        """Save an event id.

        Args:
            event_id: The event id to add.

        Returns:
            ``True`` if newly inserted, ``False`` if the id was already saved.
        """
        if any(s.id == event_id for s in self._items):
            return False
        self._items.append(SavedEvent(id=event_id, saved_at=datetime.now(tz=UTC)))
        self._flush()
        return True

    def remove(self, event_id: str) -> bool:
        """Remove an event id from the saved-list.

        Args:
            event_id: The id to remove.

        Returns:
            ``True`` if the id was present and is now gone, ``False`` otherwise.
        """
        before = len(self._items)
        self._items = [s for s in self._items if s.id != event_id]
        if len(self._items) != before:
            self._flush()
            return True
        return False

    def clear(self) -> None:
        """Remove every entry from the saved-list."""
        if self._items:
            self._items = []
            self._flush()

    def ids(self) -> set[str]:
        """Return the saved event ids as a set."""
        return {s.id for s in self._items}

    def __iter__(self) -> Iterator[SavedEvent]:
        return iter(list(self._items))

    def __len__(self) -> int:
        return len(self._items)

    def resolve(self, events: Iterable[Event]) -> list[Event]:
        """Look up saved ids against ``events``.

        Args:
            events: The candidate events (typically from :meth:`EventsStore.load`).

        Returns:
            The matching :class:`Event` objects, in save-order. Ids that
            don't appear in ``events`` are silently skipped, which makes the
            method robust against schedule drift.
        """
        by_id = {e.id: e for e in events}
        return [by_id[s.id] for s in self._items if s.id in by_id]
