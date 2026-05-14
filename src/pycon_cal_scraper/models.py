"""Data model for events and the saved-list.

Defines:

* :class:`EventType` — an enum mirroring PyCon's slot-class taxonomy.
* :class:`Event` — a tz-aware, validated record for a single schedule entry.
* :class:`SavedEvent` — a pointer into the events store with a save timestamp.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


class EventType(StrEnum):
    talk = "talk"
    tutorial = "tutorial"
    sponsor = "sponsor"
    keynote = "keynote"
    plenary = "plenary"
    poster = "poster"
    sprint = "sprint"
    charla = "charla"
    break_ = "break"

    @classmethod
    def from_slot_class(cls, slot_class: str) -> EventType | None:
        """Map a CSS slot class (e.g. ``slot-talk``) to an :class:`EventType`.

        Args:
            slot_class: A class string from a ``<section class="slot slot-X">``
                element on a PyCon schedule page.

        Returns:
            The matching :class:`EventType`, or ``None`` if the class doesn't
            correspond to a known event kind.
        """
        if not slot_class.startswith("slot-"):
            return None
        token = slot_class[len("slot-") :]
        # PyCon uses several talk-flavored slot classes that all share talk semantics:
        # the difference is a visual "track" tag (Security, AI, Lightning Talks). We
        # surface those as `track` on the Event and keep the type as `talk`.
        talk_aliases = {"talk", "security", "ai", "lightning-talks"}
        if token in talk_aliases:
            return cls.talk
        if token == "sponsor-workshop":
            return cls.sponsor
        if token == "break":
            return cls.break_
        try:
            return cls(token)
        except ValueError:
            return None

    @classmethod
    def track_from_slot_class(cls, slot_class: str) -> str | None:
        """Extract the visual "track" tag from a slot class.

        Args:
            slot_class: A class string from a slot section.

        Returns:
            A human-readable track label (``"Security"``, ``"AI"``, or
            ``"Lightning Talks"``), or ``None`` if the class has no track.
        """
        token = slot_class[len("slot-") :] if slot_class.startswith("slot-") else slot_class
        if token == "security":
            return "Security"
        if token == "ai":
            return "AI"
        if token == "lightning-talks":
            return "Lightning Talks"
        return None


class Event(BaseModel):
    """A single PyCon schedule entry.

    Attributes:
        id: PyCon presentation id (digits) for backed events, or a synthetic
            slug like ``keynote:lin-qiao`` for plenaries/keynotes.
        url: Canonical URL — the ``/presentation/<id>/`` page when one exists,
            otherwise the list page that surfaced the event.
        title: Display title of the event.
        type: Coarse classification (talk, tutorial, keynote, …).
        speakers: Speaker names, in display order.
        start: Timezone-aware start of the slot.
        end: Timezone-aware end of the slot; must be strictly after ``start``.
        room: Physical room/venue label, when known.
        track: Visual track tag (Security / AI / Lightning Talks), when set.
        audience_level: Free-text experience level from the detail page.
        abstract: Short pitch from the detail page.
        description: Long description; identical to ``abstract`` for most events.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    url: HttpUrl
    title: str
    type: EventType
    speakers: list[str] = Field(default_factory=list)
    start: datetime
    end: datetime
    room: str | None = None
    track: str | None = None
    audience_level: str | None = None
    abstract: str | None = None
    description: str | None = None

    @field_validator("speakers", mode="before")
    @classmethod
    def _normalize_speakers(cls, value: object) -> object:
        if isinstance(value, str):
            return [s.strip() for s in value.split(",") if s.strip()]
        return value

    @field_validator("start", "end")
    @classmethod
    def _require_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("datetime must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _check_interval(self) -> Self:
        if self.end <= self.start:
            raise ValueError("end must be after start")
        return self

    @property
    def duration_minutes(self) -> int:
        """Return the slot length in whole minutes."""
        return int((self.end - self.start).total_seconds() // 60)


class SavedEvent(BaseModel):
    """A pointer into the events store for the saved-list.

    Attributes:
        id: Matches :attr:`Event.id`; used to resolve back to an :class:`Event`.
        saved_at: UTC timestamp captured when the user first saved the event.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    saved_at: datetime
