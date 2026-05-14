"""Tests for the ranked search function."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from pycon_cal_scraper.models import Event, EventType
from pycon_cal_scraper.search import (
    SearchWeights,
    apply_lexical_negatives,
    keyword_search,
    parse_query,
    search,
)

PACIFIC = ZoneInfo("America/Los_Angeles")


def _event(
    eid: str, title: str, speakers: list[str] | None = None, abstract: str | None = None
) -> Event:
    return Event.model_validate(
        {
            "id": eid,
            "url": f"https://us.pycon.org/2026/schedule/presentation/{eid}/",
            "title": title,
            "type": EventType.talk,
            "speakers": speakers or [],
            "start": datetime(2026, 5, 15, 9, 0, tzinfo=PACIFIC),
            "end": datetime(2026, 5, 15, 9, 30, tzinfo=PACIFIC),
            "abstract": abstract,
        }
    )


def test_empty_query_returns_all_events() -> None:
    events = [_event("1", "A"), _event("2", "B")]
    assert {e.id for e in search(events, "")} == {"1", "2"}


def test_title_match_outranks_abstract_match() -> None:
    events = [
        _event("1", "Async patterns", abstract="..."),
        _event("2", "Memory management", abstract="A nice intro to async."),
    ]
    results = search(events, "async")
    assert [e.id for e in results] == ["1", "2"]


def test_speaker_match_outranks_abstract_match() -> None:
    events = [
        _event("1", "Memory management", speakers=["Async Person"]),
        _event("2", "Logging tricks", abstract="Discusses async tactics."),
    ]
    results = search(events, "async")
    assert [e.id for e in results] == ["1", "2"]


def test_case_insensitive_match() -> None:
    events = [_event("1", "ASYNC patterns"), _event("2", "boring")]
    assert {e.id for e in search(events, "AsYnC")} == {"1"}


def test_multi_token_query_requires_all_tokens() -> None:
    events = [
        _event("1", "Async memory tricks"),
        _event("2", "Memory only"),
        _event("3", "Async only"),
    ]
    assert {e.id for e in search(events, "async memory")} == {"1"}


def test_no_match_returns_empty() -> None:
    events = [_event("1", "Async")]
    assert search(events, "rust") == []


def test_typo_in_query_still_matches() -> None:
    """A single-edit typo within threshold should still surface the event."""
    events = [_event("1", "Async patterns"), _event("2", "Logging tricks")]
    # 'asnyc' is 'async' with two characters transposed (distance 2 -> threshold 1
    # for a 5-char token), but 'asyn' (distance 1) is within threshold.
    results = search(events, "asyn")
    assert results and results[0].id == "1"


def test_fuzzy_exact_outranks_fuzzy_near() -> None:
    """Exact matches must outrank fuzzy ones, even when both are valid."""
    events = [
        _event("1", "Async patterns"),  # exact match for 'async'
        _event("2", "Asynk weirdness"),  # fuzzy match (distance 1 for 'async' vs 'asynk')
    ]
    results = search(events, "async")
    assert [e.id for e in results] == ["1", "2"]


def test_levenshtein_threshold_rejects_far_matches() -> None:
    """A token shouldn't match a wildly different field word."""
    events = [_event("1", "Memory management")]
    # 'asynk' shares no prefix with any title word; threshold is 1, distance is >> 1.
    assert search(events, "asynk") == []


# ---------------------------------------------------------------------------
# keyword_search
# ---------------------------------------------------------------------------


def test_keyword_search_returns_empty_for_no_hits() -> None:
    events = [_event("1", "Memory management")]
    assert keyword_search(events, "rust") == []


def test_keyword_search_returns_empty_for_no_tokens() -> None:
    events = [_event("1", "Anything")]
    assert keyword_search(events, "   ") == []


def test_keyword_search_ranks_by_total_hit_count() -> None:
    """An event mentioning the token more times should outrank a single-mention one."""
    events = [
        _event("1", "Async best practices", abstract="async async async"),  # 4 title+abstract hits
        _event("2", "Async intro", abstract="brief"),  # 1 title hit
        _event("3", "Memory", abstract="Mentions async once."),  # 1 abstract hit
    ]
    results = keyword_search(events, "async")
    assert [e.id for e, _, _ in results] == ["1", "2", "3"]
    assert results[0][1] > results[1][1] > results[2][1]


def test_keyword_search_skips_fuzzy_matches() -> None:
    """Unlike :func:`search`, keyword_search does NOT fuzzy-match."""
    events = [_event("1", "Asynk patterns")]
    assert keyword_search(events, "async") == []


def test_keyword_search_uses_field_weights() -> None:
    """Title hits should outscore abstract hits per the weights bundle."""
    events = [
        _event("1", "Rust", abstract=""),  # 1 title hit
        _event("2", "Other", abstract="rust rust"),  # 2 abstract hits
    ]
    weights = SearchWeights(title=10, speaker=1, abstract=1)
    results = keyword_search(events, "rust", weights=weights)
    # Title hit (10) outranks two abstract hits (2).
    assert [e.id for e, _, _ in results] == ["1", "2"]


# ---------------------------------------------------------------------------
# parse_query / negative search
# ---------------------------------------------------------------------------


def test_parse_query_splits_positive_and_negatives() -> None:
    parsed = parse_query('rust !python !"machine learning"')
    assert parsed.positive == "rust"
    assert parsed.lexical_negatives == ("python",)
    assert parsed.semantic_negatives == ("machine learning",)


def test_parse_query_handles_only_positive() -> None:
    parsed = parse_query("async patterns")
    assert parsed.positive == "async patterns"
    assert parsed.lexical_negatives == ()
    assert parsed.semantic_negatives == ()


def test_parse_query_strips_quotes_around_positive_phrase() -> None:
    parsed = parse_query('"machine learning" !rust')
    assert parsed.positive == "machine learning"
    assert parsed.lexical_negatives == ("rust",)


def test_parse_query_handles_only_negatives() -> None:
    parsed = parse_query("!python !javascript")
    assert parsed.positive == ""
    assert parsed.lexical_negatives == ("python", "javascript")


def test_apply_lexical_negatives_drops_matching_events() -> None:
    events = [
        _event("1", "Async patterns"),
        _event("2", "Memory management", abstract="discusses async."),
        _event("3", "Rust patterns"),
    ]
    survivors = apply_lexical_negatives(events, ["async"])
    assert [e.id for e in survivors] == ["3"]


def test_apply_lexical_negatives_substring_matches_phrases() -> None:
    """Multi-word negatives use case-insensitive substring matching."""
    events = [
        _event("1", "Intro to machine learning"),
        _event("2", "Async patterns"),
    ]
    survivors = apply_lexical_negatives(events, ["machine learning"])
    assert [e.id for e in survivors] == ["2"]


def test_apply_lexical_negatives_empty_returns_input() -> None:
    events = [_event("1", "x")]
    assert apply_lexical_negatives(events, []) == events


def test_keyword_search_reports_matched_fields() -> None:
    """The third element of each result must flag which fields contained the token."""
    events = [
        _event("1", "Rust patterns"),
        _event("2", "Other", speakers=["Rust Person"]),
        _event("3", "Other", abstract="written in rust"),
        _event("4", "Rust", abstract="rust again"),
    ]
    by_id = {e.id: fields for e, _, fields in keyword_search(events, "rust")}
    assert by_id["1"] == frozenset({"title"})
    assert by_id["2"] == frozenset({"speakers"})
    assert by_id["3"] == frozenset({"abstract"})
    assert by_id["4"] == frozenset({"title", "abstract"})
