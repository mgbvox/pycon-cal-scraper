"""Ranked fuzzy search across scraped events.

Each query token is matched against the words in each searchable field
(title, speakers, abstract) using Levenshtein distance. A token contributes
to a field's score only if its closest field-word is within a length-aware
threshold; the contribution is proportional to ``1 - distance/len(token)``
and multiplied by the field's weight.

Field weights make the ranking obvious: a title hit always outranks a
speaker hit, which always outranks an abstract hit. Query semantics are
AND across tokens — every token must find some match somewhere.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pycon_cal_scraper.models import Event

# Relative weights. Title > Speaker > Abstract.
_W_TITLE = 4
_W_SPEAKER = 2
_W_ABSTRACT = 1

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass(frozen=True)
class SearchWeights:
    """Per-field weight bundle for :func:`search`.

    Attributes:
        title: Multiplier applied to a token's title-field score.
        speaker: Multiplier applied to a token's speaker-field score.
        abstract: Multiplier applied to a token's abstract-field score.
    """

    title: int = _W_TITLE
    speaker: int = _W_SPEAKER
    abstract: int = _W_ABSTRACT


DEFAULT_WEIGHTS = SearchWeights()


@dataclass(frozen=True)
class ParsedQuery:
    """A search query split into positive text and negative excludes.

    Attributes:
        positive: The free-text portion of the query (negatives stripped).
        lexical_negatives: Bare ``!word`` exclusions — drop events with that
            word as an exact lowercased token in title, speakers, or abstract.
        semantic_negatives: Quoted ``!"phrase"`` exclusions — drop events
            whose embedding is too close to the phrase. The CLI handles the
            embedding lookup; this struct only records the phrases.
    """

    positive: str
    lexical_negatives: tuple[str, ...] = ()
    semantic_negatives: tuple[str, ...] = ()


# Match either: optional ! followed by a "..." string, or optional ! followed by
# any run of non-whitespace.
_QUERY_TOKEN_RE = re.compile(r'!?"(?:[^"\\]|\\.)*"|!?\S+')


def parse_query(text: str) -> ParsedQuery:
    """Parse a query string into positive text plus lexical/semantic negatives.

    Args:
        text: Raw query, e.g. ``'rust !python !"machine learning"'``.

    Returns:
        A :class:`ParsedQuery`.

    Examples:
        >>> q = parse_query('rust !python !"machine learning"')
        >>> q.positive
        'rust'
        >>> q.lexical_negatives
        ('python',)
        >>> q.semantic_negatives
        ('machine learning',)
    """
    positive_parts: list[str] = []
    lex_neg: list[str] = []
    sem_neg: list[str] = []
    for match in _QUERY_TOKEN_RE.finditer(text):
        token = match.group(0)
        if token.startswith('!"') and token.endswith('"') and len(token) > 3:
            sem_neg.append(token[2:-1])
        elif token.startswith("!") and len(token) > 1:
            inner = token[1:]
            # `!"x"` written as a single shell token also works — fall through to semantic.
            if inner.startswith('"') and inner.endswith('"') and len(inner) >= 2:
                sem_neg.append(inner[1:-1])
            else:
                lex_neg.append(inner)
        elif token.startswith('"') and token.endswith('"') and len(token) >= 2:
            positive_parts.append(token[1:-1])
        else:
            positive_parts.append(token)
    return ParsedQuery(
        positive=" ".join(positive_parts),
        lexical_negatives=tuple(lex_neg),
        semantic_negatives=tuple(sem_neg),
    )


def _event_token_set(event: Event) -> set[str]:
    """Return every lowercased word from title/speakers/abstract as a set."""
    text = " ".join(
        [
            event.title,
            " ".join(event.speakers),
            event.abstract or "",
            event.description or "",
        ]
    )
    return set(_tokenize(text))


def event_contains_word(event: Event, word: str) -> bool:
    """Return ``True`` iff ``word`` (case-insensitive) appears as a whole token."""
    return word.lower() in _event_token_set(event)


def event_contains_substring(event: Event, needle: str) -> bool:
    """Return ``True`` iff ``needle`` appears anywhere in the searched text fields."""
    text = " ".join(
        [
            event.title,
            " ".join(event.speakers),
            event.abstract or "",
            event.description or "",
        ]
    ).lower()
    return needle.lower() in text


def apply_lexical_negatives(events: Iterable[Event], negatives: Sequence[str]) -> list[Event]:
    """Drop events containing any of the negative tokens in any searched field.

    Single-word negatives use exact-token matching; multi-word negatives use
    substring matching (so ``"machine learning"`` matches that phrase
    verbatim). Returns ``events`` unchanged when ``negatives`` is empty.
    """
    if not negatives:
        return list(events)
    out: list[Event] = []
    for event in events:
        skip = False
        for neg in negatives:
            if " " in neg.strip():
                if event_contains_substring(event, neg):
                    skip = True
                    break
            elif event_contains_word(event, neg):
                skip = True
                break
        if not skip:
            out.append(event)
    return out


def _tokenize(text: str) -> list[str]:
    """Split ``text`` into lowercase word tokens."""
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def _levenshtein(a: str, b: str) -> int:
    """Compute the Levenshtein edit distance between two strings.

    Standard two-row dynamic-programming implementation. Runs in
    ``O(len(a) * len(b))`` time and ``O(min(len(a), len(b)))`` memory.

    Args:
        a: First string.
        b: Second string.

    Returns:
        The minimum number of single-character insertions, deletions, or
        substitutions needed to transform ``a`` into ``b``.
    """
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (ca != cb)
            current[j] = min(insert_cost, delete_cost, replace_cost)
        previous = current
    return previous[-1]


def _token_field_score(token: str, field_words: list[str]) -> float:
    """Score how well ``token`` fuzzy-matches the words in a single field.

    Args:
        token: A lowercased query token.
        field_words: Lowercased words from the field (title / speakers /
            abstract).

    Returns:
        A score in ``[0.0, 1.0]``. ``1.0`` for an exact hit, lower for fuzzy
        hits, and ``0.0`` when no field word is within
        ``max(1, len(token) // 3)`` edits of ``token``.
    """
    if not field_words or not token:
        return 0.0
    threshold = max(1, len(token) // 3)
    best = min(_levenshtein(token, w) for w in field_words)
    if best > threshold:
        return 0.0
    return 1.0 - best / len(token)


def _score(event: Event, tokens: Sequence[str], weights: SearchWeights) -> float:
    """Return the total relevance score for ``event`` against the query tokens.

    Empty query matches everything with a score of ``1.0`` (so the original
    event order is preserved). Otherwise, every token must find a match in
    at least one field (title / speakers / abstract); the per-token scores
    are summed across fields and across tokens, weighted by ``weights``. A
    token that fails to match anywhere returns ``0.0``.

    Args:
        event: The candidate event.
        tokens: The query, already tokenized and lowercased.
        weights: Per-field weight bundle.

    Returns:
        A non-negative float — ``0.0`` means "no match".
    """
    if not tokens:
        return 1.0
    title_words = _tokenize(event.title)
    speaker_words = _tokenize(" ".join(event.speakers))
    abstract_words = _tokenize(event.abstract or event.description or "")

    total = 0.0
    for tok in tokens:
        t_score = _token_field_score(tok, title_words) * weights.title
        s_score = _token_field_score(tok, speaker_words) * weights.speaker
        a_score = _token_field_score(tok, abstract_words) * weights.abstract
        per_token = t_score + s_score + a_score
        if per_token == 0:
            return 0.0  # AND semantics: every token must match somewhere
        total += per_token
    return total


def keyword_search(
    events: Iterable[Event],
    query: str,
    *,
    weights: SearchWeights = DEFAULT_WEIGHTS,
) -> list[tuple[Event, int, frozenset[str]]]:
    """Rank events by total number of exact-token hits across fields.

    Unlike :func:`search`, this mode does no fuzzy matching: each token of
    the query must appear *exactly* in the event's title, speakers, or
    abstract to score. Each occurrence in a field contributes
    ``weights.<field>`` points. Multi-token queries are scored OR-style —
    an event only needs one matching token to appear in the results.

    Args:
        events: The candidate events.
        query: Free-text query, tokenized on word boundaries.
        weights: Per-field weight bundle.

    Returns:
        ``(event, total_hits_weighted, matched_fields)`` triples sorted
        by score, best first. ``matched_fields`` is a frozenset drawn
        from ``{"title", "speakers", "abstract"}`` — the fields whose
        word-list contained at least one query token. Stable on ties
        (original order preserved). Returns ``[]`` if no token matches
        anywhere.
    """
    tokens = _tokenize(query)
    if not tokens:
        return []
    scored: list[tuple[int, int, Event, frozenset[str]]] = []
    for idx, event in enumerate(events):
        title_words = _tokenize(event.title)
        speaker_words = _tokenize(" ".join(event.speakers))
        abstract_words = _tokenize(event.abstract or event.description or "")
        score = 0
        matched: set[str] = set()
        for tok in tokens:
            t_hits = title_words.count(tok)
            s_hits = speaker_words.count(tok)
            a_hits = abstract_words.count(tok)
            if t_hits:
                matched.add("title")
            if s_hits:
                matched.add("speakers")
            if a_hits:
                matched.add("abstract")
            score += t_hits * weights.title
            score += s_hits * weights.speaker
            score += a_hits * weights.abstract
        if score > 0:
            scored.append((-score, idx, event, frozenset(matched)))
    scored.sort()
    return [(e, -neg, fields) for neg, _, e, fields in scored]


def search(
    events: Iterable[Event],
    query: str,
    *,
    weights: SearchWeights = DEFAULT_WEIGHTS,
) -> list[Event]:
    """Search ``events`` with ``query`` and return matches by relevance.

    Args:
        events: The candidate events (typically the full scraped set).
        query: A free-text query. Tokenized on word boundaries; matching is
            case-insensitive and tolerates small typos (Levenshtein
            distance up to ``len(token) // 3``).
        weights: Per-field weight bundle. Defaults to :data:`DEFAULT_WEIGHTS`.

    Returns:
        The matching events, best matches first. Stable on score ties
        (original order preserved). Returns ``[]`` if no event matches all
        query tokens.
    """
    tokens = _tokenize(query)
    scored: list[tuple[float, int, Event]] = []
    for idx, event in enumerate(events):
        score = _score(event, tokens, weights)
        if score > 0:
            # Negate score so ascending sort puts the best matches first.
            # idx breaks ties by original order so the sort is stable across runs.
            scored.append((-score, idx, event))
    scored.sort()
    return [e for _, _, e in scored]
