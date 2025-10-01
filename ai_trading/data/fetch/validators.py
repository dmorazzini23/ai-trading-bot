"""Lightweight parameter validators for data fetchers."""
from __future__ import annotations

VALID_FEEDS = {"iex", "sip", "yahoo", "finnhub"}
VALID_ADJUSTMENTS = {"raw", "split", "all"}


def validate_feed(feed: str | None) -> None:
    if feed is None:
        return
    if feed not in VALID_FEEDS:
        raise ValueError(f"invalid feed: {feed}")


def validate_adjustment(adj: str | None) -> None:
    if adj is None:
        return
    if adj not in VALID_ADJUSTMENTS:
        raise ValueError(f"invalid adjustment: {adj}")
