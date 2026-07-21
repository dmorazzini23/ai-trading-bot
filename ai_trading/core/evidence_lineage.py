"""Stable identifiers and timestamp helpers for opportunity evidence."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from typing import Any, Iterable


OPPORTUNITY_CORRELATION_VERSION = "opportunity-v1"


def normalize_evidence_timestamp(value: Any) -> datetime | None:
    """Return an aware UTC timestamp when *value* is parseable."""

    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def normalize_opportunity_side(value: Any) -> str:
    """Normalize an opportunity side without conflating it with an order ID."""

    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if token in {"buy", "long", "buy_to_open", "cover", "buy_to_cover"}:
        return "buy"
    if token in {
        "sell",
        "short",
        "sell_short",
        "sellshort",
        "sell_to_close",
        "sell_to_reduce",
    }:
        return "sell"
    return "hold"


def _stable_tokens(values: Iterable[Any] | None) -> list[str]:
    tokens = {
        str(value or "").strip().lower()
        for value in (values or ())
        if str(value or "").strip()
    }
    return sorted(tokens)


def deterministic_opportunity_correlation_id(
    *,
    symbol: Any,
    source_timestamp: Any,
    side: Any,
    strategy_id: Any = None,
    sleeves: Iterable[Any] | None = None,
    opportunity_key: Any = None,
) -> str:
    """Build an order-independent, deterministic opportunity correlation ID.

    The identifier is rooted in the market observation and decision context. It
    intentionally excludes client/broker order IDs, gate results, and processing
    timestamps so retries, controlled skips, and replacement orders share the
    same opportunity identity.
    """

    timestamp = normalize_evidence_timestamp(source_timestamp)
    if timestamp is None:
        raise ValueError("source_timestamp is required for opportunity correlation")
    payload = {
        "version": OPPORTUNITY_CORRELATION_VERSION,
        "symbol": str(symbol or "UNKNOWN").strip().upper() or "UNKNOWN",
        "source_timestamp": timestamp.isoformat(),
        "side": normalize_opportunity_side(side),
        "strategy_id": str(strategy_id or "").strip().lower(),
        "sleeves": _stable_tokens(sleeves),
        "opportunity_key": str(opportunity_key or "").strip(),
    }
    material = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"opp_{hashlib.sha256(material.encode('utf-8')).hexdigest()[:32]}"


__all__ = [
    "OPPORTUNITY_CORRELATION_VERSION",
    "deterministic_opportunity_correlation_id",
    "normalize_evidence_timestamp",
    "normalize_opportunity_side",
]
