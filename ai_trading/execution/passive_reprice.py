"""Pure contracts for bounded paper-sampling passive repricing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
import math
from typing import Any


@dataclass(frozen=True, slots=True)
class PassiveQuoteDecision:
    """Validated non-marketable replacement quote."""

    allowed: bool
    reason: str
    limit_price: float | None
    quote_age_ms: float | None
    spread_bps: float | None


def deterministic_passive_reprice_id(
    root_client_order_id: str,
    *,
    generation: int,
) -> str:
    """Return a stable Alpaca-compatible child id for one replacement generation."""

    root = "".join(
        character
        for character in str(root_client_order_id or "").strip()
        if character.isalnum() or character in {"-", "_", "."}
    )
    if not root:
        raise ValueError("root_client_order_id is required")
    parsed_generation = int(generation)
    if parsed_generation < 1:
        raise ValueError("generation must be positive")
    suffix = f"-pr{parsed_generation}"
    return f"{root[: max(1, 48 - len(suffix))]}{suffix}"[:48]


def _finite_positive(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0.0:
        return None
    return parsed


def _directional_tick_price(
    value: float,
    *,
    tick_size: float,
    side: str,
) -> float | None:
    try:
        price_decimal = Decimal(str(value))
        tick_decimal = Decimal(str(tick_size))
        if tick_decimal <= 0:
            return None
        rounding = ROUND_FLOOR if side == "buy" else ROUND_CEILING
        ticks = (price_decimal / tick_decimal).to_integral_value(rounding=rounding)
        snapped = ticks * tick_decimal
        parsed = float(snapped)
    except (InvalidOperation, TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0.0 else None


def validate_passive_reprice_quote(
    *,
    side: str,
    bid: Any,
    ask: Any,
    quote_ts: datetime | None,
    now: datetime | None = None,
    tick_size: float = 0.01,
    max_quote_age_ms: float,
    max_spread_bps: float,
) -> PassiveQuoteDecision:
    """Validate a fresh unlocked NBBO and return a directionally snapped passive price."""

    side_key = str(side or "").strip().lower()
    if side_key not in {"buy", "sell"}:
        return PassiveQuoteDecision(False, "side_not_supported", None, None, None)
    bid_value = _finite_positive(bid)
    ask_value = _finite_positive(ask)
    if bid_value is None or ask_value is None:
        return PassiveQuoteDecision(False, "quote_missing", None, None, None)
    if ask_value <= bid_value:
        return PassiveQuoteDecision(False, "quote_locked_or_crossed", None, None, None)

    mid = (bid_value + ask_value) / 2.0
    spread_bps = ((ask_value - bid_value) / mid) * 10_000.0
    if not math.isfinite(spread_bps) or spread_bps > max(0.0, float(max_spread_bps)):
        return PassiveQuoteDecision(False, "spread_above_max", None, None, spread_bps)

    quote_age_ms: float | None = None
    if quote_ts is None:
        return PassiveQuoteDecision(False, "quote_timestamp_missing", None, None, spread_bps)
    timestamp = quote_ts if quote_ts.tzinfo is not None else quote_ts.replace(tzinfo=UTC)
    current = now or datetime.now(UTC)
    current = current if current.tzinfo is not None else current.replace(tzinfo=UTC)
    quote_age_ms = max(
        (current.astimezone(UTC) - timestamp.astimezone(UTC)).total_seconds() * 1000.0,
        0.0,
    )
    if quote_age_ms > max(0.0, float(max_quote_age_ms)):
        return PassiveQuoteDecision(
            False,
            "quote_age_above_max",
            None,
            quote_age_ms,
            spread_bps,
        )

    passive_basis = bid_value if side_key == "buy" else ask_value
    snapped = _directional_tick_price(
        passive_basis,
        tick_size=float(tick_size),
        side=side_key,
    )
    if snapped is None:
        return PassiveQuoteDecision(False, "tick_quantization_failed", None, quote_age_ms, spread_bps)
    if side_key == "buy" and (snapped > bid_value or snapped >= ask_value):
        return PassiveQuoteDecision(False, "passive_price_would_cross", None, quote_age_ms, spread_bps)
    if side_key == "sell" and (snapped < ask_value or snapped <= bid_value):
        return PassiveQuoteDecision(False, "passive_price_would_cross", None, quote_age_ms, spread_bps)
    return PassiveQuoteDecision(True, "ok", snapped, quote_age_ms, spread_bps)


__all__ = [
    "PassiveQuoteDecision",
    "deterministic_passive_reprice_id",
    "validate_passive_reprice_quote",
]
