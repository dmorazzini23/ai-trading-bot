"""Marketable limit price for an urgent netting exit.

An urgent exit is a protective exit (stop loss / position backstop) or a full
close, decided once by the netting cycle. It goes out as a DAY limit priced to
fill: a sell at the bid less a small buffer, a buy-to-cover at the ask plus a
small buffer, rounded to the tick away from the touch so it stays marketable.
Without a usable quote it is priced from the last bar close, moved by a wider
buffer in the same direction. It is never a market order. Every other order
keeps its bar-close limit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, InvalidOperation

from ai_trading.market.symbol_specs import get_tick_size

# Past the touch by 10 bps: absorbs the quote moving a tick or two between the
# read and broker arrival (about 1 cent on a $10 name, 10 cents on $100) while
# costing far less than the 3% collar or a stop that keeps slipping. The broker
# fills a marketable limit at the touch or better, so the buffer is a ceiling
# on slippage, not a cost paid on every fill.
URGENT_EXIT_QUOTE_BUFFER_BPS = 10.0
# Without a usable quote the last bar close can be a minute or more old, so
# the buffer is wider (50 bps, still well inside the 3% collar) to stay
# marketable through a normal one-minute move.
URGENT_EXIT_FALLBACK_BUFFER_BPS = 50.0
# A protective exit follows a real fast move: its touch may sit up to 10% from
# the last bar close (a stop typically fires 5-8% from entry, and a gap can add
# a few percent). Further than that the quote is treated as bad data and the
# exit falls back to the bar close. A normal full close is held instead when
# its touch leaves the regular price collar (pretrade PRICE_COLLAR_BLOCK).
PROTECTIVE_EXIT_MAX_TOUCH_DEVIATION_PCT = 0.10
# Used when the symbol spec has no usable tick.
_DEFAULT_TICK = Decimal("0.01")

PRICE_BASIS_QUOTE = "quote"
PRICE_BASIS_BAR_CLOSE = "bar_close_fallback"

_BUY_SIDES = frozenset({"buy", "cover", "buy_to_cover"})
_SELL_SIDES = frozenset({"sell", "exit", "sell_long"})


@dataclass(frozen=True, slots=True)
class UrgentExitPrice:
    limit_price: float
    basis: str
    reference_price: float
    buffer_bps: float
    tick: float
    quote_unusable_reason: str | None = None


def _finite_positive(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0.0:
        return None
    return parsed


def exit_quote_unusable_reason(
    *,
    bid: float | None,
    ask: float | None,
    quote_age_ms: float | None,
    max_quote_age_ms: int,
) -> str | None:
    """``None`` when the quote can price an urgent exit, else why it cannot.

    Staleness uses the same threshold and the same unknown-age rule as
    ``STALE_QUOTE_BLOCK`` (an unknown age is not stale).
    """

    bid_value = _finite_positive(bid)
    ask_value = _finite_positive(ask)
    if bid_value is None or ask_value is None:
        return "missing"
    if ask_value < bid_value:
        return "crossed"
    if max_quote_age_ms > 0 and quote_age_ms is not None and float(quote_age_ms) > float(max_quote_age_ms):
        return "stale"
    return None


def _tick_for(symbol: str) -> Decimal:
    try:
        tick = get_tick_size(symbol)
    except (KeyError, ValueError, TypeError, AttributeError):
        return _DEFAULT_TICK
    return tick if isinstance(tick, Decimal) and tick > 0 else _DEFAULT_TICK


def _marketable_limit(reference: float, buffer_bps: float, tick: Decimal, *, buy: bool) -> Decimal:
    """``reference`` moved by ``buffer_bps`` past the touch, rounded away from it
    to the tick (up for a buy, down for a sell). Decimal throughout, so a price
    that lands on a tick is never pushed a tick further by float error."""

    try:
        factor = Decimal(1) + (Decimal(1) if buy else Decimal(-1)) * Decimal(repr(buffer_bps)) / Decimal(10_000)
        raw = Decimal(repr(float(reference))) * factor
        steps = (raw / tick).to_integral_value(rounding=ROUND_CEILING if buy else ROUND_FLOOR)
    except InvalidOperation:
        raise ValueError(f"cannot price {reference!r} to tick {tick}") from None
    rounded = steps * tick
    # A sell rounded down to zero (sub-tick price) goes out at one tick.
    return rounded if rounded > 0 else tick


def urgent_exit_limit_price(
    *,
    symbol: str,
    side: str,
    bid: float | None,
    ask: float | None,
    quote_age_ms: float | None,
    max_quote_age_ms: int,
    fallback_price: float,
    max_touch_deviation_pct: float | None = None,
) -> UrgentExitPrice | None:
    """Marketable limit for an urgent exit, or ``None`` if it cannot be priced.

    ``None`` (unknown side, or no quote and no positive bar close) leaves the
    order at its bar-close limit, exactly as before. With
    ``max_touch_deviation_pct`` (protective exits) a touch further than that
    from ``fallback_price`` makes the quote unusable (``far_from_bar_close``).
    """

    side_token = str(side or "").strip().lower()
    if side_token in _BUY_SIDES:
        buy = True
    elif side_token in _SELL_SIDES:
        buy = False
    else:
        return None
    unusable = exit_quote_unusable_reason(
        bid=bid, ask=ask, quote_age_ms=quote_age_ms, max_quote_age_ms=max_quote_age_ms
    )
    close_for_bound = _finite_positive(fallback_price)
    if unusable is None and max_touch_deviation_pct is not None and close_for_bound is not None:
        touch = float(ask) if buy else float(bid)  # type: ignore[arg-type]
        if abs(touch - close_for_bound) / close_for_bound > float(max_touch_deviation_pct):
            unusable = "far_from_bar_close"
    if unusable is None:
        reference = float(ask) if buy else float(bid)  # type: ignore[arg-type]
        buffer_bps = URGENT_EXIT_QUOTE_BUFFER_BPS
        basis = PRICE_BASIS_QUOTE
    else:
        close_value = _finite_positive(fallback_price)
        if close_value is None:
            return None
        reference = close_value
        buffer_bps = URGENT_EXIT_FALLBACK_BUFFER_BPS
        basis = PRICE_BASIS_BAR_CLOSE
    tick = _tick_for(symbol)
    limit = _marketable_limit(reference, buffer_bps, tick, buy=buy)
    return UrgentExitPrice(
        limit_price=float(limit),
        basis=basis,
        reference_price=float(reference),
        buffer_bps=float(buffer_bps),
        tick=float(tick),
        quote_unusable_reason=unusable,
    )


__all__ = [
    "PRICE_BASIS_BAR_CLOSE",
    "PRICE_BASIS_QUOTE",
    "PROTECTIVE_EXIT_MAX_TOUCH_DEVIATION_PCT",
    "URGENT_EXIT_FALLBACK_BUFFER_BPS",
    "URGENT_EXIT_QUOTE_BUFFER_BPS",
    "UrgentExitPrice",
    "exit_quote_unusable_reason",
    "urgent_exit_limit_price",
]
