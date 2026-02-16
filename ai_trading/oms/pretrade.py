"""OMS pre-trade controls for size, collars, duplicates, and throttles."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ai_trading.config.management import get_env


@dataclass(slots=True)
class OrderIntent:
    symbol: str
    side: str
    qty: int
    notional: float
    limit_price: float | None
    bar_ts: datetime
    client_order_id: str
    last_price: float | None = None
    mid: float | None = None
    spread: float | None = None
    sleeve: str | None = None


class SlidingWindowRateLimiter:
    """Rate limiter for order and cancel message budgets."""

    def __init__(
        self,
        *,
        global_orders_per_min: int = 0,
        per_symbol_orders_per_min: int = 0,
        cancels_per_min: int = 0,
        cancel_loop_max_without_fill: int = 0,
        cancel_loop_block_bars: int = 0,
    ) -> None:
        self.global_orders_per_min = max(0, int(global_orders_per_min))
        self.per_symbol_orders_per_min = max(0, int(per_symbol_orders_per_min))
        self.cancels_per_min = max(0, int(cancels_per_min))
        self.cancel_loop_max_without_fill = max(0, int(cancel_loop_max_without_fill))
        self.cancel_loop_block_bars = max(0, int(cancel_loop_block_bars))

        self._order_ts: deque[float] = deque()
        self._symbol_order_ts: dict[str, deque[float]] = {}
        self._cancel_ts: deque[float] = deque()
        self._cancel_without_fill: dict[str, int] = {}
        self._symbol_bar_index: dict[str, int] = {}
        self._symbol_last_bar_ts: dict[str, datetime] = {}
        self._symbol_block_until_bar: dict[str, int] = {}

    @staticmethod
    def _now() -> float:
        import time

        return time.monotonic()

    @staticmethod
    def _prune(window: deque[float], now: float, seconds: float) -> None:
        while window and now - window[0] > seconds:
            window.popleft()

    def _advance_bar(self, symbol: str, bar_ts: datetime) -> int:
        ts = bar_ts if bar_ts.tzinfo else bar_ts.replace(tzinfo=UTC)
        previous = self._symbol_last_bar_ts.get(symbol)
        if previous is None or ts > previous:
            self._symbol_last_bar_ts[symbol] = ts
            self._symbol_bar_index[symbol] = self._symbol_bar_index.get(symbol, 0) + 1
        return self._symbol_bar_index.get(symbol, 0)

    def allow_order(self, symbol: str, bar_ts: datetime) -> tuple[bool, str | None, dict[str, Any]]:
        now = self._now()
        bar_idx = self._advance_bar(symbol, bar_ts)
        blocked_until = self._symbol_block_until_bar.get(symbol, 0)
        if blocked_until and bar_idx < blocked_until:
            return False, "CANCEL_LOOP_BLOCK", {"symbol": symbol, "blocked_until_bar": blocked_until}

        self._prune(self._order_ts, now, 60.0)
        if self.global_orders_per_min > 0 and len(self._order_ts) >= self.global_orders_per_min:
            return False, "RATE_THROTTLE_BLOCK", {"scope": "global", "limit": self.global_orders_per_min}

        symbol_window = self._symbol_order_ts.setdefault(symbol, deque())
        self._prune(symbol_window, now, 60.0)
        if (
            self.per_symbol_orders_per_min > 0
            and len(symbol_window) >= self.per_symbol_orders_per_min
        ):
            return False, "RATE_THROTTLE_BLOCK", {"scope": "symbol", "symbol": symbol, "limit": self.per_symbol_orders_per_min}

        return True, None, {}

    def record_order(self, symbol: str, bar_ts: datetime) -> None:
        now = self._now()
        self._advance_bar(symbol, bar_ts)
        self._order_ts.append(now)
        symbol_window = self._symbol_order_ts.setdefault(symbol, deque())
        symbol_window.append(now)

    def record_cancel(self, symbol: str, *, bar_ts: datetime, filled: bool) -> None:
        now = self._now()
        self._cancel_ts.append(now)
        self._prune(self._cancel_ts, now, 60.0)

        if filled:
            self._cancel_without_fill[symbol] = 0
            return

        current = self._cancel_without_fill.get(symbol, 0) + 1
        self._cancel_without_fill[symbol] = current
        if (
            self.cancel_loop_max_without_fill > 0
            and current >= self.cancel_loop_max_without_fill
            and self.cancel_loop_block_bars > 0
        ):
            bar_idx = self._advance_bar(symbol, bar_ts)
            self._symbol_block_until_bar[symbol] = bar_idx + self.cancel_loop_block_bars

    def cancel_rate_ok(self) -> bool:
        if self.cancels_per_min <= 0:
            return True
        now = self._now()
        self._prune(self._cancel_ts, now, 60.0)
        return len(self._cancel_ts) < self.cancels_per_min


def _cfg_value(
    cfg: Any,
    *,
    field: str,
    env_keys: tuple[str, ...],
    default: Any,
    cast: type[float] | type[int],
) -> float | int:
    raw = getattr(cfg, field, None) if cfg is not None else None
    if raw is not None:
        try:
            return cast(raw)
        except (TypeError, ValueError):
            pass
    for env_key in env_keys:
        raw_env = get_env(env_key, None)
        if raw_env is None:
            continue
        return cast(get_env(env_key, default, cast=cast))
    return cast(default)


def _ledger_fingerprints(ledger: Any) -> set[tuple[str, str, int, str]]:
    if ledger is None:
        return set()
    seen = getattr(ledger, "_pretrade_seen_fingerprints", None)
    if isinstance(seen, set):
        return seen
    seen = set()
    setattr(ledger, "_pretrade_seen_fingerprints", seen)
    return seen


def validate_pretrade(
    intent: OrderIntent,
    *,
    cfg: Any,
    ledger: Any,
    rate_limiter: SlidingWindowRateLimiter,
) -> tuple[bool, str, dict[str, Any]]:
    """Validate pre-trade controls and return allow/deny decision."""

    max_order_dollars = float(
        _cfg_value(
            cfg,
            field="max_order_dollars",
            env_keys=("MAX_ORDER_DOLLARS", "AI_TRADING_MAX_ORDER_DOLLARS"),
            default=0.0,
            cast=float,
        )
    )
    max_order_shares = int(
        _cfg_value(
            cfg,
            field="max_order_shares",
            env_keys=("MAX_ORDER_SHARES", "AI_TRADING_MAX_ORDER_SHARES"),
            default=0,
            cast=int,
        )
    )
    collar_pct = float(
        _cfg_value(
            cfg,
            field="price_collar_pct",
            env_keys=("PRICE_COLLAR_PCT", "AI_TRADING_PRICE_COLLAR_PCT"),
            default=0.03,
            cast=float,
        )
    )

    qty_abs = abs(int(intent.qty))
    notional_abs = abs(float(intent.notional))
    if (max_order_shares > 0 and qty_abs > max_order_shares) or (
        max_order_dollars > 0 and notional_abs > max_order_dollars
    ):
        return False, "ORDER_SIZE_BLOCK", {"qty": qty_abs, "notional": notional_abs}

    reference = intent.mid if intent.mid and intent.mid > 0 else intent.last_price
    if intent.limit_price is not None and reference and reference > 0:
        deviation = abs(float(intent.limit_price) - float(reference)) / float(reference)
        if deviation > max(0.0, collar_pct):
            return False, "PRICE_COLLAR_BLOCK", {"reference": reference, "limit_price": intent.limit_price, "deviation": deviation}

    if ledger is not None and intent.client_order_id:
        seen_fn = getattr(ledger, "seen_client_order_id", None)
        if callable(seen_fn) and bool(seen_fn(intent.client_order_id)):
            return False, "DUPLICATE_ORDER_BLOCK", {"client_order_id": intent.client_order_id}

    fingerprint = (
        str(intent.symbol).upper(),
        str(intent.side).lower(),
        qty_abs,
        intent.bar_ts.isoformat(),
    )
    fingerprints = _ledger_fingerprints(ledger)
    if fingerprint in fingerprints:
        return False, "DUPLICATE_ORDER_BLOCK", {"fingerprint": fingerprint}

    allowed, reason, details = rate_limiter.allow_order(intent.symbol, intent.bar_ts)
    if not allowed:
        return False, reason or "RATE_THROTTLE_BLOCK", details
    if not rate_limiter.cancel_rate_ok():
        return False, "RATE_THROTTLE_BLOCK", {"scope": "cancel"}

    rate_limiter.record_order(intent.symbol, intent.bar_ts)
    if ledger is not None:
        fingerprints.add(fingerprint)
    return True, "OK", {}
