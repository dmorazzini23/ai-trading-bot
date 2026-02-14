"""Execution attribution helpers (slippage/spread/latency)."""
from __future__ import annotations

from datetime import datetime
from typing import Any


def arrival_slippage_bps(arrival_price: float, fill_price: float, side: str) -> float:
    if arrival_price <= 0:
        return 0.0
    side_norm = str(side).lower()
    if side_norm == "sell":
        return (arrival_price - fill_price) / arrival_price * 10000.0
    return (fill_price - arrival_price) / arrival_price * 10000.0


def spread_paid_bps(bid: float | None, ask: float | None, fill_price: float, side: str) -> float:
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return 0.0
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return 0.0
    side_norm = str(side).lower()
    if side_norm == "sell":
        return (mid - fill_price) / mid * 10000.0
    return (fill_price - mid) / mid * 10000.0


def fill_latency_ms(order_ts: datetime | None, fill_ts: datetime | None) -> float:
    if order_ts is None or fill_ts is None:
        return 0.0
    return max(0.0, (fill_ts - order_ts).total_seconds() * 1000.0)


def compute_attribution_metrics(
    *,
    arrival_price: float | None,
    fill_price: float | None,
    side: str,
    bid: float | None = None,
    ask: float | None = None,
    order_ts: datetime | None = None,
    fill_ts: datetime | None = None,
) -> dict[str, Any]:
    if arrival_price is None or fill_price is None:
        return {}
    return {
        "arrival_slippage_bps": arrival_slippage_bps(arrival_price, fill_price, side),
        "spread_paid_bps": spread_paid_bps(bid, ask, fill_price, side),
        "fill_latency_ms": fill_latency_ms(order_ts, fill_ts),
    }

