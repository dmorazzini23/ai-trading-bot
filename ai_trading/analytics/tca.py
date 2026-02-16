"""Transaction Cost Analysis (TCA) primitives."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from ai_trading.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class ExecutionBenchmark:
    arrival_price: float
    mid_at_arrival: float | None = None
    bid_at_arrival: float | None = None
    ask_at_arrival: float | None = None
    bar_close_price: float | None = None
    decision_ts: datetime | None = None
    submit_ts: datetime | None = None
    first_fill_ts: datetime | None = None


@dataclass(slots=True)
class FillSummary:
    fill_vwap: float | None
    total_qty: float
    fees: float
    status: str
    partial_fill: bool = False


def implementation_shortfall_bps(
    side: str,
    arrival_price: float | None = None,
    fill_vwap: float | None = None,
    fees: float = 0.0,
    qty: float = 0.0,
    decision_price: float | None = None,
) -> float:
    """Return implementation shortfall in bps using signed direction.

    Appendix AA canonical formula:
    IS_bps = sign * (fill_price - decision_price) / decision_price * 10_000
    """

    reference = decision_price if decision_price is not None else arrival_price
    if reference is None:
        raise ValueError("decision_price is required")
    base = float(reference)
    if fill_vwap is None:
        raise ValueError("fill_vwap is required")
    fill = float(fill_vwap)
    if base <= 0:
        return 0.0
    sign = 1.0 if str(side).lower() == "buy" else -1.0
    is_bps = sign * (fill - base) / base * 10_000.0
    qty_abs = abs(float(qty))
    if qty_abs > 0 and fees:
        fee_bps = (float(fees) / (qty_abs * base)) * 10_000.0
        is_bps += fee_bps
    return float(is_bps)


def spread_paid_bps(side: str, mid_at_arrival: float, fill_vwap: float) -> float:
    mid = float(mid_at_arrival)
    fill = float(fill_vwap)
    if mid <= 0:
        return 0.0
    if str(side).lower() == "buy":
        return max(0.0, (fill - mid) / mid * 10_000.0)
    return max(0.0, (mid - fill) / mid * 10_000.0)


def fill_latency_ms(submit_ts: datetime, first_fill_ts: datetime) -> int:
    submit_utc = submit_ts if submit_ts.tzinfo else submit_ts.replace(tzinfo=UTC)
    fill_utc = first_fill_ts if first_fill_ts.tzinfo else first_fill_ts.replace(tzinfo=UTC)
    return int(max(0.0, (fill_utc - submit_utc).total_seconds() * 1000.0))


def cancel_replace_rate(window: list[Mapping[str, Any]]) -> float:
    if not window:
        return 0.0
    submits = 0
    cancels_replaces = 0
    for event in window:
        action = str(event.get("action", "")).lower()
        if action in {"submit", "new"}:
            submits += 1
        if action in {"cancel", "replace", "cancel_replace"}:
            cancels_replaces += 1
    if submits <= 0:
        return 0.0
    return float(cancels_replaces) / float(submits)


def build_tca_record(
    *,
    client_order_id: str,
    symbol: str,
    side: str,
    benchmark: ExecutionBenchmark,
    fill: FillSummary,
    sleeve: str | None = None,
    regime_profile: str | None = None,
    provider: str | None = None,
    order_type: str | None = None,
    quote_proxy: bool = False,
    generated_ts: datetime | None = None,
) -> dict[str, Any]:
    arrival = float(benchmark.arrival_price)
    fill_vwap = float(fill.fill_vwap) if fill.fill_vwap is not None else arrival
    is_bps = implementation_shortfall_bps(
        side=side,
        arrival_price=arrival,
        fill_vwap=fill_vwap,
        fees=float(fill.fees),
        qty=float(fill.total_qty),
    )
    spread_bps = None
    if benchmark.mid_at_arrival is not None:
        spread_bps = spread_paid_bps(side, benchmark.mid_at_arrival, fill_vwap)

    latency = None
    if benchmark.submit_ts is not None and benchmark.first_fill_ts is not None:
        latency = fill_latency_ms(benchmark.submit_ts, benchmark.first_fill_ts)

    record = {
        "ts": (generated_ts if generated_ts is not None else datetime.now(UTC)).isoformat(),
        "client_order_id": client_order_id,
        "symbol": symbol,
        "side": side,
        "sleeve": sleeve,
        "regime_profile": regime_profile,
        "provider": provider,
        "order_type": order_type,
        "decision_price": arrival,
        "submit_price_reference": (
            float(benchmark.mid_at_arrival) if benchmark.mid_at_arrival is not None else arrival
        ),
        "fill_price": fill_vwap,
        "arrival_price": arrival,
        "fill_vwap": fill_vwap,
        "qty": float(fill.total_qty),
        "fees": float(fill.fees),
        "status": fill.status,
        "partial_fill": bool(fill.partial_fill),
        "is_bps": is_bps,
        "spread_paid_bps": spread_bps,
        "fill_latency_ms": latency,
        "quote_proxy": bool(quote_proxy),
        "benchmark": {
            "mid_at_arrival": benchmark.mid_at_arrival,
            "bid_at_arrival": benchmark.bid_at_arrival,
            "ask_at_arrival": benchmark.ask_at_arrival,
            "bar_close_price": benchmark.bar_close_price,
            "decision_ts": benchmark.decision_ts.isoformat() if benchmark.decision_ts else None,
            "submit_ts": benchmark.submit_ts.isoformat() if benchmark.submit_ts else None,
            "first_fill_ts": benchmark.first_fill_ts.isoformat() if benchmark.first_fill_ts else None,
        },
    }
    return record


def write_tca_record(path: str, record: Mapping[str, Any]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True))
        handle.write("\n")
    logger.info(
        "TCA_RECORD_WRITTEN",
        extra={"path": str(dest), "client_order_id": record.get("client_order_id")},
    )
