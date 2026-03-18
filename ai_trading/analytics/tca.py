"""Transaction Cost Analysis (TCA) primitives."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from ai_trading.logging import get_logger

logger = get_logger(__name__)


def _safe_float(value: Any) -> float | None:
    """Return ``value`` as float when possible."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_utc_datetime(value: Any) -> datetime | None:
    """Parse datetimes from runtime payloads into UTC."""

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


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


def resolve_pending_tca_from_fill(
    *,
    pending_record: Mapping[str, Any],
    fill_price: float,
    fill_qty: float,
    status: str,
    fill_ts: datetime | None = None,
    fee_amount: float | None = None,
    generated_ts: datetime | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Return a resolved TCA payload from a pending TCA record and fill details."""

    fill_price_value = _safe_float(fill_price)
    fill_qty_value = _safe_float(fill_qty)
    if fill_price_value is None or fill_price_value <= 0.0:
        raise ValueError("fill_price must be positive")
    if fill_qty_value is None or fill_qty_value <= 0.0:
        raise ValueError("fill_qty must be positive")

    benchmark_raw = pending_record.get("benchmark")
    benchmark = dict(benchmark_raw) if isinstance(benchmark_raw, Mapping) else {}
    decision_price = _safe_float(pending_record.get("decision_price"))
    if decision_price is None:
        decision_price = _safe_float(pending_record.get("arrival_price"))
    if decision_price is None:
        decision_price = _safe_float(pending_record.get("submit_price_reference"))
    if decision_price is None or decision_price <= 0.0:
        raise ValueError("pending_record_missing_decision_price")

    status_token = str(status or pending_record.get("status") or "filled").strip().lower() or "filled"
    side_token = str(pending_record.get("side") or "buy").strip().lower() or "buy"
    submit_ts = _coerce_utc_datetime(benchmark.get("submit_ts"))
    fill_timestamp = _coerce_utc_datetime(fill_ts) or datetime.now(UTC)

    fee_value = _safe_float(fee_amount)
    if fee_value is None:
        fee_value = _safe_float(pending_record.get("fees"))
    fees = abs(float(fee_value or 0.0))
    requested_qty = _safe_float(pending_record.get("qty"))
    partial_fill = status_token == "partially_filled"
    if (
        not partial_fill
        and requested_qty is not None
        and requested_qty > 0.0
        and fill_qty_value < requested_qty
    ):
        partial_fill = True

    is_bps_value = implementation_shortfall_bps(
        side=side_token,
        decision_price=float(decision_price),
        fill_vwap=float(fill_price_value),
        fees=fees,
        qty=float(fill_qty_value),
    )
    mid_at_arrival = _safe_float(benchmark.get("mid_at_arrival"))
    spread_value = (
        spread_paid_bps(side_token, float(mid_at_arrival), float(fill_price_value))
        if mid_at_arrival is not None
        else None
    )
    latency_value = fill_latency_ms(submit_ts, fill_timestamp) if submit_ts is not None else None

    output_ts = _coerce_utc_datetime(generated_ts) or fill_timestamp
    resolved = dict(pending_record)
    resolved.update(
        {
            "ts": output_ts.isoformat(),
            "status": status_token,
            "order_status": status_token,
            "decision_price": float(decision_price),
            "arrival_price": float(decision_price),
            "submit_price_reference": (
                float(mid_at_arrival) if mid_at_arrival is not None else float(decision_price)
            ),
            "fill_price": float(fill_price_value),
            "fill_vwap": float(fill_price_value),
            "resolved_fill_price": float(fill_price_value),
            "resolved_fill_qty": float(fill_qty_value),
            "qty": float(fill_qty_value),
            "fees": float(fees),
            "partial_fill": bool(partial_fill),
            "is_bps": float(is_bps_value),
            "spread_paid_bps": spread_value,
            "fill_latency_ms": latency_value,
            "pending_event": False,
            "pending_resolved": True,
            "pending_resolved_ts": fill_timestamp.isoformat(),
        }
    )
    if source not in (None, ""):
        resolved["pending_resolved_source"] = str(source)

    benchmark["first_fill_ts"] = fill_timestamp.isoformat()
    if submit_ts is not None:
        benchmark["submit_ts"] = submit_ts.isoformat()
    resolved["benchmark"] = benchmark
    return resolved


def reconcile_pending_tca_with_fill(
    path: str,
    *,
    client_order_id: str | None,
    order_id: str | None = None,
    fill_price: float,
    fill_qty: float,
    status: str,
    fill_ts: datetime | None = None,
    fee_amount: float | None = None,
    source: str | None = None,
) -> tuple[bool, str]:
    """Append a resolved TCA row for a previously pending TCA order when available."""

    identifiers: set[str] = set()
    if client_order_id not in (None, ""):
        identifiers.add(str(client_order_id))
    if order_id not in (None, ""):
        identifiers.add(str(order_id))
    if not identifiers:
        return False, "missing_identifiers"

    if (_safe_float(fill_price) or 0.0) <= 0.0 or (_safe_float(fill_qty) or 0.0) <= 0.0:
        return False, "invalid_fill_payload"

    target = Path(path)
    if not target.exists():
        return False, "tca_path_missing"

    latest_pending: dict[str, Any] | None = None
    already_resolved = False
    try:
        with target.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, Mapping):
                    continue

                row_identifiers: set[str] = set()
                for key in ("client_order_id", "order_id", "broker_order_id"):
                    value = row.get(key)
                    if value in (None, ""):
                        continue
                    row_identifiers.add(str(value))
                if not identifiers.intersection(row_identifiers):
                    continue

                if bool(row.get("pending_event")):
                    latest_pending = dict(row)
                    continue

                resolved_fill = _safe_float(row.get("fill_price"))
                if resolved_fill is None:
                    resolved_fill = _safe_float(row.get("fill_vwap"))
                resolved_status = str(row.get("status") or row.get("order_status") or "").strip().lower()
                if (
                    resolved_fill is not None
                    and resolved_fill > 0.0
                    and resolved_status in {"filled", "partially_filled"}
                ):
                    already_resolved = True
    except OSError:
        return False, "tca_path_unreadable"

    if already_resolved:
        return False, "already_resolved"
    if latest_pending is None:
        return False, "pending_record_missing"

    try:
        resolved = resolve_pending_tca_from_fill(
            pending_record=latest_pending,
            fill_price=float(fill_price),
            fill_qty=float(fill_qty),
            status=status,
            fill_ts=fill_ts,
            fee_amount=fee_amount,
            generated_ts=datetime.now(UTC),
            source=source,
        )
    except ValueError as exc:
        return False, f"invalid_pending_record:{exc}"

    write_tca_record(path, resolved)
    return True, "reconciled"
