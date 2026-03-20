"""Transaction Cost Analysis (TCA) primitives."""
from __future__ import annotations

import json
from collections import deque
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


def _collect_row_identifiers(row: Mapping[str, Any]) -> set[str]:
    """Return normalized identifier tokens from a runtime record."""

    tokens: set[str] = set()
    for key in (
        "client_order_id",
        "order_id",
        "broker_order_id",
        "alpaca_order_id",
        "fill_id",
        "execution_id",
        "id",
    ):
        value = row.get(key)
        if value in (None, ""):
            continue
        token = str(value).strip()
        if token:
            tokens.add(token)
    return tokens


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
    fallback_identifiers: list[str] | tuple[str, ...] | set[str] | None = None,
    symbol: str | None = None,
    side: str | None = None,
    fill_price: float,
    fill_qty: float,
    status: str,
    fill_ts: datetime | None = None,
    fee_amount: float | None = None,
    source: str | None = None,
    allow_symbol_qty_fallback: bool = False,
    fallback_window_seconds: float = 8.0 * 3600.0,
    qty_tolerance_ratio: float = 0.05,
) -> tuple[bool, str]:
    """Append a resolved TCA row for a previously pending TCA order when available."""

    identifiers: set[str] = set()
    if client_order_id not in (None, ""):
        identifiers.add(str(client_order_id))
    if order_id not in (None, ""):
        identifiers.add(str(order_id))
    if fallback_identifiers:
        for value in fallback_identifiers:
            if value in (None, ""):
                continue
            token = str(value).strip()
            if token:
                identifiers.add(token)

    fill_qty_value = _safe_float(fill_qty)
    if (_safe_float(fill_price) or 0.0) <= 0.0 or (fill_qty_value or 0.0) <= 0.0:
        return False, "invalid_fill_payload"

    symbol_token = str(symbol or "").strip().upper()
    side_token = str(side or "").strip().lower()
    if side_token not in {"buy", "sell"}:
        side_token = ""
    fallback_enabled = bool(allow_symbol_qty_fallback and symbol_token and side_token and fill_qty_value)
    if not identifiers and not fallback_enabled:
        return False, "missing_identifiers"

    fill_ts_utc = _coerce_utc_datetime(fill_ts)
    try:
        fallback_window_s = float(fallback_window_seconds)
    except (TypeError, ValueError):
        fallback_window_s = 8.0 * 3600.0
    if not fallback_window_s or fallback_window_s <= 0.0:
        fallback_window_s = 8.0 * 3600.0
    fallback_window_s = max(60.0, min(fallback_window_s, 14.0 * 86400.0))

    try:
        qty_ratio = float(qty_tolerance_ratio)
    except (TypeError, ValueError):
        qty_ratio = 0.05
    qty_ratio = max(0.0, min(qty_ratio, 1.0))
    qty_tolerance_abs = max(0.5, float(fill_qty_value or 0.0) * qty_ratio)

    target = Path(path)
    if not target.exists():
        return False, "tca_path_missing"

    latest_pending: dict[str, Any] | None = None
    latest_pending_rank: tuple[int, float] | None = None
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

                row_identifiers = _collect_row_identifiers(row)
                id_matched = bool(identifiers and identifiers.intersection(row_identifiers))
                fallback_matched = False
                if not id_matched and fallback_enabled:
                    row_symbol = str(row.get("symbol") or "").strip().upper()
                    row_side = str(row.get("side") or "").strip().lower()
                    row_qty = _safe_float(row.get("qty"))
                    if (
                        row_symbol == symbol_token
                        and row_side == side_token
                        and row_qty is not None
                        and row_qty > 0.0
                        and abs(float(row_qty) - float(fill_qty_value or 0.0)) <= qty_tolerance_abs
                    ):
                        row_ts = _coerce_utc_datetime(row.get("ts"))
                        if row_ts is None or fill_ts_utc is None:
                            fallback_matched = True
                        else:
                            age_s = abs((fill_ts_utc - row_ts).total_seconds())
                            fallback_matched = age_s <= fallback_window_s
                if not id_matched and not fallback_matched:
                    continue

                if bool(row.get("pending_event")):
                    rank = 2 if id_matched else 1
                    row_ts = _coerce_utc_datetime(row.get("ts"))
                    row_ts_epoch = float(row_ts.timestamp()) if row_ts is not None else 0.0
                    row_rank = (rank, row_ts_epoch)
                    if latest_pending_rank is None or row_rank > latest_pending_rank:
                        latest_pending_rank = row_rank
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
            fill_qty=float(fill_qty_value),
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


def finalize_stale_pending_tca(
    path: str,
    *,
    stale_after_seconds: float,
    now: datetime | None = None,
    max_records: int = 500,
    source: str | None = "maintenance_stale_nonfill",
    fill_events_path: str | None = None,
    fill_match_window_seconds: float = 24.0 * 3600.0,
    fill_qty_tolerance_ratio: float = 0.05,
    fill_events_max_records: int = 200000,
    compact_matched_pending: bool = False,
) -> dict[str, Any]:
    """Finalize stale pending TCA rows as terminal non-fill events.

    The function rewrites stale pending rows in place (rather than appending
    extra rows) so simple ``pending_event=true`` counts decrease over time.

    When ``fill_events_path`` is provided, pending rows are not finalized if a
    matching fill is present in canonical fill events (id match first, then
    strict symbol/side/qty/time fallback). When ``compact_matched_pending`` is
    true, matched pending rows are rewritten in-place as resolved superseded
    entries so pending backlog counters track truly unmatched rows.
    """

    target = Path(path)
    if not target.exists():
        return {
            "ok": False,
            "reason": "tca_path_missing",
            "path": str(target),
            "scanned_pending": 0,
            "finalized": 0,
        }

    stale_after = _safe_float(stale_after_seconds)
    if stale_after is None or stale_after <= 0.0:
        return {
            "ok": False,
            "reason": "invalid_stale_after_seconds",
            "path": str(target),
            "scanned_pending": 0,
            "finalized": 0,
        }
    max_finalize = int(max(max_records, 0))
    if max_finalize <= 0:
        return {
            "ok": True,
            "reason": "max_records_zero",
            "path": str(target),
            "scanned_pending": 0,
            "finalized": 0,
        }

    now_utc = _coerce_utc_datetime(now) or datetime.now(UTC)
    resolved_ids: set[str] = set()
    entries: list[tuple[str, Any]] = []
    malformed = 0
    scanned_pending = 0

    try:
        with target.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    entries.append(("raw", raw_line.rstrip("\n")))
                    continue
                if not isinstance(row, Mapping):
                    entries.append(("raw", raw_line.rstrip("\n")))
                    continue
                row_dict = dict(row)
                entries.append(("json", row_dict))
                if bool(row.get("pending_event")):
                    scanned_pending += 1
                    continue
                identifiers = _collect_row_identifiers(row)
                if not identifiers:
                    continue

                status_token = str(row.get("status") or row.get("order_status") or "").strip().lower()
                fill_price = _safe_float(row.get("fill_price"))
                if fill_price is None:
                    fill_price = _safe_float(row.get("fill_vwap"))
                if (
                    fill_price is not None
                    and fill_price > 0.0
                    and status_token in {"filled", "partially_filled"}
                ):
                    resolved_ids.update(identifiers)
    except OSError:
        return {
            "ok": False,
            "reason": "tca_path_unreadable",
            "path": str(target),
            "scanned_pending": 0,
            "finalized": 0,
        }

    fill_event_ids: set[str] = set()
    fill_event_fallback_rows: dict[tuple[str, str], list[tuple[float, datetime | None]]] = {}
    fill_events_scanned = 0
    fill_events_considered = 0
    if fill_events_path not in (None, ""):
        fill_path = Path(str(fill_events_path))
        if fill_path.exists():
            try:
                max_fill_rows = int(max(fill_events_max_records, 0))
            except (TypeError, ValueError):
                max_fill_rows = 200000
            max_fill_rows = max(1, min(max_fill_rows, 1_000_000))
            ring: deque[str] = deque(maxlen=max_fill_rows)
            try:
                with fill_path.open("r", encoding="utf-8", errors="ignore") as handle:
                    for line in handle:
                        ring.append(line)
            except OSError:
                ring = deque()
            for raw_line in ring:
                payload = raw_line.strip()
                if not payload:
                    continue
                try:
                    row = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, Mapping):
                    continue
                fill_events_scanned += 1
                if str(row.get("event") or "").strip().lower() != "fill_recorded":
                    continue
                fill_price = _safe_float(row.get("fill_price"))
                fill_qty = _safe_float(row.get("fill_qty"))
                if fill_price is None:
                    fill_price = _safe_float(row.get("entry_price"))
                if fill_qty is None:
                    fill_qty = _safe_float(row.get("qty"))
                if (fill_price or 0.0) <= 0.0 or (fill_qty or 0.0) <= 0.0:
                    continue
                fill_events_considered += 1
                fill_event_ids.update(_collect_row_identifiers(row))
                sym = str(row.get("symbol") or "").strip().upper()
                side = str(row.get("side") or "").strip().lower()
                if not sym or side not in {"buy", "sell"}:
                    continue
                fill_row_ts = _coerce_utc_datetime(row.get("entry_time") or row.get("ts"))
                fill_event_fallback_rows.setdefault((sym, side), []).append(
                    (float(fill_qty), fill_row_ts)
                )

    try:
        fill_match_window_s = float(fill_match_window_seconds)
    except (TypeError, ValueError):
        fill_match_window_s = 24.0 * 3600.0
    if not fill_match_window_s or fill_match_window_s <= 0.0:
        fill_match_window_s = 24.0 * 3600.0
    fill_match_window_s = max(300.0, min(fill_match_window_s, 30.0 * 86400.0))
    try:
        fill_qty_ratio = float(fill_qty_tolerance_ratio)
    except (TypeError, ValueError):
        fill_qty_ratio = 0.05
    fill_qty_ratio = max(0.0, min(fill_qty_ratio, 1.0))

    finalized_count = 0
    compacted_resolved_matches = 0
    compacted_fill_event_matches = 0
    skipped_not_stale = 0
    skipped_already_resolved = 0
    skipped_fill_event_match = 0
    skipped_already_finalized = 0
    compact_enabled = bool(compact_matched_pending)

    def _build_compacted_row(row: dict[str, Any], *, kind: str) -> dict[str, Any]:
        compacted = dict(row)
        compacted["pending_event"] = False
        compacted["pending_resolved"] = True
        compacted["pending_resolved_ts"] = now_utc.isoformat()
        compacted["pending_compacted"] = True
        compacted["pending_compaction_kind"] = kind
        compacted["pending_terminal_nonfill"] = False
        if source not in (None, ""):
            compacted["pending_resolved_source"] = str(source)
        status_token = str(compacted.get("order_status") or compacted.get("status") or "").strip()
        if not status_token:
            status_token = "matched_fill_compacted"
        compacted["status"] = status_token
        compacted["order_status"] = status_token
        requested_qty = _safe_float(compacted.get("requested_qty"))
        row_qty = _safe_float(compacted.get("qty"))
        if requested_qty in (None, 0.0) and row_qty not in (None, 0.0):
            compacted["requested_qty"] = float(row_qty)
        return compacted

    for index, (entry_kind, entry_payload) in enumerate(entries):
        if entry_kind != "json":
            continue
        if not isinstance(entry_payload, dict):
            continue
        pending_row = entry_payload
        if not bool(pending_row.get("pending_event")):
            continue
        identifiers = _collect_row_identifiers(pending_row)
        changes_count = finalized_count + compacted_resolved_matches + compacted_fill_event_matches
        if changes_count >= max_finalize:
            break
        if identifiers and identifiers.intersection(resolved_ids):
            if compact_enabled:
                entries[index] = ("json", _build_compacted_row(pending_row, kind="resolved_fill_match"))
                compacted_resolved_matches += 1
                continue
            skipped_already_resolved += 1
            continue
        if identifiers and identifiers.intersection(fill_event_ids):
            if compact_enabled:
                entries[index] = ("json", _build_compacted_row(pending_row, kind="fill_event_id_match"))
                compacted_fill_event_matches += 1
                continue
            skipped_fill_event_match += 1
            continue
        pending_ts = _coerce_utc_datetime(pending_row.get("ts"))
        if pending_ts is None:
            skipped_not_stale += 1
            continue
        pending_symbol = str(pending_row.get("symbol") or "").strip().upper()
        pending_side = str(pending_row.get("side") or "").strip().lower()
        pending_qty = _safe_float(pending_row.get("qty"))
        if (
            pending_symbol
            and pending_side in {"buy", "sell"}
            and pending_qty is not None
            and pending_qty > 0.0
        ):
            qty_tolerance_abs = max(0.5, float(pending_qty) * fill_qty_ratio)
            fallback_fill_match = False
            for fill_qty, fill_ts in fill_event_fallback_rows.get((pending_symbol, pending_side), []):
                if abs(float(fill_qty) - float(pending_qty)) > qty_tolerance_abs:
                    continue
                if fill_ts is None:
                    fallback_fill_match = True
                    break
                age_delta_s = abs((fill_ts - pending_ts).total_seconds())
                if age_delta_s <= fill_match_window_s:
                    fallback_fill_match = True
                    break
            if fallback_fill_match:
                if compact_enabled:
                    entries[index] = ("json", _build_compacted_row(pending_row, kind="fill_event_fallback_match"))
                    compacted_fill_event_matches += 1
                    continue
                skipped_fill_event_match += 1
                continue
        age_seconds = max(0.0, (now_utc - pending_ts).total_seconds())
        if age_seconds < float(stale_after):
            skipped_not_stale += 1
            continue

        terminal_status = str(
            pending_row.get("order_status") or pending_row.get("pending_reason") or "no_fill"
        ).strip().lower() or "no_fill"
        requested_qty = _safe_float(pending_row.get("qty"))
        terminal_row = dict(pending_row)
        terminal_row.update(
            {
                "ts": now_utc.isoformat(),
                "status": terminal_status,
                "order_status": terminal_status,
                "pending_event": False,
                "pending_resolved": True,
                "pending_resolved_ts": now_utc.isoformat(),
                "pending_terminal_nonfill": True,
                "tca_finalization_kind": "nonfill_terminal",
                "fill_price": None,
                "fill_vwap": None,
                "resolved_fill_price": None,
                "resolved_fill_qty": 0.0,
                "qty": 0.0,
                "is_bps": None,
                "spread_paid_bps": None,
                "fill_latency_ms": None,
                "partial_fill": False,
            }
        )
        if requested_qty is not None and requested_qty > 0.0:
            terminal_row["requested_qty"] = float(requested_qty)
        if source not in (None, ""):
            terminal_row["pending_resolved_source"] = str(source)
        entries[index] = ("json", terminal_row)
        finalized_count += 1

    pending_remaining = 0
    for entry_kind, entry_payload in entries:
        if entry_kind == "json" and isinstance(entry_payload, Mapping):
            if bool(entry_payload.get("pending_event")):
                pending_remaining += 1

    if finalized_count > 0 or compacted_resolved_matches > 0 or compacted_fill_event_matches > 0:
        try:
            tmp_path = target.with_suffix(f"{target.suffix}.tmp")
            with tmp_path.open("w", encoding="utf-8") as handle:
                for entry_kind, entry_payload in entries:
                    if entry_kind == "json" and isinstance(entry_payload, dict):
                        handle.write(json.dumps(entry_payload, sort_keys=True, default=str))
                    else:
                        handle.write(str(entry_payload))
                    handle.write("\n")
            tmp_path.replace(target)
        except OSError:
            return {
                "ok": False,
                "reason": "tca_path_unwritable",
                "path": str(target),
                "scanned_pending": int(scanned_pending),
                "finalized": 0,
                "compacted_resolved_matches": int(compacted_resolved_matches),
                "compacted_fill_event_matches": int(compacted_fill_event_matches),
                "pending_remaining": int(pending_remaining),
            }
        log_extra = {
            "path": str(target),
            "stale_after_seconds": float(stale_after),
            "scanned_pending": int(scanned_pending),
            "finalized": int(finalized_count),
            "compacted_resolved_matches": int(compacted_resolved_matches),
            "compacted_fill_event_matches": int(compacted_fill_event_matches),
            "skipped_not_stale": int(skipped_not_stale),
            "skipped_already_resolved": int(skipped_already_resolved),
            "skipped_fill_event_match": int(skipped_fill_event_match),
            "skipped_already_finalized": int(skipped_already_finalized),
            "fill_events_scanned": int(fill_events_scanned),
            "fill_events_considered": int(fill_events_considered),
            "max_records": int(max_finalize),
            "pending_remaining": int(pending_remaining),
        }
        if finalized_count > 0:
            logger.info("TCA_PENDING_NONFILL_FINALIZED", extra=log_extra)
        if compacted_resolved_matches > 0 or compacted_fill_event_matches > 0:
            logger.info("TCA_PENDING_MATCHED_COMPACTED", extra=log_extra)

    return {
        "ok": True,
        "reason": "processed",
        "path": str(target),
        "stale_after_seconds": float(stale_after),
        "scanned_pending": int(scanned_pending),
        "finalized": int(finalized_count),
        "compacted_resolved_matches": int(compacted_resolved_matches),
        "compacted_fill_event_matches": int(compacted_fill_event_matches),
        "skipped_not_stale": int(skipped_not_stale),
        "skipped_already_resolved": int(skipped_already_resolved),
        "skipped_fill_event_match": int(skipped_fill_event_match),
        "skipped_already_finalized": int(skipped_already_finalized),
        "fill_events_scanned": int(fill_events_scanned),
        "fill_events_considered": int(fill_events_considered),
        "malformed_rows": int(malformed),
        "max_records": int(max_finalize),
        "pending_remaining": int(pending_remaining),
    }
