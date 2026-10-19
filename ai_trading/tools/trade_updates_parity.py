"""Compare shadow trade-update stream fills with the polled fill record.

Usage::

    python -m ai_trading.tools.trade_updates_parity --session YYYY-MM-DD

Reads ``runtime/trade_updates_stream.jsonl`` (shadow artifact) and
``runtime/fill_events.jsonl`` (authoritative polled fills) read-only, writes
the JSON report to stdout (and ``--output`` when given) and a short summary
to stderr.  Nothing here writes
the OMS event store or the fill record.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Iterator, Mapping

from ai_trading.config.management import get_env
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path
from ai_trading.telemetry import trade_updates_stream as tus

_FILL_EVENTS_DEFAULT = "runtime/fill_events.jsonl"
_STREAM_FILL_EVENTS = frozenset({"fill", "partial_fill"})
_QTY_TOLERANCE = 1e-6
_MAX_MISMATCHES = 200
AUTHORITY_PRICE_DELTA_P95_BPS = 1.0

__all__ = [
    "compute_trade_updates_parity",
    "load_polled_fill_records",
    "load_stream_records",
    "main",
    "refresh_parity_latest",
    "resolve_polled_fill_path",
]


def resolve_polled_fill_path() -> Path:
    raw = str(get_env("AI_TRADING_FILL_EVENTS_PATH", "", cast=str) or "").strip()
    return resolve_runtime_artifact_path(raw or None, default_relative=_FILL_EVENTS_DEFAULT)


def _session_markers(session: str) -> tuple[str, ...]:
    # A New York session date spans its own UTC date and the next one.
    day = date.fromisoformat(session)
    return (f'"{day.isoformat()}', f'"{(day + timedelta(days=1)).isoformat()}')


def _iter_jsonl(path: Path, markers: tuple[str, ...]) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if markers and not any(marker in line for marker in markers):
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                yield row


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _parse_ts(value: Any) -> datetime | None:
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


def _stream_record_ts(row: Mapping[str, Any]) -> Any:
    return row.get("event_timestamp") or row.get("received_at")


def load_stream_records(path: Path, session: str) -> list[dict[str, Any]]:
    return [
        row
        for row in _iter_jsonl(path, _session_markers(session))
        if tus.session_date_for(_stream_record_ts(row)) == session
    ]


def load_polled_fill_records(path: Path, session: str) -> list[dict[str, Any]]:
    return [
        row
        for row in _iter_jsonl(path, _session_markers(session))
        if str(row.get("event") or "fill_recorded") == "fill_recorded"
        and tus.session_date_for(row.get("ts")) == session
    ]


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def _stream_orders(records: Iterable[Mapping[str, Any]]) -> tuple[dict[str, dict[str, Any]], int]:
    orders: dict[str, dict[str, Any]] = {}
    fill_events = 0
    for row in records:
        if str(row.get("event") or "").lower() not in _STREAM_FILL_EVENTS:
            continue
        order_id = str(row.get("order_id") or "").strip()
        filled_qty = _float(row.get("filled_qty"))
        if not order_id or filled_qty is None:
            continue
        fill_events += 1
        current = orders.get(order_id)
        if current is not None and filled_qty <= current["filled_qty"]:
            continue
        order_ts = row.get("order_timestamps")
        filled_at = order_ts.get("filled_at") if isinstance(order_ts, Mapping) else None
        orders[order_id] = {
            "order_id": order_id,
            "symbol": row.get("symbol"),
            "filled_qty": filled_qty,
            "filled_avg_price": _float(row.get("filled_avg_price")),
            "broker_fill_ts": row.get("event_timestamp") or filled_at,
            "event": row.get("event"),
        }
    return orders, fill_events


def _polled_orders(records: Iterable[Mapping[str, Any]]) -> tuple[dict[str, dict[str, Any]], int]:
    orders: dict[str, dict[str, Any]] = {}
    seen_fill_ids: set[tuple[str, str]] = set()
    fill_count = 0
    for row in records:
        order_id = str(row.get("order_id") or row.get("broker_order_id") or "").strip()
        qty = _float(row.get("fill_qty"))
        if qty is None:
            qty = _float(row.get("qty"))
        if not order_id or qty is None:
            continue
        fill_id = str(row.get("fill_id") or "").strip()
        if fill_id:
            if (order_id, fill_id) in seen_fill_ids:
                continue
            seen_fill_ids.add((order_id, fill_id))
        fill_count += 1
        price = _float(row.get("fill_price"))
        if price is None:
            price = _float(row.get("entry_price"))
        entry = orders.setdefault(
            order_id,
            {
                "order_id": order_id,
                "symbol": row.get("symbol"),
                "filled_qty": 0.0,
                "notional": 0.0,
                "priced_qty": 0.0,
                "detected_at": None,
            },
        )
        entry["filled_qty"] += qty
        if price is not None:
            entry["notional"] += price * qty
            entry["priced_qty"] += qty
        detected = _parse_ts(row.get("ts"))
        if detected is not None and (
            entry["detected_at"] is None or detected > entry["detected_at"]
        ):
            entry["detected_at"] = detected
    for entry in orders.values():
        priced_qty = entry.pop("priced_qty")
        notional = entry.pop("notional")
        entry["filled_avg_price"] = notional / priced_qty if priced_qty > 0 else None
    return orders, fill_count


def compute_trade_updates_parity(
    stream_records: Iterable[Mapping[str, Any]],
    polled_records: Iterable[Mapping[str, Any]],
    *,
    session: str | None = None,
) -> dict[str, Any]:
    """Match stream fills to polled fills by order id and cumulative filled qty."""

    stream_list = list(stream_records)
    stream_orders, stream_fill_events = _stream_orders(stream_list)
    polled_orders, polled_fill_count = _polled_orders(polled_records)
    matched = 0
    stream_only: list[str] = []
    polled_only: list[str] = []
    mismatches: list[dict[str, Any]] = []
    price_deltas: list[float] = []
    latencies: list[float] = []

    for order_id in sorted(set(stream_orders) | set(polled_orders)):
        stream = stream_orders.get(order_id)
        polled = polled_orders.get(order_id)
        if stream is None and polled is not None:
            polled_only.append(order_id)
            mismatches.append(
                {
                    "order_id": order_id,
                    "kind": "polled_only",
                    "symbol": polled.get("symbol"),
                    "polled_filled_qty": polled["filled_qty"],
                }
            )
            continue
        if polled is None and stream is not None:
            stream_only.append(order_id)
            mismatches.append(
                {
                    "order_id": order_id,
                    "kind": "stream_only",
                    "symbol": stream.get("symbol"),
                    "stream_filled_qty": stream["filled_qty"],
                }
            )
            continue
        if stream is None or polled is None:
            continue
        if abs(stream["filled_qty"] - polled["filled_qty"]) > _QTY_TOLERANCE:
            mismatches.append(
                {
                    "order_id": order_id,
                    "kind": "qty_mismatch",
                    "symbol": stream.get("symbol") or polled.get("symbol"),
                    "stream_filled_qty": stream["filled_qty"],
                    "polled_filled_qty": polled["filled_qty"],
                }
            )
            continue
        matched += 1
        stream_price = stream.get("filled_avg_price")
        polled_price = polled.get("filled_avg_price")
        if stream_price and polled_price:
            price_deltas.append(abs(stream_price - polled_price) / polled_price * 10_000.0)
        broker_ts = _parse_ts(stream.get("broker_fill_ts"))
        detected = polled.get("detected_at")
        if broker_ts is not None and isinstance(detected, datetime):
            latencies.append((detected - broker_ts).total_seconds())

    qty_mismatch = sum(1 for item in mismatches if item["kind"] == "qty_mismatch")
    p95 = _percentile(price_deltas, 0.95)
    clean = bool(
        stream_orders
        and not stream_only
        and not polled_only
        and qty_mismatch == 0
        and (p95 is None or p95 < AUTHORITY_PRICE_DELTA_P95_BPS)
    )
    return {
        "session": session,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "shadow_only": True,
        "stream_event_count": len(stream_list),
        "stream_fill_count": stream_fill_events,
        "stream_fill_orders": len(stream_orders),
        "polled_fill_count": polled_fill_count,
        "polled_fill_orders": len(polled_orders),
        "matched": matched,
        "stream_only": len(stream_only),
        "polled_only": len(polled_only),
        "qty_mismatch": qty_mismatch,
        "price_delta_bps_median": _round(_percentile(price_deltas, 0.5)),
        "price_delta_bps_p95": _round(p95),
        "latency_seconds_median": _round(_percentile(latencies, 0.5), 3),
        "latency_seconds_p95": _round(_percentile(latencies, 0.95), 3),
        "clean": clean,
        "mismatches": mismatches[:_MAX_MISMATCHES],
        "mismatches_truncated": len(mismatches) > _MAX_MISMATCHES,
    }


def build_session_report(
    session: str,
    *,
    stream_path: Path | None = None,
    polled_path: Path | None = None,
) -> dict[str, Any]:
    stream_file = stream_path or tus.resolve_stream_artifact_path()
    polled_file = polled_path or resolve_polled_fill_path()
    stream_exists = stream_file.exists()
    polled_exists = polled_file.exists()
    stream_records = load_stream_records(stream_file, session) if stream_exists else []
    polled_records = load_polled_fill_records(polled_file, session) if polled_exists else []
    report = compute_trade_updates_parity(stream_records, polled_records, session=session)
    if not stream_exists:
        reason = "stream_artifact_missing"
    elif not stream_records:
        reason = "no_stream_events_for_session"
    elif not polled_exists:
        reason = "polled_fill_record_missing"
    else:
        reason = "ok"
    report["available"] = bool(stream_exists and stream_records)
    report["reason"] = reason
    report["sources"] = {
        "stream_path": str(stream_file),
        "stream_exists": stream_exists,
        "polled_path": str(polled_file),
        "polled_exists": polled_exists,
    }
    return report


def _today_session() -> str:
    return datetime.now(UTC).astimezone(tus.SESSION_TZ).date().isoformat()


_REFRESH_LOCK = Lock()
_REFRESH_CACHE: dict[str, Any] = {}


def _file_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def refresh_parity_latest(session: str | None = None) -> dict[str, Any]:
    """Recompute today's parity and write ``runtime/trade_updates_parity_latest.json``.

    Called from the stream thread on an interval, never from a request path.
    Skips recomputation when neither input file changed.
    """

    target_session = session or _today_session()
    stream_file = tus.resolve_stream_artifact_path()
    polled_file = resolve_polled_fill_path()
    signature = (
        target_session,
        _file_signature(stream_file),
        _file_signature(polled_file),
    )
    with _REFRESH_LOCK:
        cached = _REFRESH_CACHE.get("summary")
        if cached is not None and _REFRESH_CACHE.get("signature") == signature:
            return dict(cached)
        report = build_session_report(
            target_session, stream_path=stream_file, polled_path=polled_file
        )
        output = tus.resolve_parity_latest_path(for_write=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, output)
        _REFRESH_CACHE["signature"] = signature
        _REFRESH_CACHE["summary"] = report
    tus.record_parity_summary(report)
    return dict(report)


def format_summary(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"trade_updates_parity session={report.get('session')} "
            f"available={report.get('available')} reason={report.get('reason')}",
            f"  stream fills={report.get('stream_fill_count')} "
            f"(orders={report.get('stream_fill_orders')}) "
            f"polled fills={report.get('polled_fill_count')} "
            f"(orders={report.get('polled_fill_orders')})",
            f"  matched={report.get('matched')} stream_only={report.get('stream_only')} "
            f"polled_only={report.get('polled_only')} qty_mismatch={report.get('qty_mismatch')}",
            f"  price_delta_bps median={report.get('price_delta_bps_median')} "
            f"p95={report.get('price_delta_bps_p95')} "
            f"latency_s median={report.get('latency_seconds_median')} clean={report.get('clean')}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default=None, help="New York session date YYYY-MM-DD (default: today)")
    parser.add_argument("--stream-path", default=None, help="override the shadow stream artifact path")
    parser.add_argument("--polled-path", default=None, help="override the polled fill_events.jsonl path")
    parser.add_argument("--output", default=None, help="also write the JSON report to this file")
    args = parser.parse_args(argv)
    session = args.session or _today_session()
    try:
        date.fromisoformat(session)
    except ValueError:
        parser.error(f"--session must be YYYY-MM-DD, got {session!r}")
    report = build_session_report(
        session,
        stream_path=Path(args.stream_path) if args.stream_path else None,
        polled_path=Path(args.polled_path) if args.polled_path else None,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    sys.stderr.write(format_summary(report) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
