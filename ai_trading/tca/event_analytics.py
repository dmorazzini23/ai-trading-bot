"""Event-driven TCA analytics from immutable OMS events."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import math
from statistics import median
from typing import Any


def _parse_ts(raw: Any) -> datetime | None:
    if raw in (None, ""):
        return None
    text = str(raw).strip()
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


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _load_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload_raw = row.get("payload_json")
    if isinstance(payload_raw, str):
        try:
            decoded = json.loads(payload_raw)
            if isinstance(decoded, dict):
                return decoded
        except json.JSONDecodeError:
            pass
    payload = row.get("payload")
    if isinstance(payload, dict):
        return dict(payload)
    return {}


def summarize_oms_event_tca(
    *,
    database_url: str | None = None,
    intent_store_path: str | None = None,
    lookback_days: int | None = None,
    limit: int = 50000,
) -> dict[str, Any]:
    """Summarize execution quality metrics directly from immutable OMS events."""

    from ai_trading.oms.event_store import EventStore

    event_store = EventStore(path=intent_store_path, url=database_url)
    cutoff = None
    if lookback_days is not None and int(lookback_days) > 0:
        cutoff = datetime.now(UTC) - timedelta(days=int(lookback_days))

    slippage_samples: list[float] = []
    fills_notional = 0.0
    fills_count = 0
    submit_ack_count = 0
    submit_reject_count = 0
    scanned = 0
    decision_count = 0
    decision_inserted_count = 0

    try:
        rows = event_store.list_oms_events(limit=max(1, int(limit)))
        for row in rows:
            scanned += 1
            event_ts = _parse_ts(row.get("event_ts"))
            if cutoff is not None and (event_ts is None or event_ts < cutoff):
                continue
            event_type = str(row.get("event_type") or "").strip().upper()
            payload = _load_payload(row)

            if event_type == "SUBMIT_ACK":
                submit_ack_count += 1
            elif event_type == "SUBMIT_REJECT":
                submit_reject_count += 1
            elif event_type in {"ORDER_FILLED", "ORDER_PARTIALLY_FILLED"}:
                fills_count += 1
                qty = _as_float(payload.get("fill_qty"))
                if qty is None:
                    qty = _as_float(payload.get("qty"))
                price = _as_float(payload.get("fill_price"))
                if price is None:
                    price = _as_float(payload.get("price"))
                expected_price = _as_float(payload.get("expected_price"))
                if qty is not None and price is not None and qty > 0 and price > 0:
                    fills_notional += abs(float(qty) * float(price))
                if (
                    expected_price is not None
                    and price is not None
                    and expected_price > 0
                    and price > 0
                ):
                    slippage_bps = ((price - expected_price) / expected_price) * 10000.0
                    slippage_samples.append(float(slippage_bps))

        decision_rows = event_store.list_decision_events(limit=max(1, int(limit)))
        for row in decision_rows:
            decision_count += 1
            decision_ts = _parse_ts(row.get("decision_ts"))
            if cutoff is not None and (decision_ts is None or decision_ts < cutoff):
                continue
            decision_inserted_count += 1
    finally:
        event_store.close()

    sorted_slippage = sorted(slippage_samples)
    p90_slippage = 0.0
    if sorted_slippage:
        index = int(math.ceil(len(sorted_slippage) * 0.90)) - 1
        index = max(0, min(index, len(sorted_slippage) - 1))
        p90_slippage = float(sorted_slippage[index])
    reject_rate = (
        (float(submit_reject_count) / float(submit_ack_count + submit_reject_count)) * 100.0
        if (submit_ack_count + submit_reject_count) > 0
        else 0.0
    )
    return {
        "scanned_oms_events": int(scanned),
        "filled_events": int(fills_count),
        "submit_ack_events": int(submit_ack_count),
        "submit_reject_events": int(submit_reject_count),
        "submit_reject_rate_pct": float(reject_rate),
        "fill_notional": float(fills_notional),
        "avg_fill_notional": (
            float(fills_notional / fills_count) if fills_count > 0 else 0.0
        ),
        "slippage_sample_count": int(len(slippage_samples)),
        "median_slippage_bps": float(median(sorted_slippage)) if sorted_slippage else 0.0,
        "p90_slippage_bps": float(p90_slippage),
        "decision_events_scanned": int(decision_count),
        "decision_events_in_window": int(decision_inserted_count),
        "lookback_days": int(lookback_days) if lookback_days is not None else None,
    }


__all__ = ["summarize_oms_event_tca"]
