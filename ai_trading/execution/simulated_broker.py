"""Deterministic simulated broker with asynchronous fill events."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import math
import random
from typing import Any, Mapping


def _to_utc(raw: datetime | str | None) -> datetime:
    if raw is None:
        return datetime.now(UTC)
    if isinstance(raw, datetime):
        if raw.tzinfo is None:
            return raw.replace(tzinfo=UTC)
        return raw.astimezone(UTC)
    text = str(raw)
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@dataclass(slots=True)
class _ScheduledFill:
    due_at: datetime
    order_id: str
    fill_ratio: float


class SimulatedBroker:
    """Simple deterministic broker simulator for replay and parity tests."""

    def __init__(
        self,
        *,
        seed: int = 42,
        fill_probability: float = 0.95,
        partial_fill_probability: float = 0.35,
    ) -> None:
        self._rng = random.Random(int(seed))
        self._fill_probability = max(0.0, min(1.0, float(fill_probability)))
        self._partial_fill_probability = max(
            0.0, min(1.0, float(partial_fill_probability))
        )
        self._counter = 0
        self._orders: dict[str, dict[str, Any]] = {}
        self._events: deque[dict[str, Any]] = deque()
        self._scheduled: list[_ScheduledFill] = []

    def submit_order(
        self,
        order: Mapping[str, Any],
        *,
        timestamp: datetime | str | None = None,
        spread_bps: float = 8.0,
        volatility_pct: float = 0.01,
    ) -> dict[str, Any]:
        """Submit order and schedule async fill event(s)."""

        self._counter += 1
        now = _to_utc(timestamp)
        order_id = f"sim-{self._counter:08d}"
        qty = float(order.get("qty", 0.0) or 0.0)
        symbol = str(order.get("symbol", "")).upper()
        side = str(order.get("side", "buy")).lower()
        order_type = str(order.get("type", order.get("order_type", "limit"))).lower()
        price = order.get("limit_price", order.get("price", None))
        limit_price = float(price) if price is not None else None
        client_order_id = str(order.get("client_order_id", "") or order_id)
        model_order: dict[str, Any] = {
            "id": order_id,
            "client_order_id": client_order_id,
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "qty": qty,
            "filled_qty": 0.0,
            "filled_avg_price": None,
            "limit_price": limit_price,
            "status": "accepted",
            "submitted_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "spread_bps": float(spread_bps),
            "volatility_pct": float(volatility_pct),
        }
        self._orders[order_id] = model_order

        base_delay_ms = 150 + int(abs(spread_bps) * 20.0)
        vol_delay_ms = int(max(0.0, volatility_pct) * 10_000)
        jitter_ms = self._rng.randint(0, 750)
        delay_ms = base_delay_ms + vol_delay_ms + jitter_ms
        due_at = now + timedelta(milliseconds=delay_ms)

        if self._rng.random() > self._fill_probability:
            # No fill scheduled; order stays accepted/open.
            return dict(model_order)

        fill_ratio = 1.0
        if self._rng.random() < self._partial_fill_probability:
            fill_ratio = min(0.95, max(0.1, self._rng.uniform(0.25, 0.75)))
        self._scheduled.append(
            _ScheduledFill(due_at=due_at, order_id=order_id, fill_ratio=fill_ratio)
        )
        self._scheduled.sort(key=lambda item: item.due_at)
        return dict(model_order)

    def cancel_order(self, order_id: str, *, timestamp: datetime | str | None = None) -> bool:
        """Cancel an open order."""

        model_order = self._orders.get(order_id)
        if model_order is None:
            return False
        status = str(model_order.get("status", ""))
        if status in {"filled", "canceled", "rejected"}:
            return False
        now = _to_utc(timestamp)
        model_order["status"] = "canceled"
        model_order["updated_at"] = now.isoformat()
        self._scheduled = [item for item in self._scheduled if item.order_id != order_id]
        self._events.append(
            {
                "event_type": "order_canceled",
                "order_id": order_id,
                "symbol": model_order.get("symbol"),
                "ts": now.isoformat(),
                "status": "canceled",
            }
        )
        return True

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        """Return current order snapshot."""

        payload = self._orders.get(order_id)
        if payload is None:
            return None
        return dict(payload)

    def list_orders(self, *, status: str | None = None) -> list[dict[str, Any]]:
        """Return order snapshots optionally filtered by status."""

        if status is None:
            return [dict(item) for item in self._orders.values()]
        token = status.strip().lower()
        return [
            dict(item)
            for item in self._orders.values()
            if str(item.get("status", "")).lower() == token
        ]

    def process_until(
        self,
        *,
        now: datetime | str | None,
        market_price_by_symbol: Mapping[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Emit all async events with due time <= ``now``."""

        current = _to_utc(now)
        drained: list[dict[str, Any]] = []
        ready: list[_ScheduledFill] = []
        while self._scheduled and self._scheduled[0].due_at <= current:
            ready.append(self._scheduled.pop(0))
        for item in ready:
            model_order = self._orders.get(item.order_id)
            if model_order is None:
                continue
            if str(model_order.get("status", "")).lower() in {
                "canceled",
                "filled",
                "rejected",
            }:
                continue
            symbol = str(model_order.get("symbol", "")).upper()
            market_px = None
            if market_price_by_symbol is not None:
                raw_market = market_price_by_symbol.get(symbol)
                if raw_market is not None:
                    market_px = float(raw_market)
            fill_price = self._resolve_fill_price(model_order, market_px)
            qty = float(model_order.get("qty", 0.0) or 0.0)
            filled_qty = float(model_order.get("filled_qty", 0.0) or 0.0)
            remaining = max(0.0, qty - filled_qty)
            if remaining <= 0:
                model_order["status"] = "filled"
                continue
            fill_qty = min(remaining, max(1.0, round(qty * item.fill_ratio, 6)))
            new_filled = min(qty, filled_qty + fill_qty)
            avg = model_order.get("filled_avg_price")
            if avg is None:
                avg_price = fill_price
            else:
                previous_qty = max(0.0, filled_qty)
                total_value = (previous_qty * float(avg)) + (fill_qty * fill_price)
                avg_price = total_value / max(1e-9, previous_qty + fill_qty)
            model_order["filled_qty"] = new_filled
            model_order["filled_avg_price"] = avg_price
            model_order["updated_at"] = item.due_at.isoformat()
            if new_filled >= qty - 1e-9:
                model_order["status"] = "filled"
            else:
                model_order["status"] = "partially_filled"
            event = {
                "event_type": "fill",
                "order_id": model_order["id"],
                "client_order_id": model_order.get("client_order_id"),
                "symbol": symbol,
                "side": model_order.get("side"),
                "fill_qty": fill_qty,
                "fill_price": fill_price,
                "filled_qty": new_filled,
                "status": model_order["status"],
                "ts": item.due_at.isoformat(),
            }
            self._events.append(event)
            drained.append(event)
        while self._events:
            drained.append(self._events.popleft())
        return drained

    def _resolve_fill_price(
        self,
        order: Mapping[str, Any],
        market_price: float | None,
    ) -> float:
        limit_price = order.get("limit_price")
        side = str(order.get("side", "buy")).lower()
        base = None
        if market_price is not None and math.isfinite(market_price) and market_price > 0:
            base = market_price
        if base is None and limit_price is not None:
            try:
                parsed_limit = float(limit_price)
            except (TypeError, ValueError):
                parsed_limit = None
            if parsed_limit is not None and math.isfinite(parsed_limit) and parsed_limit > 0:
                base = parsed_limit
        if base is None:
            base = 100.0
        spread_bps = float(order.get("spread_bps", 8.0) or 8.0)
        spread_component = base * (spread_bps / 10_000.0)
        vol_pct = float(order.get("volatility_pct", 0.01) or 0.01)
        vol_component = base * max(0.0, vol_pct) * self._rng.uniform(0.0, 0.35)
        jitter = base * self._rng.uniform(-0.0008, 0.0008)
        if side == "buy":
            return float(base + spread_component * 0.5 + vol_component + jitter)
        return float(base - spread_component * 0.5 - vol_component + jitter)

