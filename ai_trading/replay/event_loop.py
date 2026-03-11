"""Event-driven replay loop with async simulated broker events."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Iterable, Mapping

from ai_trading.config.management import get_env
from ai_trading.execution.simulated_broker import SimulatedBroker


def _to_utc(raw: Any) -> datetime:
    text = str(raw)
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class ReplayInvariantViolation:
    code: str
    message: str
    ts: str
    details: dict[str, Any] = field(default_factory=dict)


class ReplayEventLoop:
    """Replay bars while processing asynchronous order events between bars."""

    def __init__(
        self,
        *,
        strategy: Callable[[Mapping[str, Any]], Mapping[str, Any] | None],
        broker: SimulatedBroker | None = None,
        seed: int | None = None,
        max_symbol_notional: float | None = None,
        max_gross_notional: float | None = None,
        initial_positions: Mapping[str, float] | None = None,
        clip_intents_to_caps: bool | None = None,
    ) -> None:
        replay_seed = int(seed if seed is not None else get_env("REPLAY_SEED", "42", cast=int))
        self.strategy = strategy
        if broker is None:
            fill_probability = float(
                get_env("AI_TRADING_REPLAY_FILL_PROBABILITY", 0.95, cast=float)
            )
            partial_fill_probability = float(
                get_env("AI_TRADING_REPLAY_PARTIAL_FILL_PROBABILITY", 0.35, cast=float)
            )
            min_fill_delay_ms = int(
                get_env("AI_TRADING_REPLAY_FILL_MIN_DELAY_MS", 150, cast=int)
            )
            max_fill_delay_ms = int(
                get_env("AI_TRADING_REPLAY_FILL_MAX_DELAY_MS", 2500, cast=int)
            )
            cancel_reject_probability = float(
                get_env("AI_TRADING_REPLAY_CANCEL_REJECT_PROBABILITY", 0.0, cast=float)
            )
            broker = SimulatedBroker(
                seed=replay_seed,
                fill_probability=fill_probability,
                partial_fill_probability=partial_fill_probability,
                min_fill_delay_ms=min_fill_delay_ms,
                max_fill_delay_ms=max_fill_delay_ms,
                cancel_reject_probability=cancel_reject_probability,
            )
        self.broker = broker
        self.seed = replay_seed
        self.max_symbol_notional = float(
            max_symbol_notional
            if max_symbol_notional is not None
            else get_env("GLOBAL_MAX_SYMBOL_DOLLARS", "25000", cast=float)
        )
        self.max_gross_notional = float(
            max_gross_notional
            if max_gross_notional is not None
            else get_env("GLOBAL_MAX_GROSS_DOLLARS", "150000", cast=float)
        )
        if clip_intents_to_caps is None:
            clip_intents_to_caps = bool(
                get_env("AI_TRADING_REPLAY_CLIP_INTENTS_TO_CAPS", True, cast=bool)
            )
        self.clip_intents_to_caps = bool(clip_intents_to_caps)
        self._seen_intent_keys: set[str] = set()
        self._positions: dict[str, float] = {}
        if isinstance(initial_positions, Mapping):
            for raw_symbol, raw_qty in initial_positions.items():
                symbol = str(raw_symbol or "").strip().upper()
                if not symbol:
                    continue
                try:
                    qty = float(raw_qty or 0.0)
                except (TypeError, ValueError):
                    continue
                if abs(qty) <= 1e-9:
                    continue
                self._positions[symbol] = qty
        self._last_price_by_symbol: dict[str, float] = {}

    def run(self, bars: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        """Execute replay loop and enforce parity invariants."""

        ordered = sorted((dict(bar) for bar in bars), key=lambda row: _to_utc(row.get("ts")))
        intents: list[dict[str, Any]] = []
        orders: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        cap_adjustments: list[dict[str, Any]] = []
        violations: list[ReplayInvariantViolation] = []

        for bar in ordered:
            ts = _to_utc(bar.get("ts"))
            symbol = str(bar.get("symbol", "")).upper()
            close = float(bar.get("close", 0.0) or 0.0)
            if close <= 0:
                continue
            self._last_price_by_symbol[symbol] = close

            fill_events = self.broker.process_until(
                now=ts,
                market_price_by_symbol={symbol: close},
            )
            for event in fill_events:
                events.append(event)
                if event.get("event_type") != "fill":
                    continue
                fill_symbol = str(event.get("symbol", "")).upper()
                fill_qty = float(event.get("fill_qty", 0.0) or 0.0)
                side = str(event.get("side", "buy")).lower()
                fill_price = float(event.get("fill_price", 0.0) or 0.0)
                if fill_symbol and fill_price > 0:
                    self._last_price_by_symbol[fill_symbol] = fill_price
                signed_qty = fill_qty if side == "buy" else -fill_qty
                self._positions[fill_symbol] = self._positions.get(fill_symbol, 0.0) + signed_qty
                if abs(self._positions[fill_symbol]) < 1e-9:
                    self._positions.pop(fill_symbol, None)

            proposal = self.strategy(bar)
            if proposal is None:
                continue

            side = str(proposal.get("side", "buy")).lower()
            qty = float(proposal.get("qty", 0.0) or 0.0)
            if qty <= 0:
                continue
            if self.clip_intents_to_caps:
                clipped_qty, adjustment = self._clip_qty_to_symbol_cap(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    close=close,
                )
                if adjustment is not None:
                    adjustment["ts"] = ts.isoformat()
                    cap_adjustments.append(adjustment)
                qty = clipped_qty
                if qty <= 0:
                    continue
            intent_key = str(
                proposal.get(
                    "intent_key",
                    f"{symbol}|{side}|{round(qty, 6)}|{ts.isoformat()}",
                )
            )
            if intent_key in self._seen_intent_keys:
                violations.append(
                    ReplayInvariantViolation(
                        code="duplicate_intent",
                        message=f"Duplicate intent key: {intent_key}",
                        ts=ts.isoformat(),
                    )
                )
                continue
            self._seen_intent_keys.add(intent_key)

            violation_details = self._notional_limit_violation(
                symbol=symbol,
                close=close,
                side=side,
                qty=qty,
            )
            if violation_details is not None:
                violations.append(
                    ReplayInvariantViolation(
                        code="position_cap_exceeded",
                        message=f"Position/gross cap exceeded for {symbol}",
                        ts=ts.isoformat(),
                        details=violation_details,
                    )
                )
                continue

            order_payload = {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "type": str(proposal.get("type", proposal.get("order_type", "limit"))),
                "price": proposal.get("price", close),
                "limit_price": proposal.get("limit_price", proposal.get("price", close)),
                "client_order_id": str(proposal.get("client_order_id", intent_key)),
            }
            intents.append(
                {
                    "ts": ts.isoformat(),
                    "symbol": symbol,
                    "intent_key": intent_key,
                    "side": side,
                    "qty": qty,
                }
            )
            order = self.broker.submit_order(order_payload, timestamp=ts)
            orders.append(order)

        if ordered:
            last_ts = _to_utc(ordered[-1].get("ts"))
            trailing_events = self.broker.process_until(
                now=last_ts + timedelta(minutes=5),
                market_price_by_symbol={},
            )
            for event in trailing_events:
                events.append(event)
                if event.get("event_type") != "fill":
                    continue
                fill_symbol = str(event.get("symbol", "")).upper()
                fill_qty = float(event.get("fill_qty", 0.0) or 0.0)
                side = str(event.get("side", "buy")).lower()
                fill_price = float(event.get("fill_price", 0.0) or 0.0)
                if fill_symbol and fill_price > 0:
                    self._last_price_by_symbol[fill_symbol] = fill_price
                signed_qty = fill_qty if side == "buy" else -fill_qty
                self._positions[fill_symbol] = self._positions.get(fill_symbol, 0.0) + signed_qty
                if abs(self._positions[fill_symbol]) < 1e-9:
                    self._positions.pop(fill_symbol, None)

        return {
            "seed": self.seed,
            "intents": intents,
            "orders": orders,
            "events": events,
            "violations": [
                {
                    "code": violation.code,
                    "message": violation.message,
                    "ts": violation.ts,
                    **dict(violation.details),
                }
                for violation in violations
            ],
            "cap_adjustments": cap_adjustments,
            "positions": dict(self._positions),
        }

    def _clip_qty_to_symbol_cap(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        close: float,
    ) -> tuple[float, dict[str, Any] | None]:
        if qty <= 0 or close <= 0 or self.max_symbol_notional <= 0:
            return max(0.0, float(qty)), None
        current_symbol_qty = float(self._positions.get(symbol, 0.0) or 0.0)
        signed_qty = float(qty) if side == "buy" else -float(qty)
        projected_symbol_qty = current_symbol_qty + signed_qty
        current_symbol_notional = abs(current_symbol_qty * close)
        projected_symbol_notional = abs(projected_symbol_qty * close)
        if projected_symbol_notional <= self.max_symbol_notional + 1e-9:
            return float(qty), None
        if projected_symbol_notional <= current_symbol_notional + 1e-9:
            return float(qty), None

        max_abs_qty = max(0.0, float(self.max_symbol_notional) / float(close))
        if abs(current_symbol_qty) > max_abs_qty + 1e-9:
            adjusted_qty = 0.0
        else:
            projected_sign = 1.0 if projected_symbol_qty > 0 else -1.0
            target_qty = projected_sign * max_abs_qty
            adjusted_signed_delta = target_qty - current_symbol_qty
            if side == "buy":
                adjusted_qty = max(0.0, min(float(qty), adjusted_signed_delta))
            else:
                adjusted_qty = max(0.0, min(float(qty), -adjusted_signed_delta))

        adjusted_signed = adjusted_qty if side == "buy" else -adjusted_qty
        adjusted_projected_qty = current_symbol_qty + adjusted_signed
        adjustment = {
            "symbol": symbol,
            "side": side,
            "requested_qty": float(qty),
            "adjusted_qty": float(adjusted_qty),
            "close": float(close),
            "max_symbol_notional": float(self.max_symbol_notional),
            "current_symbol_qty": float(current_symbol_qty),
            "current_symbol_notional": float(current_symbol_notional),
            "requested_projected_symbol_qty": float(projected_symbol_qty),
            "requested_projected_symbol_notional": float(projected_symbol_notional),
            "adjusted_projected_symbol_qty": float(adjusted_projected_qty),
            "adjusted_projected_symbol_notional": float(
                abs(adjusted_projected_qty * close)
            ),
        }
        return float(adjusted_qty), adjustment

    def _notional_limit_violation(
        self,
        *,
        symbol: str,
        close: float,
        side: str,
        qty: float,
    ) -> dict[str, Any] | None:
        signed_qty = qty if side == "buy" else -qty
        current_symbol_qty = self._positions.get(symbol, 0.0)
        projected_symbol_qty = current_symbol_qty + signed_qty
        current_symbol_notional = abs(current_symbol_qty * close)
        projected_symbol_notional = abs(projected_symbol_qty * close)
        if (
            projected_symbol_notional > (self.max_symbol_notional + 1e-9)
            and projected_symbol_notional > (current_symbol_notional + 1e-9)
        ):
            return {
                "dimension": "symbol",
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "close": close,
                "current_symbol_qty": current_symbol_qty,
                "projected_symbol_qty": projected_symbol_qty,
                "current_symbol_notional": current_symbol_notional,
                "projected_symbol_notional": projected_symbol_notional,
                "max_symbol_notional": self.max_symbol_notional,
            }

        gross_before = 0.0
        for sym, position_qty in self._positions.items():
            if sym == symbol:
                px = close
            else:
                px = self._last_price_by_symbol.get(sym, close)
            gross_before += abs(position_qty * px)
        gross_after = gross_before - current_symbol_notional + projected_symbol_notional
        if gross_after > (self.max_gross_notional + 1e-9) and gross_after > (gross_before + 1e-9):
            return {
                "dimension": "gross",
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "close": close,
                "current_symbol_qty": current_symbol_qty,
                "projected_symbol_qty": projected_symbol_qty,
                "gross_before": gross_before,
                "gross_after": gross_after,
                "max_gross_notional": self.max_gross_notional,
                "current_symbol_notional": current_symbol_notional,
                "projected_symbol_notional": projected_symbol_notional,
            }
        return None
