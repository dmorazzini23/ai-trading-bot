"""Broker reconciliation helpers."""
from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

from dataclasses import dataclass, field
from typing import Any
import math
from alpaca.common.exceptions import APIError

from ai_trading.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ReconcileResult:
    ok: bool
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def fetch_broker_positions(api: Any) -> dict[str, float]:
    positions: dict[str, float] = {}
    if api is None:
        return positions
    try:
        get_all_positions = getattr(api, "get_all_positions", None)
        list_positions = getattr(api, "list_positions", None)
        if callable(get_all_positions):
            raw_positions = get_all_positions()
        elif callable(list_positions):
            raw_positions = list_positions()
        else:
            return positions
        if raw_positions is None:
            raise ValueError("broker positions unavailable")
        for pos in raw_positions:
            symbol = pos.get("symbol") if isinstance(pos, dict) else getattr(pos, "symbol", None)
            qty = pos.get("qty", pos.get("quantity")) if isinstance(pos, dict) else getattr(pos, "qty", getattr(pos, "quantity", None))
            side = pos.get("side") if isinstance(pos, dict) else getattr(pos, "side", None)
            side_token = str(getattr(side, "value", side) or "").strip().lower()
            if not symbol or str(symbol) in positions or qty is None:
                raise ValueError("invalid broker position snapshot")
            if symbol:
                qty_float = float(qty)
                if not math.isfinite(qty_float):
                    raise ValueError("nonfinite broker quantity")
                if qty_float > 0 and side_token in {"short", "sell_short", "sell-short", "sell short"}:
                    qty_float = -qty_float
                positions[str(symbol)] = qty_float
    except APIError:
        raise  # Preserve SDK status for dependency retry/auth classification.
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
        logger.warning("BROKER_POSITIONS_FETCH_FAILED", extra={"error": str(exc)})
        raise RuntimeError(f"broker_positions_fetch_failed: {exc}") from exc
    return positions


def reconcile(
    internal_positions: dict[str, float],
    broker_positions: dict[str, float],
    *,
    tolerance_shares: float = 0.0,
) -> ReconcileResult:
    if not math.isfinite(tolerance_shares) or tolerance_shares < 0:
        raise ValueError("invalid reconciliation tolerance")
    mismatches: list[dict[str, Any]] = []
    all_symbols = set(internal_positions) | set(broker_positions)
    for symbol in sorted(all_symbols):
        internal_qty = float(internal_positions.get(symbol, 0.0))
        broker_qty = float(broker_positions.get(symbol, 0.0))
        if not math.isfinite(internal_qty) or not math.isfinite(broker_qty):
            raise ValueError("nonfinite reconciliation quantity")
        delta = broker_qty - internal_qty
        if abs(delta) > tolerance_shares:
            mismatches.append(
                {
                    "symbol": symbol,
                    "internal_qty": internal_qty,
                    "broker_qty": broker_qty,
                    "delta": delta,
                }
            )
    return ReconcileResult(
        ok=not mismatches,
        mismatches=mismatches,
        summary={
            "symbols_checked": len(all_symbols),
            "mismatch_count": len(mismatches),
        },
    )
