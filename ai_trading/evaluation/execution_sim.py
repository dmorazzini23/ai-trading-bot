"""Executed-trade simulation utilities for model evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(slots=True)
class ExecutionSimConfig:
    """Configuration for fold-level executed-trade simulation."""

    signal_threshold: float = 0.0
    transaction_cost_bps: float = 0.0
    slippage_bps: float = 0.0
    allow_short: bool = True
    max_abs_position: float = 1.0

    @classmethod
    def from_mapping(
        cls,
        params: Mapping[str, Any] | None,
    ) -> "ExecutionSimConfig":
        source = params or {}
        try:
            signal_threshold = max(0.0, float(source.get("signal_threshold", 0.0) or 0.0))
        except (TypeError, ValueError):
            signal_threshold = 0.0
        try:
            transaction_cost_bps = max(
                0.0, float(source.get("transaction_cost_bps", 0.0) or 0.0)
            )
        except (TypeError, ValueError):
            transaction_cost_bps = 0.0
        try:
            slippage_bps = max(0.0, float(source.get("slippage_bps", 0.0) or 0.0))
        except (TypeError, ValueError):
            slippage_bps = 0.0
        allow_short = bool(source.get("allow_short", True))
        try:
            max_abs_position = max(0.0, float(source.get("max_abs_position", 1.0) or 0.0))
        except (TypeError, ValueError):
            max_abs_position = 1.0
        return cls(
            signal_threshold=signal_threshold,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            allow_short=allow_short,
            max_abs_position=max_abs_position,
        )


def _empty_metrics() -> dict[str, float]:
    return {
        "gross_return": 0.0,
        "net_return": 0.0,
        "cost_return": 0.0,
        "turnover_units": 0.0,
        "trade_count": 0.0,
        "signal_count": 0.0,
        "max_drawdown": 0.0,
        "hit_rate": 0.0,
    }


def simulate_executed_trades(
    *,
    y_true: Sequence[float],
    y_pred: Sequence[float],
    params: Mapping[str, Any] | ExecutionSimConfig | None = None,
) -> dict[str, float]:
    """Simulate realized fold-level PnL from predictions and returns."""

    cfg = params if isinstance(params, ExecutionSimConfig) else ExecutionSimConfig.from_mapping(params)
    true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)
    steps = int(min(true_arr.size, pred_arr.size))
    if steps <= 0:
        return _empty_metrics()

    total_cost_rate = (cfg.transaction_cost_bps + cfg.slippage_bps) / 10_000.0
    prev_position = 0.0
    equity = 1.0
    running_peak = equity
    max_drawdown = 0.0
    gross_return = 0.0
    cost_return = 0.0
    turnover_units = 0.0
    trade_count = 0
    active_signals = 0
    profitable_steps = 0

    for pred, actual in zip(pred_arr[:steps], true_arr[:steps], strict=False):
        if pred > cfg.signal_threshold:
            target_position = cfg.max_abs_position
        elif pred < -cfg.signal_threshold and cfg.allow_short:
            target_position = -cfg.max_abs_position
        else:
            target_position = 0.0

        turnover = abs(target_position - prev_position)
        if turnover > 0.0:
            trade_count += 1
        if target_position != 0.0:
            active_signals += 1

        step_gross = float(target_position * float(actual))
        step_cost = float(turnover * total_cost_rate)
        step_net = max(step_gross - step_cost, -0.99)

        gross_return += step_gross
        cost_return += step_cost
        turnover_units += turnover

        equity *= 1.0 + step_net
        running_peak = max(running_peak, equity)
        drawdown = 0.0 if running_peak <= 0.0 else (running_peak - equity) / running_peak
        max_drawdown = max(max_drawdown, drawdown)
        if step_net > 0.0:
            profitable_steps += 1

        prev_position = target_position

    return {
        "gross_return": float(gross_return),
        "net_return": float(equity - 1.0),
        "cost_return": float(cost_return),
        "turnover_units": float(turnover_units),
        "trade_count": float(trade_count),
        "signal_count": float(active_signals),
        "max_drawdown": float(max_drawdown),
        "hit_rate": float(profitable_steps / max(1, active_signals)),
    }


__all__ = [
    "ExecutionSimConfig",
    "simulate_executed_trades",
]
