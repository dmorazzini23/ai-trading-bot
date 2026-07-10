"""Executed-trade simulation utilities for model evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ai_trading.replay.live_cost_alignment import resolve_live_cost_alignment

_MIN_CONTEXT_FALLBACK_COST_BPS = 1.0


def _finite_float(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if np.isfinite(numeric) else default


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return bool(default)


@dataclass(slots=True)
class ExecutionSimConfig:
    """Configuration for fold-level executed-trade simulation."""

    signal_threshold: float = 0.0
    transaction_cost_bps: float = 1.0
    slippage_bps: float = 5.0
    allow_short: bool = True
    max_abs_position: float = 1.0

    def __post_init__(self) -> None:
        self.signal_threshold = max(0.0, _finite_float(self.signal_threshold, 0.0))
        self.transaction_cost_bps = max(0.0, _finite_float(self.transaction_cost_bps, 1.0))
        self.slippage_bps = max(0.0, _finite_float(self.slippage_bps, 5.0))
        self.max_abs_position = max(0.0, _finite_float(self.max_abs_position, 1.0))

    @classmethod
    def from_mapping(
        cls,
        params: Mapping[str, Any] | None,
    ) -> "ExecutionSimConfig":
        source = params or {}
        signal_threshold = max(
            0.0, _finite_float(source.get("signal_threshold", 0.0) or 0.0, 0.0)
        )
        transaction_cost_bps = max(
            0.0, _finite_float(source.get("transaction_cost_bps", 1.0) or 0.0, 1.0)
        )
        slippage_bps = max(
            0.0, _finite_float(source.get("slippage_bps", 5.0) or 0.0, 5.0)
        )
        allow_short = _as_bool(source.get("allow_short", True), True)
        max_abs_position = max(
            0.0, _finite_float(source.get("max_abs_position", 1.0) or 0.0, 1.0)
        )
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
        "mean_applied_cost_bps": 0.0,
        "max_applied_cost_bps": 0.0,
        "cost_source_fixed_count": 0.0,
        "cost_source_live_count": 0.0,
        "cost_source_fallback_count": 0.0,
        "cost_alignment_stale_count": 0.0,
        "cost_alignment_insufficient_count": 0.0,
    }


def _execution_context_at(
    execution_context: Sequence[Mapping[str, Any]] | None,
    index: int,
) -> Mapping[str, Any] | None:
    if execution_context is None or index >= len(execution_context):
        return None
    row = execution_context[index]
    return row if isinstance(row, Mapping) else None


def _context_text(context: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = context.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _resolve_step_cost_bps(
    *,
    fixed_cost_bps: float,
    context: Mapping[str, Any] | None,
    live_cost_model: Mapping[str, Any] | None,
    side: str,
    cost_alignment_params: Mapping[str, Any] | None,
) -> tuple[float, str, str]:
    if context is None:
        return float(fixed_cost_bps), "fixed", "fixed"
    requested_fallback = _finite_float(
        context.get("fallback_cost_bps"),
        fixed_cost_bps,
    )
    fallback = max(
        float(fixed_cost_bps),
        float(requested_fallback),
        _MIN_CONTEXT_FALLBACK_COST_BPS,
    )
    alignment_params = cost_alignment_params or {}
    resolution = resolve_live_cost_alignment(
        live_cost_model,
        symbol=_context_text(context, "symbol", "ticker"),
        side=_context_text(context, "side", "order_side") or side,
        session_bucket=_context_text(
            context,
            "session_bucket",
            "session_regime",
            "session",
        ),
        order_type=_context_text(context, "order_type", "type"),
        volatility_bucket=_context_text(
            context,
            "volatility_bucket",
            "vol_bucket",
            "liquidity_bucket",
        ),
        fallback_cost_bps=fallback,
        max_age_seconds=max(
            0.0,
            _finite_float(alignment_params.get("max_age_seconds"), 86_400.0),
        ),
        min_samples=max(
            1,
            int(_finite_float(alignment_params.get("min_samples"), 5.0)),
        ),
        cost_metric=str(
            alignment_params.get("cost_metric") or "p90_total_cost_bps"
        ),
    )
    resolved = max(
        fallback,
        _finite_float(resolution.get("resolved_cost_bps"), fallback),
    )
    return (
        float(resolved),
        str(resolution.get("source") or "fallback"),
        str(resolution.get("alignment") or "unknown"),
    )


def simulate_executed_trades(
    *,
    y_true: Sequence[float],
    y_pred: Sequence[float],
    params: Mapping[str, Any] | ExecutionSimConfig | None = None,
    execution_context: Sequence[Mapping[str, Any]] | None = None,
    live_cost_model: Mapping[str, Any] | None = None,
    cost_alignment_params: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Simulate realized fold-level PnL from predictions and returns."""

    if isinstance(params, ExecutionSimConfig):
        cfg = ExecutionSimConfig(
            signal_threshold=params.signal_threshold,
            transaction_cost_bps=params.transaction_cost_bps,
            slippage_bps=params.slippage_bps,
            allow_short=params.allow_short,
            max_abs_position=params.max_abs_position,
        )
    else:
        cfg = ExecutionSimConfig.from_mapping(params)
    true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)
    steps = int(min(true_arr.size, pred_arr.size))
    if steps <= 0:
        return _empty_metrics()

    fixed_cost_bps = cfg.transaction_cost_bps + cfg.slippage_bps
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
    applied_cost_bps: list[float] = []
    source_counts = {"fixed": 0, "live": 0, "fallback": 0}
    stale_count = 0
    insufficient_count = 0

    for step_index, (pred, actual) in enumerate(
        zip(pred_arr[:steps], true_arr[:steps], strict=False)
    ):
        if not np.isfinite(actual):
            target_position = 0.0
            turnover = abs(target_position - prev_position)
            step_cost = 0.0
            if turnover > 0.0:
                side = "buy" if target_position > prev_position else "sell"
                cost_bps, source, alignment = _resolve_step_cost_bps(
                    fixed_cost_bps=fixed_cost_bps,
                    context=_execution_context_at(execution_context, step_index),
                    live_cost_model=live_cost_model,
                    side=side,
                    cost_alignment_params=cost_alignment_params,
                )
                step_cost = float(turnover * (cost_bps / 10_000.0))
                applied_cost_bps.append(cost_bps)
                source_counts[source if source in source_counts else "fallback"] += 1
                stale_count += int(alignment == "stale")
                insufficient_count += int(alignment == "insufficient_samples")
            if turnover > 0.0:
                trade_count += 1
            cost_return += step_cost
            turnover_units += turnover
            equity *= 1.0 + max(-step_cost, -0.99)
            running_peak = max(running_peak, equity)
            drawdown = 0.0 if running_peak <= 0.0 else (running_peak - equity) / running_peak
            max_drawdown = max(max_drawdown, drawdown)
            prev_position = target_position
            continue
        if not np.isfinite(pred):
            pred = 0.0
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
        step_cost = 0.0
        if turnover > 0.0:
            side = "buy" if target_position > prev_position else "sell"
            cost_bps, source, alignment = _resolve_step_cost_bps(
                fixed_cost_bps=fixed_cost_bps,
                context=_execution_context_at(execution_context, step_index),
                live_cost_model=live_cost_model,
                side=side,
                cost_alignment_params=cost_alignment_params,
            )
            step_cost = float(turnover * (cost_bps / 10_000.0))
            applied_cost_bps.append(cost_bps)
            source_counts[source if source in source_counts else "fallback"] += 1
            stale_count += int(alignment == "stale")
            insufficient_count += int(alignment == "insufficient_samples")
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
        "mean_applied_cost_bps": float(np.mean(applied_cost_bps))
        if applied_cost_bps
        else 0.0,
        "max_applied_cost_bps": float(max(applied_cost_bps))
        if applied_cost_bps
        else 0.0,
        "cost_source_fixed_count": float(source_counts["fixed"]),
        "cost_source_live_count": float(source_counts["live"]),
        "cost_source_fallback_count": float(source_counts["fallback"]),
        "cost_alignment_stale_count": float(stale_count),
        "cost_alignment_insufficient_count": float(insufficient_count),
    }


__all__ = [
    "ExecutionSimConfig",
    "simulate_executed_trades",
]
