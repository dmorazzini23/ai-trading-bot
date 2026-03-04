"""Multi-factor quality/value cross-sectional strategy sleeve."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.enums import RiskLevel
from .base import BaseStrategy, StrategySignal


def _zscore(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(list(values.values()), dtype=float)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr))
    if sigma <= 1e-9:
        return {k: 0.0 for k in values}
    return {k: float((v - mu) / sigma) for k, v in values.items()}


class MultiFactorQualityValueStrategy(BaseStrategy):
    """Ranks symbols by quality/value proxy factors and trades both tails."""

    def __init__(
        self,
        strategy_id: str = "multi_factor_quality_value",
        name: str = "Multi Factor Quality Value",
        risk_level: RiskLevel = RiskLevel.MODERATE,
        *,
        lookback: int = 63,
        top_quantile: float = 0.2,
        min_universe: int = 8,
    ) -> None:
        super().__init__(strategy_id, name, risk_level)
        self.lookback = max(30, int(lookback))
        self.top_quantile = float(min(max(top_quantile, 0.05), 0.45))
        self.min_universe = max(6, int(min_universe))

    def _load_daily_closes(self, ctx: Any) -> dict[str, Any]:
        fetcher = getattr(ctx, "data_fetcher", None)
        symbols = [str(s).strip().upper() for s in (getattr(ctx, "tickers", []) or []) if str(s).strip()]
        if fetcher is None or not symbols:
            return {}
        closes: dict[str, Any] = {}
        for symbol in symbols:
            try:
                df = fetcher.get_daily_df(ctx, symbol)
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            if df is None or getattr(df, "empty", True) or "close" not in getattr(df, "columns", []):
                continue
            series = df["close"].dropna()
            if len(series) <= self.lookback:
                continue
            closes[symbol] = series
        return closes

    def generate_signals(self, market_data: dict[str, Any]) -> list[StrategySignal]:
        closes = market_data.get("closes", {}) if isinstance(market_data, dict) else {}
        if not isinstance(closes, dict):
            return []

        momentum: dict[str, float] = {}
        stability: dict[str, float] = {}
        drawdown_inverse: dict[str, float] = {}
        for symbol, series in closes.items():
            try:
                recent = series.iloc[-self.lookback :]
                first_px = float(recent.iloc[0])
                last_px = float(recent.iloc[-1])
                if first_px <= 0:
                    continue
                ret = (last_px / first_px) - 1.0
                daily_ret = recent.pct_change().dropna()
                vol = float(np.std(np.asarray(daily_ret.values, dtype=float)))
                rolling_max = np.maximum.accumulate(np.asarray(recent.values, dtype=float))
                drawdowns = (np.asarray(recent.values, dtype=float) / np.maximum(rolling_max, 1e-9)) - 1.0
                max_dd = abs(float(np.min(drawdowns)))
            except (AttributeError, KeyError, IndexError, TypeError, ValueError):
                continue
            if not np.isfinite(ret):
                continue
            momentum[symbol] = float(ret)
            stability[symbol] = float(-vol) if np.isfinite(vol) else -1.0
            drawdown_inverse[symbol] = float(-max_dd) if np.isfinite(max_dd) else -1.0

        if len(momentum) < self.min_universe:
            return []

        z_mom = _zscore(momentum)
        z_stability = _zscore(stability)
        z_drawdown = _zscore(drawdown_inverse)
        composite: dict[str, float] = {}
        for symbol in momentum:
            composite[symbol] = (
                0.45 * float(z_mom.get(symbol, 0.0))
                + 0.30 * float(z_stability.get(symbol, 0.0))
                + 0.25 * float(z_drawdown.get(symbol, 0.0))
            )

        ranked = sorted(composite.items(), key=lambda kv: kv[1])
        bucket = max(1, int(round(len(ranked) * self.top_quantile)))
        losers = [sym for sym, _ in ranked[:bucket]]
        winners = [sym for sym, _ in ranked[-bucket:]]

        signals: list[StrategySignal] = []
        for symbol in winners:
            score = float(max(0.0, composite.get(symbol, 0.0)))
            strength = float(min(1.0, max(0.05, score / 3.0)))
            edge_bps = float(max(2.0, score * 12.0))
            signals.append(
                StrategySignal(
                    symbol=symbol,
                    side="buy",
                    strength=strength,
                    confidence=float(min(0.95, 0.55 + (0.25 * strength))),
                    strategy_id=self.strategy_id,
                    signal_type="multi_factor_quality_value",
                    expected_return=float(max(0.0, momentum.get(symbol, 0.0))),
                    metadata={
                        "expected_edge_bps": edge_bps,
                        "sleeve": self.strategy_id,
                        "factor_composite": float(composite.get(symbol, 0.0)),
                    },
                )
            )
        for symbol in losers:
            score = float(max(0.0, -composite.get(symbol, 0.0)))
            strength = float(min(1.0, max(0.05, score / 3.0)))
            edge_bps = float(max(2.0, score * 12.0))
            signals.append(
                StrategySignal(
                    symbol=symbol,
                    side="sell",
                    strength=strength,
                    confidence=float(min(0.95, 0.55 + (0.25 * strength))),
                    strategy_id=self.strategy_id,
                    signal_type="multi_factor_quality_value",
                    expected_return=float(max(0.0, -momentum.get(symbol, 0.0))),
                    metadata={
                        "expected_edge_bps": edge_bps,
                        "sleeve": self.strategy_id,
                        "factor_composite": float(composite.get(symbol, 0.0)),
                    },
                )
            )
        return signals

    def generate(self, ctx: Any) -> list[StrategySignal]:
        return self.generate_signals({"closes": self._load_daily_closes(ctx)})
