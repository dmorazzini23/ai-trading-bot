"""Time-series momentum overlay strategy sleeve."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.enums import RiskLevel
from .base import BaseStrategy, StrategySignal


class TimeSeriesMomentumOverlayStrategy(BaseStrategy):
    """Trend-following overlay on each symbol's own return history."""

    def __init__(
        self,
        strategy_id: str = "time_series_momentum_overlay",
        name: str = "TS Momentum Overlay",
        risk_level: RiskLevel = RiskLevel.MODERATE,
        *,
        fast_lookback: int = 20,
        slow_lookback: int = 120,
        trend_floor: float = 0.01,
    ) -> None:
        super().__init__(strategy_id, name, risk_level)
        self.fast_lookback = max(10, int(fast_lookback))
        self.slow_lookback = max(self.fast_lookback + 10, int(slow_lookback))
        self.trend_floor = float(max(0.001, min(trend_floor, 0.15)))

    def _load_closes(self, ctx: Any) -> dict[str, Any]:
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
            if len(series) <= self.slow_lookback + 2:
                continue
            closes[symbol] = series
        return closes

    def generate_signals(self, market_data: dict[str, Any]) -> list[StrategySignal]:
        closes = market_data.get("closes", {}) if isinstance(market_data, dict) else {}
        if not isinstance(closes, dict):
            return []
        signals: list[StrategySignal] = []
        for symbol, series in closes.items():
            try:
                now_px = float(series.iloc[-1])
                fast_px = float(series.iloc[-1 - self.fast_lookback])
                slow_px = float(series.iloc[-1 - self.slow_lookback])
            except (AttributeError, KeyError, IndexError, TypeError, ValueError):
                continue
            if now_px <= 0 or fast_px <= 0 or slow_px <= 0:
                continue
            fast_ret = (now_px / fast_px) - 1.0
            slow_ret = (now_px / slow_px) - 1.0
            trend = 0.6 * fast_ret + 0.4 * slow_ret
            abs_trend = abs(float(trend))
            if abs_trend < self.trend_floor:
                continue
            side = "buy" if trend > 0 else "sell"
            strength = float(min(1.0, max(0.05, abs_trend / (self.trend_floor * 6.0))))
            confidence = float(min(0.95, 0.55 + (0.30 * strength)))
            trend_vol = float(np.std(np.asarray(series.pct_change().dropna().tail(40).values, dtype=float)))
            edge_bps = float(max(2.0, abs_trend * 10000.0 * max(0.2, 1.0 - (trend_vol * 20.0))))
            signals.append(
                StrategySignal(
                    symbol=symbol,
                    side=side,
                    strength=strength,
                    confidence=confidence,
                    strategy_id=self.strategy_id,
                    signal_type="time_series_momentum_overlay",
                    expected_return=float(abs_trend),
                    metadata={
                        "expected_edge_bps": edge_bps,
                        "sleeve": self.strategy_id,
                        "fast_return": float(fast_ret),
                        "slow_return": float(slow_ret),
                    },
                )
            )
        return signals

    def generate(self, ctx: Any) -> list[StrategySignal]:
        return self.generate_signals({"closes": self._load_closes(ctx)})
