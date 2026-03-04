"""Cross-sectional momentum strategy sleeve."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.enums import RiskLevel
from .base import BaseStrategy, StrategySignal


def _zscore_map(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(list(values.values()), dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std <= 1e-9:
        return {k: 0.0 for k in values}
    return {k: float((v - mean) / std) for k, v in values.items()}


class CrossSectionalMomentumStrategy(BaseStrategy):
    """Ranks symbols by medium-horizon return and trades top/bottom buckets."""

    def __init__(
        self,
        strategy_id: str = "cross_sectional_momentum",
        name: str = "Cross Sectional Momentum",
        risk_level: RiskLevel = RiskLevel.MODERATE,
        *,
        lookback: int = 63,
        top_quantile: float = 0.2,
        min_universe: int = 6,
    ) -> None:
        super().__init__(strategy_id, name, risk_level)
        self.lookback = max(20, int(lookback))
        self.top_quantile = float(min(max(top_quantile, 0.05), 0.45))
        self.min_universe = max(4, int(min_universe))

    def _load_prices(self, ctx: Any) -> dict[str, Any]:
        fetcher = getattr(ctx, "data_fetcher", None)
        symbols = list(getattr(ctx, "tickers", []) or [])
        if fetcher is None or not symbols:
            return {}
        prices: dict[str, Any] = {}
        for symbol in symbols:
            try:
                df = fetcher.get_daily_df(ctx, symbol)
            except Exception:
                continue
            if df is None or getattr(df, "empty", True) or "close" not in getattr(df, "columns", []):
                continue
            series = df["close"].dropna()
            if len(series) <= self.lookback:
                continue
            prices[symbol] = series
        return prices

    def generate_signals(self, market_data: dict[str, Any]) -> list[StrategySignal]:
        prices = market_data.get("prices", {}) if isinstance(market_data, dict) else {}
        if not isinstance(prices, dict):
            return []
        returns: dict[str, float] = {}
        for symbol, series in prices.items():
            try:
                now_px = float(series.iloc[-1])
                then_px = float(series.iloc[-1 - self.lookback])
            except Exception:
                continue
            if then_px <= 0:
                continue
            ret = (now_px / then_px) - 1.0
            if np.isfinite(ret):
                returns[str(symbol)] = float(ret)
        if len(returns) < self.min_universe:
            return []
        ranked = sorted(returns.items(), key=lambda kv: kv[1])
        bucket = max(1, int(round(len(ranked) * self.top_quantile)))
        lows = [sym for sym, _ in ranked[:bucket]]
        highs = [sym for sym, _ in ranked[-bucket:]]
        zscores = _zscore_map(returns)
        signals: list[StrategySignal] = []
        for sym in highs:
            z = abs(float(zscores.get(sym, 0.0)))
            strength = float(min(1.0, max(0.05, z / 3.0)))
            signals.append(
                StrategySignal(
                    symbol=sym,
                    side="buy",
                    strength=strength,
                    confidence=float(min(0.95, 0.55 + (0.25 * strength))),
                    strategy_id=self.strategy_id,
                    signal_type="cross_sectional_momentum",
                    expected_return=float(max(0.0, returns.get(sym, 0.0))),
                    metadata={
                        "expected_edge_bps": float(max(2.0, returns.get(sym, 0.0) * 10000.0)),
                        "sleeve": self.strategy_id,
                        "zscore": float(zscores.get(sym, 0.0)),
                    },
                )
            )
        for sym in lows:
            z = abs(float(zscores.get(sym, 0.0)))
            strength = float(min(1.0, max(0.05, z / 3.0)))
            signals.append(
                StrategySignal(
                    symbol=sym,
                    side="sell",
                    strength=strength,
                    confidence=float(min(0.95, 0.55 + (0.25 * strength))),
                    strategy_id=self.strategy_id,
                    signal_type="cross_sectional_momentum",
                    expected_return=float(max(0.0, -returns.get(sym, 0.0))),
                    metadata={
                        "expected_edge_bps": float(max(2.0, -returns.get(sym, 0.0) * 10000.0)),
                        "sleeve": self.strategy_id,
                        "zscore": float(zscores.get(sym, 0.0)),
                    },
                )
            )
        return signals

    def generate(self, ctx: Any) -> list[StrategySignal]:
        return self.generate_signals({"prices": self._load_prices(ctx)})

