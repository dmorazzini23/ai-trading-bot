"""Low-beta defensive rotation strategy sleeve."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.enums import RiskLevel
from .base import BaseStrategy, StrategySignal


class LowBetaDefensiveStrategy(BaseStrategy):
    """Favors low-beta names during stressed market regimes."""

    def __init__(
        self,
        strategy_id: str = "low_beta_defensive",
        name: str = "Low Beta Defensive",
        risk_level: RiskLevel = RiskLevel.CONSERVATIVE,
        *,
        beta_lookback: int = 60,
        stress_vol_threshold: float = 0.018,
    ) -> None:
        super().__init__(strategy_id, name, risk_level)
        self.beta_lookback = max(30, int(beta_lookback))
        self.stress_vol_threshold = float(max(0.002, min(stress_vol_threshold, 0.08)))

    @staticmethod
    def _returns(series: Any, lookback: int) -> np.ndarray:
        arr = np.asarray(series.dropna().tail(lookback + 1).values, dtype=float)
        if arr.size <= lookback:
            return np.asarray([], dtype=float)
        return np.diff(arr) / np.maximum(arr[:-1], 1e-9)

    def _load_daily(self, ctx: Any) -> tuple[dict[str, Any], Any | None]:
        fetcher = getattr(ctx, "data_fetcher", None)
        symbols = [str(s).strip().upper() for s in (getattr(ctx, "tickers", []) or []) if str(s).strip()]
        if fetcher is None or not symbols:
            return ({}, None)
        market_proxy = None
        for proxy in ("SPY", "QQQ"):
            try:
                proxy_df = fetcher.get_daily_df(ctx, proxy)
            except (AttributeError, KeyError, TypeError, ValueError):
                proxy_df = None
            if proxy_df is not None and not getattr(proxy_df, "empty", True) and "close" in getattr(proxy_df, "columns", []):
                market_proxy = proxy_df["close"]
                break
        series_map: dict[str, Any] = {}
        for symbol in symbols:
            try:
                df = fetcher.get_daily_df(ctx, symbol)
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            if df is None or getattr(df, "empty", True) or "close" not in getattr(df, "columns", []):
                continue
            series_map[symbol] = df["close"]
        return series_map, market_proxy

    def generate_signals(self, market_data: dict[str, Any]) -> list[StrategySignal]:
        series_map = market_data.get("series", {}) if isinstance(market_data, dict) else {}
        market_proxy = market_data.get("market_proxy") if isinstance(market_data, dict) else None
        if not isinstance(series_map, dict) or market_proxy is None or len(series_map) < 4:
            return []

        proxy_rets = self._returns(market_proxy, self.beta_lookback)
        if proxy_rets.size < self.beta_lookback:
            return []
        market_vol = float(np.std(proxy_rets))
        stress_mode = market_vol >= self.stress_vol_threshold
        if not stress_mode:
            return []

        betas: dict[str, float] = {}
        for symbol, series in series_map.items():
            rets = self._returns(series, self.beta_lookback)
            if rets.size != proxy_rets.size or rets.size < 20:
                continue
            variance = float(np.var(proxy_rets))
            if variance <= 1e-9:
                continue
            cov = float(np.cov(rets, proxy_rets)[0, 1])
            beta = cov / variance
            if np.isfinite(beta):
                betas[symbol] = float(beta)

        if len(betas) < 4:
            return []
        ranked = sorted(betas.items(), key=lambda kv: kv[1])
        bucket = max(1, int(round(len(ranked) * 0.25)))
        low_beta = ranked[:bucket]
        signals: list[StrategySignal] = []
        for symbol, beta in low_beta:
            beta_gap = max(0.0, 1.0 - beta)
            strength = float(min(1.0, max(0.05, beta_gap / 1.5)))
            confidence = float(min(0.95, 0.60 + (0.20 * strength)))
            edge_bps = float(max(2.0, beta_gap * 10.0))
            signals.append(
                StrategySignal(
                    symbol=symbol,
                    side="buy",
                    strength=strength,
                    confidence=confidence,
                    strategy_id=self.strategy_id,
                    signal_type="low_beta_defensive",
                    metadata={
                        "expected_edge_bps": edge_bps,
                        "sleeve": self.strategy_id,
                        "market_stress_vol": market_vol,
                        "beta": float(beta),
                    },
                )
            )
        return signals

    def generate(self, ctx: Any) -> list[StrategySignal]:
        series_map, market_proxy = self._load_daily(ctx)
        return self.generate_signals({"series": series_map, "market_proxy": market_proxy})
