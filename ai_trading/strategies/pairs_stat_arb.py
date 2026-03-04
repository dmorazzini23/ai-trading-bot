"""Pairs statistical arbitrage strategy sleeve."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.enums import RiskLevel
from .base import BaseStrategy, StrategySignal


class PairsStatArbStrategy(BaseStrategy):
    """Mean-reversion signals from highly correlated symbol pairs."""

    def __init__(
        self,
        strategy_id: str = "pairs_stat_arb",
        name: str = "Pairs Stat Arb",
        risk_level: RiskLevel = RiskLevel.MODERATE,
        *,
        lookback: int = 60,
        z_entry: float = 1.6,
        min_universe: int = 6,
    ) -> None:
        super().__init__(strategy_id, name, risk_level)
        self.lookback = max(30, int(lookback))
        self.z_entry = float(max(0.5, min(z_entry, 4.0)))
        self.min_universe = max(4, int(min_universe))

    def _load_returns(self, ctx: Any) -> dict[str, np.ndarray]:
        fetcher = getattr(ctx, "data_fetcher", None)
        symbols = [str(s).strip().upper() for s in (getattr(ctx, "tickers", []) or []) if str(s).strip()]
        if fetcher is None or len(symbols) < self.min_universe:
            return {}
        returns: dict[str, np.ndarray] = {}
        for symbol in symbols:
            try:
                df = fetcher.get_daily_df(ctx, symbol)
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            if df is None or getattr(df, "empty", True) or "close" not in getattr(df, "columns", []):
                continue
            closes = np.asarray(df["close"].dropna().values, dtype=float)
            if closes.size <= self.lookback + 2:
                continue
            tail = closes[-(self.lookback + 1) :]
            rets = np.diff(tail) / np.maximum(tail[:-1], 1e-9)
            if rets.size < self.lookback:
                continue
            returns[symbol] = rets
        return returns

    def generate_signals(self, market_data: dict[str, Any]) -> list[StrategySignal]:
        returns = market_data.get("returns", {}) if isinstance(market_data, dict) else {}
        if not isinstance(returns, dict) or len(returns) < self.min_universe:
            return []

        symbols = sorted(returns.keys())
        best_pair: tuple[str, str] | None = None
        best_corr = 0.0
        for i, left in enumerate(symbols):
            x = returns[left]
            for right in symbols[i + 1 :]:
                y = returns[right]
                if x.size != y.size or x.size < 20:
                    continue
                try:
                    corr = float(np.corrcoef(x, y)[0, 1])
                except (FloatingPointError, ValueError, TypeError):
                    continue
                if np.isfinite(corr) and abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_pair = (left, right)

        if best_pair is None or abs(best_corr) < 0.65:
            return []

        left, right = best_pair
        spread = returns[left] - returns[right]
        spread_mean = float(np.mean(spread))
        spread_std = float(np.std(spread))
        if spread_std <= 1e-9:
            return []
        z = float((spread[-1] - spread_mean) / spread_std)
        abs_z = abs(z)
        if abs_z < self.z_entry:
            return []

        strength = float(min(1.0, max(0.05, abs_z / 3.5)))
        confidence = float(min(0.95, 0.55 + (0.25 * strength)))
        edge_bps = float(max(2.0, (abs_z - self.z_entry + 1.0) * 8.0))
        if z > 0:
            # spread wide: short left, long right
            return [
                StrategySignal(
                    symbol=left,
                    side="sell",
                    strength=strength,
                    confidence=confidence,
                    strategy_id=self.strategy_id,
                    signal_type="pairs_stat_arb",
                    metadata={
                        "expected_edge_bps": edge_bps,
                        "sleeve": self.strategy_id,
                        "pair_symbol": right,
                        "pair_zscore": z,
                        "pair_corr": best_corr,
                    },
                ),
                StrategySignal(
                    symbol=right,
                    side="buy",
                    strength=strength,
                    confidence=confidence,
                    strategy_id=self.strategy_id,
                    signal_type="pairs_stat_arb",
                    metadata={
                        "expected_edge_bps": edge_bps,
                        "sleeve": self.strategy_id,
                        "pair_symbol": left,
                        "pair_zscore": z,
                        "pair_corr": best_corr,
                    },
                ),
            ]
        # spread narrow negative: long left, short right
        return [
            StrategySignal(
                symbol=left,
                side="buy",
                strength=strength,
                confidence=confidence,
                strategy_id=self.strategy_id,
                signal_type="pairs_stat_arb",
                metadata={
                    "expected_edge_bps": edge_bps,
                    "sleeve": self.strategy_id,
                    "pair_symbol": right,
                    "pair_zscore": z,
                    "pair_corr": best_corr,
                },
            ),
            StrategySignal(
                symbol=right,
                side="sell",
                strength=strength,
                confidence=confidence,
                strategy_id=self.strategy_id,
                signal_type="pairs_stat_arb",
                metadata={
                    "expected_edge_bps": edge_bps,
                    "sleeve": self.strategy_id,
                    "pair_symbol": left,
                    "pair_zscore": z,
                    "pair_corr": best_corr,
                },
            ),
        ]

    def generate(self, ctx: Any) -> list[StrategySignal]:
        return self.generate_signals({"returns": self._load_returns(ctx)})
