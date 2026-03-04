"""Post-earnings announcement drift style event strategy sleeve."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.enums import RiskLevel
from .base import BaseStrategy, StrategySignal


class PEADEventStrategy(BaseStrategy):
    """Detects gap-plus-volume shocks and trades continuation drift."""

    def __init__(
        self,
        strategy_id: str = "pead_event",
        name: str = "PEAD Event Drift",
        risk_level: RiskLevel = RiskLevel.MODERATE,
        *,
        gap_threshold: float = 0.03,
        volume_multiple: float = 1.8,
        lookback: int = 40,
    ) -> None:
        super().__init__(strategy_id, name, risk_level)
        self.gap_threshold = float(max(0.005, min(gap_threshold, 0.20)))
        self.volume_multiple = float(max(1.1, min(volume_multiple, 8.0)))
        self.lookback = max(20, int(lookback))

    def _daily_frame(self, ctx: Any, symbol: str) -> Any | None:
        fetcher = getattr(ctx, "data_fetcher", None)
        if fetcher is None:
            return None
        try:
            df = fetcher.get_daily_df(ctx, symbol)
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
        if df is None or getattr(df, "empty", True):
            return None
        required = {"open", "close", "volume"}
        if not required.issubset(set(getattr(df, "columns", []))):
            return None
        return df

    def generate_signals(self, market_data: dict[str, Any]) -> list[StrategySignal]:
        frames = market_data.get("frames", {}) if isinstance(market_data, dict) else {}
        if not isinstance(frames, dict):
            return []
        signals: list[StrategySignal] = []
        for symbol, df in frames.items():
            try:
                if len(df) <= self.lookback + 2:
                    continue
                tail = df.tail(self.lookback + 2)
                prev_close = float(tail["close"].iloc[-2])
                today_open = float(tail["open"].iloc[-1])
                today_close = float(tail["close"].iloc[-1])
                if prev_close <= 0 or today_open <= 0:
                    continue
                gap = (today_open / prev_close) - 1.0
                vol_arr = np.asarray(tail["volume"].iloc[:-1].values, dtype=float)
                avg_vol = float(np.mean(vol_arr))
                today_vol = float(tail["volume"].iloc[-1])
                if avg_vol <= 0:
                    continue
                vol_mult = today_vol / avg_vol
                intraday_follow = (today_close / today_open) - 1.0
            except (AttributeError, KeyError, IndexError, TypeError, ValueError):
                continue

            if abs(gap) < self.gap_threshold or vol_mult < self.volume_multiple:
                continue
            # Continuation trigger: same sign close-open as the overnight gap.
            if np.sign(intraday_follow) != np.sign(gap):
                continue
            side = "buy" if gap > 0 else "sell"
            strength = float(min(1.0, max(0.05, abs(gap) / (self.gap_threshold * 3.0))))
            confidence = float(min(0.95, 0.55 + (0.25 * strength)))
            edge_bps = float(max(2.0, abs(gap) * 10000.0 * min(vol_mult / self.volume_multiple, 2.0) * 0.25))
            signals.append(
                StrategySignal(
                    symbol=str(symbol).strip().upper(),
                    side=side,
                    strength=strength,
                    confidence=confidence,
                    strategy_id=self.strategy_id,
                    signal_type="pead_event",
                    expected_return=float(abs(gap)),
                    metadata={
                        "expected_edge_bps": edge_bps,
                        "sleeve": self.strategy_id,
                        "event_gap": float(gap),
                        "event_volume_multiple": float(vol_mult),
                    },
                )
            )
        return signals

    def generate(self, ctx: Any) -> list[StrategySignal]:
        symbols = [str(s).strip().upper() for s in (getattr(ctx, "tickers", []) or []) if str(s).strip()]
        frames: dict[str, Any] = {}
        for symbol in symbols:
            df = self._daily_frame(ctx, symbol)
            if df is not None:
                frames[symbol] = df
        return self.generate_signals({"frames": frames})
