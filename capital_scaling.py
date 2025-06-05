from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Any

import pandas as pd


@dataclass
class CapitalBand:
    """Configuration for a single capital tier."""

    name: str
    min_equity: float
    max_equity: Optional[float]
    kelly_frac: float
    capital_cap: float
    max_pos_dollars: float
    adv_pct: float
    sector_cap: float
    corr_limit: float


class CapitalScalingEngine:
    """Manage capital-aware risk parameters."""

    def __init__(self) -> None:
        self.bands = [
            CapitalBand("small", 0, 100_000, 0.6, 0.08, 10_000, 0.002, 0.4, 0.6),
            CapitalBand("mid", 100_000, 500_000, 0.5, 0.06, 20_000, 0.0015, 0.3, 0.5),
            CapitalBand("large", 500_000, 1_000_000, 0.4, 0.05, 40_000, 0.001, 0.25, 0.4),
            CapitalBand("xl", 1_000_000, None, 0.3, 0.04, 60_000, 0.0008, 0.2, 0.35),
        ]
        self.current: CapitalBand = self.bands[0]

    def band_for_equity(self, equity: float) -> CapitalBand:
        for band in self.bands:
            if (band.max_equity is None or equity < band.max_equity) and equity >= band.min_equity:
                return band
        return self.bands[-1]

    def update(self, ctx: Any, equity: float) -> None:
        band = self.band_for_equity(equity)
        if band.name != self.current.name:
            ctx.logger.info("CAPITAL_BAND_SWITCH", extra={"band": band.name})
        self.current = band
        ctx.capital_band = band.name
        ctx.kelly_fraction = band.kelly_frac
        ctx.adv_target_pct = band.adv_pct
        ctx.max_position_dollars = band.max_pos_dollars
        ctx.params['CAPITAL_CAP'] = band.capital_cap
        ctx.sector_cap = band.sector_cap
        ctx.correlation_limit = band.corr_limit

    def compression_factor(self, equity: float) -> float:
        return 1.0 / (1.0 + equity / 1_000_000.0)


class CapitalGrowthSimulator:
    """Simple capital growth simulator under dynamic scaling."""

    def __init__(self, scaler: CapitalScalingEngine) -> None:
        self.scaler = scaler

    def simulate(self, returns: Sequence[float], starting_capital: float) -> pd.DataFrame:
        equity = starting_capital
        records = []
        for r in returns:
            band = self.scaler.band_for_equity(equity)
            frac = band.kelly_frac
            equity *= 1 + r * frac
            drawdown = 0.0
            if records:
                peak = max(rec[0] for rec in records)
                drawdown = (peak - equity) / peak if peak else 0.0
            records.append((equity, band.name, drawdown))
        return pd.DataFrame(records, columns=["equity", "band", "drawdown"])

