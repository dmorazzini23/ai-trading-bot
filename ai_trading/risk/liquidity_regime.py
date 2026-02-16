"""Liquidity regime classification and participation caps."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LiquidityRegime(str, Enum):
    THIN = "THIN"
    NORMAL = "NORMAL"
    THICK = "THICK"


@dataclass(slots=True)
class LiquidityFeatures:
    rolling_volume: float
    spread_bps: float
    volatility_proxy: float


def classify_liquidity_regime(
    features: LiquidityFeatures,
    *,
    thin_spread_bps: float = 25.0,
    thin_vol_mult: float = 1.8,
) -> LiquidityRegime:
    if features.rolling_volume <= 0:
        return LiquidityRegime.THIN
    if features.spread_bps >= thin_spread_bps or features.volatility_proxy >= thin_vol_mult:
        return LiquidityRegime.THIN
    if features.spread_bps <= max(1.0, thin_spread_bps * 0.4) and features.volatility_proxy <= max(
        0.2, thin_vol_mult * 0.6
    ):
        return LiquidityRegime.THICK
    return LiquidityRegime.NORMAL


def enforce_participation_cap(
    *,
    order_qty: float,
    rolling_volume: float,
    max_participation_pct: float,
    mode: str = "block",
    scale_min: float = 0.25,
) -> tuple[bool, float, str | None]:
    qty = abs(float(order_qty))
    vol = max(0.0, float(rolling_volume))
    cap = max(0.0, float(max_participation_pct))
    if vol <= 0 or cap <= 0:
        return False, float(order_qty), "LIQ_PARTICIPATION_BLOCK"
    allowed_qty = vol * cap
    if qty <= allowed_qty:
        return True, float(order_qty), None
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "scale":
        ratio = max(float(scale_min), min(1.0, allowed_qty / max(qty, 1e-9)))
        scaled = float(order_qty) * ratio
        return True, scaled, "LIQ_REGIME_THIN_SCALE"
    return False, float(order_qty), "LIQ_PARTICIPATION_BLOCK"
