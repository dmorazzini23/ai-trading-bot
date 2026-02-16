"""Portfolio-level risk targeting and caps."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import pstdev
from typing import Any


@dataclass(slots=True)
class PortfolioLimitsResult:
    scaled_targets: dict[str, float]
    scale: float
    reasons: list[str]


def _annualized_vol(daily_returns: list[float]) -> float:
    if len(daily_returns) < 2:
        return 0.0
    return float(pstdev(daily_returns) * (252.0 ** 0.5))


def _correlation(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True))
    var_x = sum((a - mean_x) ** 2 for a in x)
    var_y = sum((b - mean_y) ** 2 for b in y)
    if var_x <= 0 or var_y <= 0:
        return 0.0
    return float(cov / ((var_x ** 0.5) * (var_y ** 0.5)))


def apply_portfolio_limits(
    *,
    targets: dict[str, float],
    symbol_returns: dict[str, list[float]] | None = None,
    target_annual_vol: float = 0.18,
    vol_min_scale: float = 0.25,
    vol_max_scale: float = 1.25,
    max_symbol_weight: float = 0.12,
    corr_threshold: float = 0.80,
    corr_group_gross_cap: float = 0.35,
) -> PortfolioLimitsResult:
    reasons: list[str] = []
    scaled = dict(targets)
    gross = sum(abs(value) for value in scaled.values())
    scale = 1.0

    if symbol_returns:
        combined: list[float] = []
        for series in symbol_returns.values():
            combined.extend(float(x) for x in series)
        realized = _annualized_vol(combined)
        if realized > 0 and target_annual_vol > 0:
            scale = min(float(vol_max_scale), max(float(vol_min_scale), target_annual_vol / realized))
            if abs(scale - 1.0) > 1e-12:
                reasons.append("VOL_TARGET_SCALE")

    if scale != 1.0:
        for symbol in list(scaled):
            scaled[symbol] = float(scaled[symbol]) * scale
        gross = sum(abs(value) for value in scaled.values())

    if gross > 0 and max_symbol_weight > 0:
        symbol_cap = gross * float(max_symbol_weight)
        for symbol, value in list(scaled.items()):
            clipped = max(-symbol_cap, min(symbol_cap, float(value)))
            if clipped != value:
                scaled[symbol] = clipped
                if "RISK_CAP_PORTFOLIO" not in reasons:
                    reasons.append("RISK_CAP_PORTFOLIO")

    if symbol_returns and corr_group_gross_cap > 0:
        symbols = list(symbol_returns.keys())
        if len(symbols) >= 2:
            most_corr_pair: tuple[str, str] | None = None
            most_corr = 0.0
            for idx, left in enumerate(symbols):
                for right in symbols[idx + 1 :]:
                    corr = abs(_correlation(symbol_returns[left], symbol_returns[right]))
                    if corr > most_corr:
                        most_corr = corr
                        most_corr_pair = (left, right)
            if most_corr_pair and most_corr >= corr_threshold:
                left, right = most_corr_pair
                gross_now = sum(abs(v) for v in scaled.values()) or 1.0
                pair_gross = abs(scaled.get(left, 0.0)) + abs(scaled.get(right, 0.0))
                allowed = gross_now * float(corr_group_gross_cap)
                if pair_gross > allowed:
                    ratio = allowed / pair_gross
                    scaled[left] *= ratio
                    scaled[right] *= ratio
                    reasons.append("CORR_CLUSTER_CAP")

    return PortfolioLimitsResult(scaled_targets=scaled, scale=scale, reasons=reasons)
