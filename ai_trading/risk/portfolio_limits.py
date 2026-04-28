"""Portfolio-level risk targeting and caps."""
from __future__ import annotations

import math
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


def _portfolio_return_series(
    *,
    targets: dict[str, float],
    symbol_returns: dict[str, list[float]],
) -> list[float]:
    gross = sum(abs(float(value)) for value in targets.values())
    if gross <= 0:
        return []
    weighted_series: dict[str, list[float]] = {}
    signed_weights: dict[str, float] = {}
    for symbol, returns in symbol_returns.items():
        if symbol not in targets:
            continue
        weight = float(targets.get(symbol, 0.0)) / gross
        if abs(weight) <= 0:
            continue
        cleaned: list[float] = []
        for value in returns:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(parsed):
                cleaned.append(parsed)
        if len(cleaned) < 2:
            continue
        weighted_series[symbol] = cleaned
        signed_weights[symbol] = weight
    if not weighted_series:
        return []
    aligned_len = min(len(values) for values in weighted_series.values())
    if aligned_len < 2:
        return []
    portfolio_returns: list[float] = []
    for idx in range(-aligned_len, 0):
        step_return = 0.0
        for symbol, series in weighted_series.items():
            step_return += signed_weights[symbol] * series[idx]
        portfolio_returns.append(step_return)
    return portfolio_returns


def _explicit_cap_denominator(
    *,
    portfolio_equity: float | None,
    configured_gross_cap: float | None,
) -> float:
    for candidate in (configured_gross_cap, portfolio_equity):
        if candidate is None:
            continue
        try:
            parsed = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed) and parsed > 0:
            return parsed
    return 0.0


def _connected_components(graph: dict[str, set[str]]) -> list[set[str]]:
    seen: set[str] = set()
    components: list[set[str]] = []
    for symbol in graph:
        if symbol in seen:
            continue
        stack = [symbol]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.add(current)
            stack.extend(graph[current] - seen)
        components.append(component)
    return components


def apply_portfolio_limits(
    *,
    targets: dict[str, float],
    symbol_returns: dict[str, list[float]] | None = None,
    vol_targeting_enabled: bool = True,
    target_annual_vol: float = 0.18,
    vol_lookback_days: int = 20,
    vol_min_scale: float = 0.25,
    vol_max_scale: float = 1.25,
    concentration_cap_enabled: bool = True,
    max_symbol_weight: float = 0.12,
    max_cluster_weight: float = 0.25,
    corr_cap_enabled: bool = True,
    corr_lookback_days: int = 30,
    corr_threshold: float = 0.80,
    corr_group_gross_cap: float = 0.35,
    portfolio_equity: float | None = None,
    configured_gross_cap: float | None = None,
) -> PortfolioLimitsResult:
    reasons: list[str] = []
    scaled = dict(targets)
    gross = sum(abs(value) for value in scaled.values())
    scale = 1.0

    if symbol_returns and vol_targeting_enabled:
        portfolio_returns = _portfolio_return_series(
            targets=scaled,
            symbol_returns=symbol_returns,
        )
        lookback = max(0, int(vol_lookback_days))
        if lookback > 0 and len(portfolio_returns) > lookback:
            portfolio_returns = portfolio_returns[-lookback:]
        realized = _annualized_vol(portfolio_returns)
        if realized > 0 and target_annual_vol > 0:
            scale = min(float(vol_max_scale), max(float(vol_min_scale), target_annual_vol / realized))
            if abs(scale - 1.0) > 1e-12:
                reasons.append("VOL_TARGET_SCALE")

    if scale != 1.0:
        for symbol in list(scaled):
            scaled[symbol] = float(scaled[symbol]) * scale
        gross = sum(abs(value) for value in scaled.values())

    cap_denominator = _explicit_cap_denominator(
        portfolio_equity=portfolio_equity,
        configured_gross_cap=configured_gross_cap,
    )
    if concentration_cap_enabled and cap_denominator > 0 and max_symbol_weight > 0:
        symbol_cap = cap_denominator * float(max_symbol_weight)
        for symbol, value in list(scaled.items()):
            clipped = max(-symbol_cap, min(symbol_cap, float(value)))
            if clipped != value:
                scaled[symbol] = clipped
                if "RISK_CAP_PORTFOLIO" not in reasons:
                    reasons.append("RISK_CAP_PORTFOLIO")

    if symbol_returns and corr_cap_enabled and corr_group_gross_cap > 0:
        symbols = [
            symbol
            for symbol, value in scaled.items()
            if abs(float(value)) > 0 and symbol in symbol_returns
        ]
        if len(symbols) >= 2:
            graph: dict[str, set[str]] = {symbol: set() for symbol in symbols}
            for idx, left in enumerate(symbols):
                for right in symbols[idx + 1 :]:
                    left_series = symbol_returns[left]
                    right_series = symbol_returns[right]
                    lookback = max(0, int(corr_lookback_days))
                    if lookback > 0:
                        if len(left_series) > lookback:
                            left_series = left_series[-lookback:]
                        if len(right_series) > lookback:
                            right_series = right_series[-lookback:]
                    corr = abs(_correlation(list(left_series), list(right_series)))
                    if corr >= corr_threshold:
                        graph[left].add(right)
                        graph[right].add(left)
            gross_now = sum(abs(v) for v in scaled.values()) or 1.0
            for component in _connected_components(graph):
                if len(component) < 2:
                    continue
                component_gross = sum(abs(scaled.get(symbol, 0.0)) for symbol in component)
                corr_cap_fraction = float(corr_group_gross_cap)
                if max_cluster_weight > 0:
                    corr_cap_fraction = min(corr_cap_fraction, float(max_cluster_weight))
                allowed = gross_now * corr_cap_fraction
                if component_gross > allowed:
                    ratio = allowed / component_gross
                    for symbol in component:
                        scaled[symbol] *= ratio
                    if "CORR_CLUSTER_CAP" not in reasons:
                        reasons.append("CORR_CLUSTER_CAP")

    return PortfolioLimitsResult(scaled_targets=scaled, scale=scale, reasons=reasons)
