"""
Data labeling functions for trading models.

Provides explicit labelers for future returns, triple barrier labels,
and other trading-specific target variables.
"""
from __future__ import annotations
import numpy as np
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
from ai_trading.logging import logger
from ai_trading.utils.lazy_imports import load_pandas

if TYPE_CHECKING:
    import pandas as pd


def _as_price_series(prices: "pd.Series" | "pd.DataFrame") -> "pd.Series":
    pd = load_pandas()
    if isinstance(prices, pd.DataFrame):
        if "close" in prices.columns:
            return prices["close"]
        if "price" in prices.columns:
            return prices["price"]
        return prices.iloc[:, 0]
    return prices


def _as_horizons(horizons: int | Iterable[int]) -> list[int]:
    if isinstance(horizons, int):
        parsed = [int(horizons)]
    else:
        parsed = [int(value) for value in horizons]
    if not parsed or any(value <= 0 for value in parsed):
        raise ValueError("horizons must contain positive integers")
    return sorted(set(parsed))


def _cost_component(
    index: "pd.Index",
    value: float | "pd.Series" | None,
    *,
    default: float = 0.0,
) -> "pd.Series":
    pd = load_pandas()
    if isinstance(value, pd.Series):
        return pd.to_numeric(value.reindex(index), errors="coerce").fillna(float(default)).astype(float)
    return pd.Series(float(default if value is None else value), index=index, dtype=float)


def _path_extreme_return_bps(
    base: "pd.Series",
    path: "pd.Series",
    *,
    horizon: int,
    side_multiplier: float,
    reducer: str,
) -> "pd.Series":
    pd = load_pandas()
    base_safe = base.replace(0.0, np.nan)
    returns = [
        (((path.shift(-offset) / base_safe) - 1.0) * 10000.0 * side_multiplier)
        for offset in range(1, horizon + 1)
    ]
    frame = pd.concat(returns, axis=1)
    if reducer == "max":
        return frame.max(axis=1, skipna=True)
    return frame.min(axis=1, skipna=True)


def trade_quality_labels(
    prices: "pd.Series" | "pd.DataFrame",
    horizons: int | Iterable[int],
    *,
    spread_bps: float | "pd.Series" | None = None,
    slippage_bps: float | "pd.Series" = 0.0,
    fee_bps: float | "pd.Series" = 0.0,
    side: str = "long",
    binary_edge_threshold_bps: float = 0.0,
    multiclass_edge_threshold_bps: float = 5.0,
    risk_penalty: float = 1.0,
) -> "pd.DataFrame":
    """Generate leakage-safe multi-horizon trade-quality labels."""
    pd = load_pandas()
    normalized_side = str(side or "long").strip().lower()
    if normalized_side not in {"long", "short"}:
        raise ValueError("side must be 'long' or 'short'")
    side_multiplier = -1.0 if normalized_side == "short" else 1.0
    parsed_horizons = _as_horizons(horizons)
    try:
        close = pd.to_numeric(_as_price_series(prices), errors="coerce").astype(float)
        if isinstance(prices, pd.DataFrame):
            high = pd.to_numeric(prices["high"], errors="coerce").astype(float) if "high" in prices.columns else close
            low = pd.to_numeric(prices["low"], errors="coerce").astype(float) if "low" in prices.columns else close
            spread_source: Any = prices["spread_bps"] if spread_bps is None and "spread_bps" in prices.columns else spread_bps
        else:
            high = close
            low = close
            spread_source = spread_bps
        spread_cost_bps = _cost_component(close.index, spread_source)
        slippage_cost_bps = 2.0 * _cost_component(close.index, slippage_bps)
        fee_cost_bps = 2.0 * _cost_component(close.index, fee_bps)
        round_trip_cost_bps = spread_cost_bps + slippage_cost_bps + fee_cost_bps
        base_safe = close.replace(0.0, np.nan)
        start_timestamps = pd.Series(close.index, index=close.index)
        rows = []
        for horizon in parsed_horizons:
            future_close = close.shift(-horizon)
            label_end_timestamp = start_timestamps.shift(-horizon)
            gross_return_bps = (((future_close / base_safe) - 1.0) * 10000.0 * side_multiplier)
            net_edge_after_cost_bps = gross_return_bps - round_trip_cost_bps
            if normalized_side == "short":
                mae_bps = _path_extreme_return_bps(
                    base_safe,
                    high,
                    horizon=horizon,
                    side_multiplier=side_multiplier,
                    reducer="min",
                )
                mfe_bps = _path_extreme_return_bps(
                    base_safe,
                    low,
                    horizon=horizon,
                    side_multiplier=side_multiplier,
                    reducer="max",
                )
            else:
                mae_bps = _path_extreme_return_bps(
                    base_safe,
                    low,
                    horizon=horizon,
                    side_multiplier=side_multiplier,
                    reducer="min",
                )
                mfe_bps = _path_extreme_return_bps(
                    base_safe,
                    high,
                    horizon=horizon,
                    side_multiplier=side_multiplier,
                    reducer="max",
                )
            downside_bps = mae_bps.clip(upper=0.0).abs()
            risk_adjusted_return_bps = net_edge_after_cost_bps - (float(risk_penalty) * downside_bps)
            risk_adjusted_return = net_edge_after_cost_bps / downside_bps.replace(0.0, np.nan)
            frame = pd.DataFrame(
                {
                    "label_start_timestamp": start_timestamps,
                    "label_end_timestamp": label_end_timestamp,
                    "horizon_bars": int(horizon),
                    "side": normalized_side,
                    "gross_return_bps": gross_return_bps,
                    "spread_cost_bps": spread_cost_bps,
                    "slippage_cost_bps": slippage_cost_bps,
                    "fee_cost_bps": fee_cost_bps,
                    "round_trip_cost_bps": round_trip_cost_bps,
                    "net_edge_after_cost_bps": net_edge_after_cost_bps,
                    "mae_bps": mae_bps,
                    "mfe_bps": mfe_bps,
                    "risk_adjusted_return_bps": risk_adjusted_return_bps,
                    "risk_adjusted_return": risk_adjusted_return,
                },
                index=close.index,
            )
            frame["quality_binary"] = (
                frame["net_edge_after_cost_bps"] > float(binary_edge_threshold_bps)
            ).astype(int)
            threshold = abs(float(multiclass_edge_threshold_bps))
            frame["quality_multiclass"] = np.select(
                [
                    frame["net_edge_after_cost_bps"] >= threshold,
                    frame["net_edge_after_cost_bps"] <= -threshold,
                ],
                [1, -1],
                default=0,
            ).astype(int)
            rows.append(frame)
        if not rows:
            return pd.DataFrame()
        result = pd.concat(rows, axis=0).replace([np.inf, -np.inf], np.nan)
        result = result.dropna(
            subset=[
                "label_start_timestamp",
                "label_end_timestamp",
                "gross_return_bps",
                "net_edge_after_cost_bps",
                "mae_bps",
                "mfe_bps",
            ]
        )
        result = result.sort_values(["label_start_timestamp", "horizon_bars"]).reset_index(drop=True)
        logger.debug(
            "Generated trade quality labels",
            extra={"rows": len(result), "horizons": parsed_horizons, "side": normalized_side},
        )
        return result
    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Error generating trade quality labels: {e}")
        return pd.DataFrame()

def fixed_horizon_return(prices: "pd.Series" | "pd.DataFrame", horizon_bars: int, fee_bps: float = 0.0) -> "pd.Series":
    pd = load_pandas()
    """
    Calculate fixed horizon future returns net of fees.
    
    Args:
        prices: Price series or DataFrame with price column
        horizon_bars: Number of bars to look ahead
        fee_bps: Transaction fee in basis points (e.g., 5.0 for 5 bps)
        
    Returns:
        Series of future log returns net of fees
    """
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    try:
        if isinstance(prices, pd.DataFrame):
            if 'close' in prices.columns:
                prices = prices['close']
            elif 'price' in prices.columns:
                prices = prices['price']
            else:
                prices = prices.iloc[:, 0]
        future_prices = prices.shift(-horizon_bars)
        log_returns = np.log(future_prices / prices)
        fee_rate = fee_bps / 10000.0
        net_returns = log_returns - 2 * fee_rate
        net_returns.name = f'future_return_h{horizon_bars}_fee{fee_bps}bps'
        logger.debug(f'Generated {len(net_returns.dropna())} labels for horizon {horizon_bars}')
        return net_returns
    except (ValueError, TypeError) as e:
        logger.error(f'Error calculating fixed horizon returns: {e}')
        return pd.Series(dtype=float)

def triple_barrier_labels(
    prices: "pd.Series" | "pd.DataFrame",
    events: "pd.DataFrame" | None = None,
    pt_sl: tuple | None = None,
    t1: "pd.Series" | None = None,
    min_ret: float = 0.0,
    num_threads: int = 1,
    vertical_barrier_times: "pd.Series" | None = None,
) -> "pd.DataFrame":
    pd = load_pandas()
    _ = num_threads  # Reserved for API compatibility; implementation is single-threaded.
    """
    Triple barrier labeling method.
    
    This is a stub implementation with proper API structure.
    Full implementation would require advanced barrier calibration.
    
    Args:
        prices: Price series or DataFrame
        events: DataFrame with events (index=timestamps)
        pt_sl: Tuple of (profit_taking, stop_loss) thresholds
        t1: Series of timestamp endpoints for barriers
        min_ret: Minimum return to consider for labeling
        num_threads: Number of threads for parallel processing
        vertical_barrier_times: Optional vertical barrier timestamps
        
    Returns:
        DataFrame with labels: t1 (end time), ret (return), bin (label)
    """
    def _empty_result():
        return pd.DataFrame(
            {
                "t1": pd.Series(dtype="datetime64[ns]"),
                "ret": pd.Series(dtype=float),
                "bin": pd.Series(dtype=int),
            }
        )

    try:
        if isinstance(prices, pd.DataFrame):
            if 'close' in prices.columns:
                prices = prices['close']
            elif 'price' in prices.columns:
                prices = prices['price']
            else:
                prices = prices.iloc[:, 0]
        if events is None:
            events = pd.DataFrame(index=prices.index)
        if pt_sl is None:
            pt_sl = (0.02, -0.02)
        if t1 is None and vertical_barrier_times is None:
            t1 = pd.Series(index=events.index, dtype='datetime64[ns]')
            for idx in events.index:
                if isinstance(idx, pd.Timestamp):
                    barrier_time = idx + pd.Timedelta(days=5)
                    if barrier_time <= prices.index.max():
                        t1.loc[idx] = barrier_time
                    else:
                        t1.loc[idx] = prices.index.max()
        elif vertical_barrier_times is not None:
            t1 = vertical_barrier_times
        labels = []
        label_index = []
        for event_time in events.index:
            if event_time not in prices.index:
                continue
            start_price = prices.loc[event_time]
            end_time = t1.loc[event_time] if event_time in t1.index else event_time
            if pd.isna(start_price) or pd.isna(end_time):
                continue
            price_slice = prices.loc[event_time:end_time]
            if len(price_slice) <= 1:
                continue
            returns = price_slice / start_price - 1
            profit_hit = returns >= pt_sl[0]
            loss_hit = returns <= pt_sl[1]
            if profit_hit.any():
                first_profit = profit_hit.idxmax() if profit_hit.any() else None
            else:
                first_profit = None
            if loss_hit.any():
                first_loss = loss_hit.idxmax() if loss_hit.any() else None
            else:
                first_loss = None
            if first_profit is not None and first_loss is not None:
                if first_profit <= first_loss:
                    label_bin = 1
                    label_t1 = first_profit
                    label_ret = returns.loc[first_profit]
                else:
                    label_bin = -1
                    label_t1 = first_loss
                    label_ret = returns.loc[first_loss]
            elif first_profit is not None:
                label_bin = 1
                label_t1 = first_profit
                label_ret = returns.loc[first_profit]
            elif first_loss is not None:
                label_bin = -1
                label_t1 = first_loss
                label_ret = returns.loc[first_loss]
            else:
                label_bin = 0
                label_t1 = end_time
                label_ret = returns.iloc[-1]
            if abs(label_ret) >= min_ret:
                labels.append({'t1': label_t1, 'ret': label_ret, 'bin': label_bin})
                label_index.append(event_time)
        if not labels:
            result_df = _empty_result()
        else:
            result_df = pd.DataFrame(labels, index=pd.Index(label_index, name=events.index.name))
        logger.info(f"Generated triple barrier labels: {len(result_df)} events, {(result_df['bin'] == 1).sum()} profits, {(result_df['bin'] == -1).sum()} losses, {(result_df['bin'] == 0).sum()} timeouts")
        return result_df
    except (ValueError, TypeError) as e:
        logger.error(f'Error in triple barrier labeling: {e}')
        return _empty_result()

def get_daily_vol(prices: "pd.Series", span0: int = 100) -> "pd.Series":
    pd = load_pandas()
    """
    Calculate daily volatility for barrier calibration.
    
    Args:
        prices: Price series
        span0: Span for EWMA calculation
        
    Returns:
        Daily volatility series
    """
    try:
        daily_ret = prices.resample('1D').last().pct_change().dropna()
        vol = daily_ret.ewm(span=span0).std()
        return vol
    except (ValueError, TypeError) as e:
        logger.error(f'Error calculating daily volatility: {e}')
        return pd.Series(dtype=float)
