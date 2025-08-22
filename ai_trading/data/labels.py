"""
Data labeling functions for trading models.

Provides explicit labelers for future returns, triple barrier labels,
and other trading-specific target variables.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional

# Use the centralized logger as per AGENTS.md
from ai_trading.logging import logger


def fixed_horizon_return(
    prices: Union[pd.Series, pd.DataFrame],
    horizon_bars: int,
    fee_bps: float = 0.0
) -> pd.Series:
    """
    Calculate fixed horizon future returns net of fees.
    
    Args:
        prices: Price series or DataFrame with price column
        horizon_bars: Number of bars to look ahead
        fee_bps: Transaction fee in basis points (e.g., 5.0 for 5 bps)
        
    Returns:
        Series of future log returns net of fees
    """
    try:
        # Handle DataFrame input
        if isinstance(prices, pd.DataFrame):
            if 'close' in prices.columns:
                prices = prices['close']
            elif 'price' in prices.columns:
                prices = prices['price']
            else:
                prices = prices.iloc[:, 0]  # Use first column
        
        # Calculate future prices
        future_prices = prices.shift(-horizon_bars)
        
        # Calculate log returns
        log_returns = np.log(future_prices / prices)
        
        # Subtract transaction fees (convert bps to decimal)
        fee_rate = fee_bps / 10000.0
        net_returns = log_returns - (2 * fee_rate)  # Round-trip cost
        
        # Set name for the series
        net_returns.name = f'future_return_h{horizon_bars}_fee{fee_bps}bps'
        
        logger.debug(f"Generated {len(net_returns.dropna())} labels for horizon {horizon_bars}")
        
        return net_returns
        
    except Exception as e:
        logger.error(f"Error calculating fixed horizon returns: {e}")
        return pd.Series(dtype=float)


def triple_barrier_labels(
    prices: Union[pd.Series, pd.DataFrame],
    events: Optional[pd.DataFrame] = None,
    pt_sl: Optional[tuple] = None,
    t1: Optional[pd.Series] = None,
    min_ret: float = 0.0,
    num_threads: int = 1,
    vertical_barrier_times: Optional[pd.Series] = None
) -> pd.DataFrame:
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
    try:
        # Handle DataFrame input
        if isinstance(prices, pd.DataFrame):
            if 'close' in prices.columns:
                prices = prices['close']
            elif 'price' in prices.columns:
                prices = prices['price']
            else:
                prices = prices.iloc[:, 0]
        
        # Create events if not provided
        if events is None:
            events = pd.DataFrame(index=prices.index)
        
        # Set default profit-taking and stop-loss
        if pt_sl is None:
            pt_sl = (0.02, -0.02)  # 2% profit taking, 2% stop loss
        
        # Create vertical barriers if not provided
        if t1 is None and vertical_barrier_times is None:
            # Default: 5-day barrier
            t1 = pd.Series(index=events.index, dtype='datetime64[ns]')
            for idx in events.index:
                if isinstance(idx, pd.Timestamp):
                    # Add 5 business days
                    barrier_time = idx + pd.Timedelta(days=5)
                    if barrier_time <= prices.index.max():
                        t1.loc[idx] = barrier_time
                    else:
                        t1.loc[idx] = prices.index.max()
        elif vertical_barrier_times is not None:
            t1 = vertical_barrier_times
        
        # TODO: Implement full triple barrier logic
        # For now, create a simplified version
        labels = []
        
        for event_time in events.index:
            if event_time not in prices.index:
                continue
                
            start_price = prices.loc[event_time]
            end_time = t1.loc[event_time] if event_time in t1.index else event_time
            
            # Find prices between event and barrier
            price_slice = prices.loc[event_time:end_time]
            
            if len(price_slice) <= 1:
                continue
            
            # Calculate returns from start
            returns = (price_slice / start_price) - 1
            
            # Check barriers
            profit_hit = returns >= pt_sl[0]
            loss_hit = returns <= pt_sl[1]
            
            # Determine first barrier hit
            if profit_hit.any():
                first_profit = profit_hit.idxmax() if profit_hit.any() else None
            else:
                first_profit = None
                
            if loss_hit.any():
                first_loss = loss_hit.idxmax() if loss_hit.any() else None
            else:
                first_loss = None
            
            # Determine label
            if first_profit and first_loss:
                # Both hit, use first one
                if first_profit <= first_loss:
                    label_bin = 1  # Profit taking
                    label_t1 = first_profit
                    label_ret = returns.loc[first_profit]
                else:
                    label_bin = -1  # Stop loss
                    label_t1 = first_loss
                    label_ret = returns.loc[first_loss]
            elif first_profit:
                label_bin = 1
                label_t1 = first_profit
                label_ret = returns.loc[first_profit]
            elif first_loss:
                label_bin = -1
                label_t1 = first_loss
                label_ret = returns.loc[first_loss]
            else:
                # Vertical barrier (time out)
                label_bin = 0
                label_t1 = end_time
                label_ret = returns.iloc[-1]
            
            # Only include if return meets minimum threshold
            if abs(label_ret) >= min_ret:
                labels.append({
                    't1': label_t1,
                    'ret': label_ret,
                    'bin': label_bin
                })
        
        # Convert to DataFrame
        result_df = pd.DataFrame(labels, index=events.index[:len(labels)])
        
        logger.info(f"Generated triple barrier labels: {len(result_df)} events, "
                   f"{(result_df['bin'] == 1).sum()} profits, "
                   f"{(result_df['bin'] == -1).sum()} losses, "
                   f"{(result_df['bin'] == 0).sum()} timeouts")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error in triple barrier labeling: {e}")
        return pd.DataFrame(columns=['t1', 'ret', 'bin'])


def get_daily_vol(prices: pd.Series, span0: int = 100) -> pd.Series:
    """
    Calculate daily volatility for barrier calibration.
    
    Args:
        prices: Price series
        span0: Span for EWMA calculation
        
    Returns:
        Daily volatility series
    """
    try:
        # Calculate daily returns
        daily_ret = prices.resample('1D').last().pct_change().dropna()
        
        # Calculate EWMA volatility
        vol = daily_ret.ewm(span=span0).std()
        
        return vol
        
    except Exception as e:
        logger.error(f"Error calculating daily volatility: {e}")
        return pd.Series(dtype=float)