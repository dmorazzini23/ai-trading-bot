"""Performance metrics for trading results with numerical stability."""

from __future__ import annotations

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
    HAS_NUMPY = True
except ImportError:
    HAS_PANDAS = False
    HAS_NUMPY = False
    pd = None
    np = None

def requires_pandas_numpy(func):
    """Decorator to ensure pandas and numpy are available."""
    def wrapper(*args, **kwargs):
        if not HAS_PANDAS or not HAS_NUMPY:
            raise ImportError(f"pandas and numpy required for {func.__name__}")
        return func(*args, **kwargs)
    return wrapper


@requires_pandas_numpy
def compute_basic_metrics(df) -> dict[str, float]:
    """Return Sharpe ratio and max drawdown from ``df`` with a ``return`` column."""
    if "return" not in df:
        return {"sharpe": 0.0, "max_drawdown": 0.0}
    ret = df["return"].astype(float)
    if ret.empty:
        return {"sharpe": 0.0, "max_drawdown": 0.0}
    
    # AI-AGENT-REF: Epsilon-based numerical stability for division by zero protection
    epsilon = 1e-8
    
    # More robust Sharpe calculation with division by zero protection
    mean_return = ret.mean()
    std_return = ret.std()
    
    # Protect against division by zero in Sharpe ratio calculation
    if std_return <= epsilon or pd.isna(std_return):
        sharpe = 0.0
    else:
        # Annual Sharpe ratio assuming 252 trading days
        sharpe = (mean_return / std_return) * np.sqrt(252)
    
    # Safe cumulative calculation with numerical stability
    cumulative = (1 + ret.fillna(0)).cumprod()
    
    # Drawdown calculation with protection against edge cases
    if len(cumulative) == 0:
        max_dd = 0.0
    else:
        drawdown = cumulative.cummax() - cumulative
        max_dd = float(drawdown.max()) if not drawdown.empty else 0.0
    
    return {"sharpe": float(sharpe), "max_drawdown": float(max_dd)}


def compute_advanced_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Compute advanced performance metrics with numerical stability."""
    if "return" not in df:
        return {"sortino": 0.0, "calmar": 0.0, "win_rate": 0.0, "profit_factor": 0.0}
    
    ret = df["return"].astype(float).fillna(0)
    if ret.empty:
        return {"sortino": 0.0, "calmar": 0.0, "win_rate": 0.0, "profit_factor": 0.0}
    
    epsilon = 1e-8
    
    # Sortino ratio (downside deviation)
    downside_returns = ret[ret < 0]
    if len(downside_returns) == 0:
        downside_std = epsilon
    else:
        downside_std = max(downside_returns.std(), epsilon)
    
    sortino = (ret.mean() / downside_std) * np.sqrt(252)
    
    # Calmar ratio (annual return / max drawdown)
    annual_return = ret.mean() * 252
    basic_metrics = compute_basic_metrics(df)
    max_dd = max(basic_metrics["max_drawdown"], epsilon)
    calmar = annual_return / max_dd
    
    # Win rate
    winning_trades = (ret > 0).sum()
    total_trades = len(ret)
    win_rate = winning_trades / max(total_trades, 1) * 100
    
    # Profit factor
    gross_profit = ret[ret > 0].sum()
    gross_loss = abs(ret[ret < 0].sum())
    profit_factor = gross_profit / max(gross_loss, epsilon)
    
    return {
        "sortino": float(sortino),
        "calmar": float(calmar), 
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor)
    }


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with epsilon protection to prevent division by zero."""
    epsilon = 1e-8
    if abs(denominator) < epsilon:
        return default
    return numerator / denominator


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range with numerical stability."""
    if df.empty or not all(col in df.columns for col in ['high', 'low', 'close']):
        return pd.Series(dtype=float)
    
    epsilon = 1e-8
    
    # True Range calculation with epsilon protection
    high = df['high'].ffill()
    low = df['low'].ffill() 
    close = df['close'].ffill()
    
    # Ensure no negative or zero values that could cause numerical instability
    high = np.maximum(high, epsilon)
    low = np.maximum(low, epsilon)
    close = np.maximum(close, epsilon)
    
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculation with numerical stability
    atr = true_range.rolling(window=period, min_periods=1).mean()
    
    return atr.fillna(epsilon)  # Fill NaN with epsilon to prevent division by zero
