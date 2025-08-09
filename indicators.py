# Compatibility shim + minimal indicator used by tests
try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

def mean_reversion_zscore(series, window: int = 20, min_periods: int = 5):
    if pd is None or np is None:
        return None
    s = pd.Series(series, dtype="float64")
    m = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()
    z = (s - m) / sd.replace(0.0, np.nan)
    return z

# Re-export from ai_trading.indicators if available
try:
    from ai_trading.indicators import *  # re-export
except ImportError:
    pass