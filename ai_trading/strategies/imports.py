"""
Centralized import management for ai_trading modules.

Provides graceful fallbacks for dependencies that may not be available
in testing environments, ensuring the bot can function in minimal setups.
"""

import logging

logger = logging.getLogger(__name__)

# AI-AGENT-REF: Centralized dependency imports with fallbacks

# NumPy fallback
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available - using fallback implementation")

    class MockNumpy:
        nan = float("nan")
        inf = float("inf")

        def array(self, *args, **kwargs):
            return list(args[0]) if args else []

        def mean(self, arr):
            return sum(arr) / len(arr) if arr else 0

        def std(self, arr):
            if not arr:
                return 0
            mean_val = sum(arr) / len(arr)
            return (sum((x - mean_val) ** 2 for x in arr) / len(arr)) ** 0.5

        def percentile(self, arr, p):
            if not arr:
                return 0
            sorted_arr = sorted(arr)
            idx = int(len(sorted_arr) * p / 100)
            return sorted_arr[min(idx, len(sorted_arr) - 1)]

        def corrcoef(self, x, y=None):
            if y is None:
                return [[1.0]]
            return [[1.0, 0.5], [0.5, 1.0]]  # Mock correlation

        def sqrt(self, x):
            return x**0.5

        def log(self, x):
            import math

            return math.log(x)

        def exp(self, x):
            import math

            return math.exp(x)

        def abs(self, x):
            return abs(x)

        def zeros(self, *args, **kwargs):
            if len(args) == 1:
                return [0] * args[0]
            return []

        def ones(self, *args, **kwargs):
            if len(args) == 1:
                return [1] * args[0]
            return []

        @property
        def random(self):
            import random

            return random

    np = MockNumpy()

# Pandas fallback
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("Pandas not available - using fallback implementation")

    # Use simple fallback
    try:
        # Create a minimal fallback
        from datetime import datetime

        class MockDataFrame:
            def __init__(self, *args, **kwargs):
                pass

            def __len__(self):
                return 0

            def empty(self):
                return True

        class MockSeries:
            def __init__(self, *args, **kwargs):
                pass

            def __len__(self):
                return 0

        class MockPandas:
            DataFrame = MockDataFrame
            Series = MockSeries
            Timestamp = datetime

            def read_csv(self, *args, **kwargs):
                return MockDataFrame()

            def concat(self, *args, **kwargs):
                return MockDataFrame()

        pd = MockPandas()
    except Exception:
        pd = None

# Scikit-learn fallback
try:
    from sklearn import metrics
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available - using fallback implementation")

    class MockSklearn:
        class metrics:
            @staticmethod
            def accuracy_score(y_true, y_pred):
                return 0.5  # Mock accuracy

            @staticmethod
            def classification_report(y_true, y_pred):
                return "Mock classification report"

        class ensemble:
            class RandomForestClassifier:
                def __init__(self, *args, **kwargs):
                    pass

                def fit(self, X, y):
                    return self

                def predict(self, X):
                    return [0] * len(X)

                def predict_proba(self, X):
                    return [[0.5, 0.5]] * len(X)

        class model_selection:
            @staticmethod
            def train_test_split(X, y, test_size=0.2, random_state=None):
                split_idx = int(len(X) * (1 - test_size))
                return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

    sklearn_mock = MockSklearn()
    metrics = sklearn_mock.metrics
    RandomForestClassifier = sklearn_mock.ensemble.RandomForestClassifier
    train_test_split = sklearn_mock.model_selection.train_test_split

# TA library for optimized technical analysis
try:
    import numpy as np
    import pandas as pd
    import ta

    TA_AVAILABLE = True
    logger.info("TA library loaded successfully for enhanced technical analysis")
except ImportError:
    TA_AVAILABLE = False
    logger.warning(
        "TA library not available - using fallback implementation. "
        "For enhanced technical analysis, install with `pip install ta==0.11.0`."
    )

    class MockTa:
        """Mock TA library implementation providing basic technical indicators."""

        class trend:
            @staticmethod
            def sma_indicator(close, window=30, fillna=False):
                """Simple Moving Average using ta library interface."""
                if not hasattr(close, "__iter__"):
                    return float("nan")

                if hasattr(close, "rolling"):  # pandas Series
                    return close.rolling(window=window).mean()

                # List/array fallback
                result = [float("nan")] * len(close)
                for i in range(window - 1, len(close)):
                    window_data = close[i - window + 1 : i + 1]
                    result[i] = sum(window_data) / len(window_data)
                return result

            @staticmethod
            def ema_indicator(close, window=30, fillna=False):
                """Exponential Moving Average using ta library interface."""
                if not hasattr(close, "__iter__"):
                    return float("nan")

                if hasattr(close, "ewm"):  # pandas Series
                    return close.ewm(span=window).mean()

                # List/array fallback - same logic as MockTalib.EMA
                result = [float("nan")] * len(close)
                if len(close) >= window:
                    first_sma = sum(close[:window]) / window
                    result[window - 1] = first_sma

                    multiplier = 2.0 / (window + 1)
                    for i in range(window, len(close)):
                        result[i] = (close[i] * multiplier) + (
                            result[i - 1] * (1 - multiplier)
                        )

                return result

            @staticmethod
            def macd(close, window_slow=26, window_fast=12, fillna=False):
                """MACD line using ta library interface."""
                ema_fast = MockTa.trend.ema_indicator(close, window_fast)
                ema_slow = MockTa.trend.ema_indicator(close, window_slow)

                if hasattr(close, "index"):  # pandas Series
                    import pandas as pd

                    ema_fast_series = pd.Series(ema_fast, index=close.index)
                    ema_slow_series = pd.Series(ema_slow, index=close.index)
                    return ema_fast_series - ema_slow_series

                # List fallback
                return [
                    (
                        f - s
                        if not (isinstance(f, float) and f != f)
                        and not (isinstance(s, float) and s != s)
                        else float("nan")
                    )
                    for f, s in zip(ema_fast, ema_slow, strict=False)
                ]

            @staticmethod
            def macd_signal(
                close, window_slow=26, window_fast=12, window_sign=9, fillna=False
            ):
                """MACD signal line using ta library interface."""
                macd_line = MockTa.trend.macd(close, window_slow, window_fast)
                return MockTa.trend.ema_indicator(macd_line, window_sign)

            @staticmethod
            def macd_diff(
                close, window_slow=26, window_fast=12, window_sign=9, fillna=False
            ):
                """MACD histogram using ta library interface."""
                macd_line = MockTa.trend.macd(close, window_slow, window_fast)
                signal_line = MockTa.trend.macd_signal(
                    close, window_slow, window_fast, window_sign
                )

                if hasattr(close, "index"):  # pandas Series
                    import pandas as pd

                    macd_series = (
                        pd.Series(macd_line, index=close.index)
                        if not hasattr(macd_line, "index")
                        else macd_line
                    )
                    signal_series = (
                        pd.Series(signal_line, index=close.index)
                        if not hasattr(signal_line, "index")
                        else signal_line
                    )
                    return macd_series - signal_series

                # List fallback
                return [
                    (
                        m - s
                        if not (isinstance(m, float) and m != m)
                        and not (isinstance(s, float) and s != s)
                        else float("nan")
                    )
                    for m, s in zip(macd_line, signal_line, strict=False)
                ]

            @staticmethod
            def adx(high, low, close, window=14, fillna=False):
                """Average Directional Index fallback implementation."""
                # Simplified ADX calculation
                if hasattr(close, "rolling"):  # pandas Series
                    return (
                        close.rolling(window=window).std()
                        * 100
                        / close.rolling(window=window).mean()
                    )

                # Simple volatility-based proxy for ADX
                result = [float("nan")] * len(close)
                for i in range(window - 1, len(close)):
                    window_data = close[i - window + 1 : i + 1]
                    mean_val = sum(window_data) / len(window_data)
                    std_val = (
                        sum((x - mean_val) ** 2 for x in window_data) / len(window_data)
                    ) ** 0.5
                    result[i] = (
                        min(100, (std_val * 100 / mean_val)) if mean_val != 0 else 50
                    )
                return result

            @staticmethod
            def cci(high, low, close, window=20, constant=0.015, fillna=False):
                """Commodity Channel Index fallback implementation."""
                # Simplified CCI calculation
                if hasattr(close, "rolling"):  # pandas Series
                    typical_price = (high + low + close) / 3
                    sma = typical_price.rolling(window=window).mean()
                    mad = typical_price.rolling(window=window).apply(
                        lambda x: abs(x - x.mean()).mean()
                    )
                    return (typical_price - sma) / (constant * mad)

                # List fallback
                result = [float("nan")] * len(close)
                for i in range(window - 1, len(close)):
                    typical_prices = [
                        (high[j] + low[j] + close[j]) / 3
                        for j in range(i - window + 1, i + 1)
                    ]
                    sma = sum(typical_prices) / len(typical_prices)
                    mad = sum(abs(tp - sma) for tp in typical_prices) / len(
                        typical_prices
                    )
                    if mad != 0:
                        result[i] = (typical_prices[-1] - sma) / (constant * mad)
                    else:
                        result[i] = 0
                return result

        class momentum:
            @staticmethod
            def rsi(close, window=14, fillna=False):
                """Relative Strength Index using ta library interface."""
                if not hasattr(close, "__iter__"):
                    return float("nan")

                if hasattr(close, "diff"):  # pandas Series
                    delta = close.diff()
                    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
                    loss = (-delta).where(delta < 0, 0).rolling(window=window).mean()
                    rs = gain / loss
                    return 100 - (100 / (1 + rs))

                # List/array fallback - same logic as MockTalib.RSI
                if len(close) < window + 1:
                    return [float("nan")] * len(close)

                result = [float("nan")] * window
                gains = []
                losses = []

                for i in range(1, window + 1):
                    change = close[i] - close[i - 1]
                    gains.append(max(change, 0))
                    losses.append(max(-change, 0))

                avg_gain = sum(gains) / window
                avg_loss = sum(losses) / window

                for i in range(window, len(close)):
                    if i > window:
                        change = close[i] - close[i - 1]
                        gain = max(change, 0)
                        loss = max(-change, 0)
                        avg_gain = (avg_gain * (window - 1) + gain) / window
                        avg_loss = (avg_loss * (window - 1) + loss) / window

                    if avg_loss == 0:
                        result.append(100)
                    else:
                        rs = avg_gain / avg_loss
                        result.append(100 - (100 / (1 + rs)))

                return result

            @staticmethod
            def stoch(high, low, close, k=14, d=3, smooth_k=3, fillna=False):
                """Stochastic Oscillator %K using ta library interface."""
                if hasattr(close, "rolling"):  # pandas Series
                    lowest_low = low.rolling(window=k).min()
                    highest_high = high.rolling(window=k).max()
                    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
                    return k_percent.rolling(window=smooth_k).mean()

                # List fallback
                result = [float("nan")] * len(close)
                for i in range(k - 1, len(close)):
                    window_high = max(high[i - k + 1 : i + 1])
                    window_low = min(low[i - k + 1 : i + 1])
                    if window_high == window_low:
                        result[i] = 50.0
                    else:
                        result[i] = (
                            100 * (close[i] - window_low) / (window_high - window_low)
                        )
                return result

            @staticmethod
            def stoch_signal(high, low, close, k=14, d=3, smooth_k=3, fillna=False):
                """Stochastic Oscillator %D using ta library interface."""
                k_values = MockTa.momentum.stoch(high, low, close, k, d, smooth_k)
                return MockTa.trend.sma_indicator(k_values, d)

            @staticmethod
            def williams_r(high, low, close, lbp=14, fillna=False):
                """Williams %R using ta library interface."""
                if hasattr(close, "rolling"):  # pandas Series
                    highest_high = high.rolling(window=lbp).max()
                    lowest_low = low.rolling(window=lbp).min()
                    return -100 * (highest_high - close) / (highest_high - lowest_low)

                # List fallback
                result = [float("nan")] * len(close)
                for i in range(lbp - 1, len(close)):
                    window_high = max(high[i - lbp + 1 : i + 1])
                    window_low = min(low[i - lbp + 1 : i + 1])
                    if window_high == window_low:
                        result[i] = -50.0
                    else:
                        result[i] = (
                            -100 * (window_high - close[i]) / (window_high - window_low)
                        )
                return result

        class volatility:
            @staticmethod
            def average_true_range(high, low, close, window=14, fillna=False):
                """Average True Range using ta library interface."""
                if hasattr(close, "shift"):  # pandas Series
                    tr1 = high - low
                    tr2 = abs(high - close.shift())
                    tr3 = abs(low - close.shift())
                    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    return true_range.rolling(window=window).mean()

                # List fallback - same logic as MockTalib.ATR
                if len(high) < 2:
                    return [float("nan")] * len(high)

                true_ranges = [float("nan")]
                for i in range(1, len(high)):
                    tr1 = high[i] - low[i]
                    tr2 = abs(high[i] - close[i - 1])
                    tr3 = abs(low[i] - close[i - 1])
                    true_ranges.append(max(tr1, tr2, tr3))

                atr_values = []
                for i in range(len(true_ranges)):
                    if i < window:
                        atr_values.append(float("nan"))
                    else:
                        window_data = true_ranges[i - window + 1 : i + 1]
                        atr_values.append(sum(window_data) / len(window_data))

                return atr_values

            @staticmethod
            def bollinger_hband(close, window=20, window_dev=2, fillna=False):
                """Bollinger Bands upper band using ta library interface."""
                if hasattr(close, "rolling"):  # pandas Series
                    sma = close.rolling(window=window).mean()
                    std = close.rolling(window=window).std()
                    return sma + (std * window_dev)

                # List fallback
                result = [float("nan")] * len(close)
                for i in range(window - 1, len(close)):
                    window_data = close[i - window + 1 : i + 1]
                    mean_val = sum(window_data) / len(window_data)
                    variance = sum((x - mean_val) ** 2 for x in window_data) / len(
                        window_data
                    )
                    std_dev = variance**0.5
                    result[i] = mean_val + (window_dev * std_dev)
                return result

            @staticmethod
            def bollinger_lband(close, window=20, window_dev=2, fillna=False):
                """Bollinger Bands lower band using ta library interface."""
                if hasattr(close, "rolling"):  # pandas Series
                    sma = close.rolling(window=window).mean()
                    std = close.rolling(window=window).std()
                    return sma - (std * window_dev)

                # List fallback
                result = [float("nan")] * len(close)
                for i in range(window - 1, len(close)):
                    window_data = close[i - window + 1 : i + 1]
                    mean_val = sum(window_data) / len(window_data)
                    variance = sum((x - mean_val) ** 2 for x in window_data) / len(
                        window_data
                    )
                    std_dev = variance**0.5
                    result[i] = mean_val - (window_dev * std_dev)
                return result

            @staticmethod
            def bollinger_wband(close, window=20, window_dev=2, fillna=False):
                """Bollinger Band Width using ta library interface."""
                upper = MockTa.volatility.bollinger_hband(close, window, window_dev)
                lower = MockTa.volatility.bollinger_lband(close, window, window_dev)

                if hasattr(close, "index"):  # pandas Series
                    import pandas as pd

                    upper_series = (
                        pd.Series(upper, index=close.index)
                        if not hasattr(upper, "index")
                        else upper
                    )
                    lower_series = (
                        pd.Series(lower, index=close.index)
                        if not hasattr(lower, "index")
                        else lower
                    )
                    return (upper_series - lower_series) / MockTa.trend.sma_indicator(
                        close, window
                    )

                # List fallback
                sma = MockTa.trend.sma_indicator(close, window)
                return [
                    (
                        (u - l) / s
                        if s != 0 and not (isinstance(u, float) and u != u)
                        else float("nan")
                    )
                    for u, l, s in zip(upper, lower, sma, strict=False)
                ]

            @staticmethod
            def donchian_channel_hband(high, window=20, offset=0, fillna=False):
                """Donchian Channel upper band using ta library interface."""
                if hasattr(high, "rolling"):  # pandas Series
                    return high.rolling(window=window).max()

                # List fallback
                result = [float("nan")] * len(high)
                for i in range(window - 1, len(high)):
                    result[i] = max(high[i - window + 1 : i + 1])
                return result

            @staticmethod
            def donchian_channel_lband(low, window=20, offset=0, fillna=False):
                """Donchian Channel lower band using ta library interface."""
                if hasattr(low, "rolling"):  # pandas Series
                    return low.rolling(window=window).min()

                # List fallback
                result = [float("nan")] * len(low)
                for i in range(window - 1, len(low)):
                    result[i] = min(low[i - window + 1 : i + 1])
                return result

        class volume:
            @staticmethod
            def on_balance_volume(close, volume, fillna=False):
                """On-Balance Volume using ta library interface."""
                if hasattr(close, "shift"):  # pandas Series
                    price_change = close.diff()
                    obv = (
                        volume
                        * (
                            (price_change > 0).astype(int)
                            - (price_change < 0).astype(int)
                        )
                    ).cumsum()
                    return obv

                # List fallback
                result = [0]  # Start with 0
                for i in range(1, len(close)):
                    if close[i] > close[i - 1]:
                        result.append(result[-1] + volume[i])
                    elif close[i] < close[i - 1]:
                        result.append(result[-1] - volume[i])
                    else:
                        result.append(result[-1])
                return result

            @staticmethod
            def volume_weighted_average_price(
                high, low, close, volume, window=14, fillna=False
            ):
                """Volume Weighted Average Price using ta library interface."""
                if hasattr(close, "rolling"):  # pandas Series
                    typical_price = (high + low + close) / 3
                    return (typical_price * volume).rolling(
                        window=window
                    ).sum() / volume.rolling(window=window).sum()

                # List fallback
                result = [float("nan")] * len(close)
                for i in range(window - 1, len(close)):
                    window_tp = [
                        (high[j] + low[j] + close[j]) / 3
                        for j in range(i - window + 1, i + 1)
                    ]
                    window_vol = volume[i - window + 1 : i + 1]
                    total_tpv = sum(
                        tp * vol for tp, vol in zip(window_tp, window_vol, strict=False)
                    )
                    total_vol = sum(window_vol)
                    result[i] = (
                        total_tpv / total_vol if total_vol != 0 else float("nan")
                    )
                return result

            @staticmethod
            def acc_dist_index(high, low, close, volume, fillna=False):
                """Accumulation/Distribution Index using ta library interface."""
                if hasattr(close, "shift"):  # pandas Series
                    money_flow_multiplier = ((close - low) - (high - close)) / (
                        high - low
                    )
                    money_flow_volume = money_flow_multiplier * volume
                    return money_flow_volume.cumsum()

                # List fallback
                result = [0]  # Start with 0
                for i in range(len(close)):
                    if high[i] == low[i]:
                        mfm = 0
                    else:
                        mfm = ((close[i] - low[i]) - (high[i] - close[i])) / (
                            high[i] - low[i]
                        )
                    mfv = mfm * volume[i]
                    if i == 0:
                        result[0] = mfv
                    else:
                        result.append(result[-1] + mfv)
                return result

    ta = MockTa()

# Export commonly used items
__all__ = [
    "np",
    "pd",
    "metrics",
    "RandomForestClassifier",
    "train_test_split",
    "ta",
    "NUMPY_AVAILABLE",
    "PANDAS_AVAILABLE",
    "SKLEARN_AVAILABLE",
    "TA_AVAILABLE",
]
