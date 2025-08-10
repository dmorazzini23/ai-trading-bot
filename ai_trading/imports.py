"""
Centralized Import Management for AI Trading Dependencies

This module provides graceful fallbacks for dependencies that may not be available
in testing environments, ensuring the bot can function in minimal setups.

AI-AGENT-REF: Centralized dependency imports with comprehensive fallbacks
"""

from __future__ import annotations

import logging
from datetime import UTC
from typing import Any

# Initialize logger for import warnings
logger = logging.getLogger(__name__)

# Availability flags
NUMPY_AVAILABLE = False
PANDAS_AVAILABLE = False
SKLEARN_AVAILABLE = False
TA_AVAILABLE = False
TALIB_AVAILABLE = False
PANDAS_TA_AVAILABLE = False

# ============================================================================
# NumPy Import and Mock Implementation
# ============================================================================

try:
    import numpy as np

    NUMPY_AVAILABLE = True
    logger.debug("NumPy imported successfully")
except ImportError:
    logger.warning("NumPy not available, using MockNumpy fallback")

    class MockNumpy:
        """Mock NumPy implementation for testing environments."""

        def __init__(self):
            # Constants
            self.nan = float("nan")
            self.inf = float("inf")
            self.pi = 3.141592653589793
            self.e = 2.718281828459045

            # Data types
            self.float64 = float
            self.int32 = int
            self.bool_ = bool

            self.random = self._MockRandom()

        # Constants as class attributes for backward compatibility
        nan = float("nan")
        inf = float("inf")
        pi = 3.141592653589793
        e = 2.718281828459045

        # Data types as class attributes
        float64 = float
        int32 = int
        bool_ = bool

        class _MockRandom:
            def seed(self, seed: int) -> None:
                pass

            def rand(self, *shape) -> float | list:
                if not shape:
                    return 0.5
                return [0.5] * shape[0] if len(shape) == 1 else [[0.5]]

            def randn(self, *shape) -> float | list:
                return self.rand(*shape)

            def normal(self, loc=0.0, scale=1.0, size=None):
                if size is None:
                    return loc
                return [loc] * size if isinstance(size, int) else [[loc]]

        # Array operations
        def array(self, data: Any, dtype: Any = None) -> list:
            if isinstance(data, list | tuple):
                return list(data)
            return [data]

        def zeros(self, shape: int | tuple) -> list:
            if isinstance(shape, int):
                return [0.0] * shape
            return [[0.0] * shape[1] for _ in range(shape[0])]

        def ones(self, shape: int | tuple) -> list:
            if isinstance(shape, int):
                return [1.0] * shape
            return [[1.0] * shape[1] for _ in range(shape[0])]

        def full(self, shape: int | tuple, fill_value: float) -> list:
            if isinstance(shape, int):
                return [fill_value] * shape
            return [[fill_value] * shape[1] for _ in range(shape[0])]

        # Mathematical operations
        def mean(self, data: Any, axis: int | None = None) -> float:
            if isinstance(data, list | tuple) and data:
                return sum(data) / len(data)
            return 0.0

        def std(self, data: Any, axis: int | None = None) -> float:
            return 1.0

        def sum(self, data: Any, axis: int | None = None) -> float:
            if isinstance(data, list | tuple):
                return sum(data)
            return 0.0

        def max(self, data: Any, axis: int | None = None) -> float:
            if isinstance(data, list | tuple) and data:
                return max(data)
            return 0.0

        def min(self, data: Any, axis: int | None = None) -> float:
            if isinstance(data, list | tuple) and data:
                return min(data)
            return 0.0

        # Logical operations
        def isnan(self, data: Any) -> bool:
            try:
                return data != data  # NaN != NaN is True
            except Exception:
                import logging

                logger = logging.getLogger(__name__)
                logger.debug("Mock numpy helper failed", exc_info=True)
                return False

        def isinf(self, data: Any) -> bool:
            return False

        def isfinite(self, data: Any) -> bool:
            return True

        # Mathematical functions
        def abs(self, data: Any) -> Any:
            if isinstance(data, list | tuple):
                return [abs(x) for x in data]
            return abs(data)

        def sqrt(self, data: Any) -> Any:
            if isinstance(data, list | tuple):
                return [x**0.5 for x in data]
            return data**0.5

        def log(self, data: Any) -> Any:
            import math

            if isinstance(data, list | tuple):
                return [math.log(max(x, 1e-10)) for x in data]
            return math.log(max(data, 1e-10))

        def exp(self, data: Any) -> Any:
            import math

            if isinstance(data, list | tuple):
                return [math.exp(min(x, 700)) for x in data]  # Prevent overflow
            return math.exp(min(data, 700))

        # Array manipulation
        def concatenate(self, arrays: list, axis: int = 0) -> list:
            if not arrays:
                return []
            result = []
            for arr in arrays:
                if isinstance(arr, list | tuple):
                    result.extend(arr)
                else:
                    result.append(arr)
            return result

        def reshape(self, data: Any, shape: tuple) -> Any:
            return data  # Simplified

        def transpose(self, data: Any) -> Any:
            return data  # Simplified

    np = MockNumpy()

# ============================================================================
# Pandas Import and Mock Implementation
# ============================================================================

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
    logger.debug("Pandas imported successfully")
except ImportError:
    logger.warning("Pandas not available, using MockPandas fallback")

    from datetime import datetime

    class MockSeries:
        """Mock Pandas Series implementation."""

        def __init__(self, data: Any = None, index: Any = None, name: Any = None):
            self.data = (
                data
                if isinstance(data, list | tuple)
                else [data] if data is not None else []
            )
            self.index = index or list(range(len(self.data)))
            self.name = name

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, key):
            if isinstance(key, int):
                return self.data[key] if 0 <= key < len(self.data) else None
            return MockSeries([self.data[i] for i in key if 0 <= i < len(self.data)])

        def mean(self) -> float:
            return sum(self.data) / len(self.data) if self.data else 0.0

        def std(self) -> float:
            return 1.0

        def sum(self) -> float:
            return sum(self.data) if self.data else 0.0

        def max(self) -> float:
            return max(self.data) if self.data else 0.0

        def min(self) -> float:
            return min(self.data) if self.data else 0.0

        def shift(self, periods: int = 1) -> MockSeries:
            if periods > 0:
                shifted_data = [None] * periods + self.data[:-periods]
            elif periods < 0:
                shifted_data = self.data[abs(periods) :] + [None] * abs(periods)
            else:
                shifted_data = self.data.copy()
            return MockSeries(shifted_data, self.index, self.name)

        def rolling(self, window: int) -> MockRolling:
            return MockRolling(self, window)

        def dropna(self) -> MockSeries:
            return MockSeries([x for x in self.data if x is not None], name=self.name)

        def fillna(self, value: Any) -> MockSeries:
            return MockSeries(
                [value if x is None else x for x in self.data], name=self.name
            )

        def values(self) -> list:
            return self.data

        def tolist(self) -> list:
            return self.data

    class MockRolling:
        """Mock Pandas Rolling implementation."""

        def __init__(self, series: MockSeries, window: int):
            self.series = series
            self.window = window

        def mean(self) -> MockSeries:
            result = []
            for i in range(len(self.series.data)):
                start = max(0, i - self.window + 1)
                window_data = self.series.data[start : i + 1]
                if len(window_data) >= self.window:
                    result.append(sum(window_data) / len(window_data))
                else:
                    result.append(None)
            return MockSeries(result, self.series.index, self.series.name)

        def std(self) -> MockSeries:
            return MockSeries(
                [1.0] * len(self.series.data), self.series.index, self.series.name
            )

    class MockDataFrame:
        """Mock Pandas DataFrame implementation."""

        def __init__(self, data: Any = None, index: Any = None, columns: Any = None):
            if isinstance(data, dict):
                self.data = data
                self.columns = list(data.keys()) if columns is None else columns
            elif isinstance(data, list | tuple):
                self.columns = columns or [
                    f"col_{i}" for i in range(len(data[0]) if data else 0)
                ]
                self.data = (
                    {
                        col: [row[i] if i < len(row) else None for row in data]
                        for i, col in enumerate(self.columns)
                    }
                    if data
                    else {}
                )
            else:
                self.data = {}
                self.columns = columns or []

            self.index = index or list(range(len(next(iter(self.data.values()), []))))

        def __len__(self) -> int:
            return len(self.index)

        def __getitem__(self, key):
            if isinstance(key, str):
                return MockSeries(self.data.get(key, []), self.index, key)
            elif isinstance(key, list | tuple):
                return MockDataFrame(
                    {k: self.data.get(k, []) for k in key}, self.index, key
                )
            return MockDataFrame()

        def __setitem__(self, key: str, value):
            if isinstance(value, MockSeries):
                self.data[key] = value.data
            elif isinstance(value, list | tuple):
                self.data[key] = value
            else:
                self.data[key] = [value] * len(self.index)
            if key not in self.columns:
                self.columns.append(key)

        @property
        def empty(self) -> bool:
            return len(self.data) == 0 or len(self.index) == 0

        @property
        def shape(self) -> tuple[int, int]:
            return (len(self.index), len(self.columns))

        def head(self, n: int = 5) -> MockDataFrame:
            subset_data = {col: values[:n] for col, values in self.data.items()}
            return MockDataFrame(subset_data, self.index[:n], self.columns)

        def tail(self, n: int = 5) -> MockDataFrame:
            subset_data = {col: values[-n:] for col, values in self.data.items()}
            return MockDataFrame(subset_data, self.index[-n:], self.columns)

        def copy(self) -> MockDataFrame:
            return MockDataFrame(
                self.data.copy(), self.index.copy(), self.columns.copy()
            )

        def dropna(self) -> MockDataFrame:
            return self  # Simplified

        def fillna(self, value: Any) -> MockDataFrame:
            filled_data = {}
            for col, values in self.data.items():
                filled_data[col] = [value if v is None else v for v in values]
            return MockDataFrame(filled_data, self.index, self.columns)

        def mean(self) -> MockSeries:
            means = {}
            for col, values in self.data.items():
                numeric_values = [v for v in values if isinstance(v, int | float)]
                means[col] = (
                    sum(numeric_values) / len(numeric_values) if numeric_values else 0.0
                )
            return MockSeries(list(means.values()), list(means.keys()))

        def std(self) -> MockSeries:
            return MockSeries([1.0] * len(self.columns), self.columns)

        def to_dict(self) -> dict:
            return self.data.copy()

        def values(self) -> list[list]:
            return [
                [self.data[col][i] for col in self.columns]
                for i in range(len(self.index))
            ]

    class MockPandas:
        """Mock Pandas module implementation."""

        DataFrame = MockDataFrame
        Series = MockSeries
        Timestamp = datetime

        def read_csv(self, *args, **kwargs) -> MockDataFrame:
            return MockDataFrame()

        def read_parquet(self, *args, **kwargs) -> MockDataFrame:
            return MockDataFrame()

        def concat(self, objs: list, *args, **kwargs) -> MockDataFrame:
            if not objs:
                return MockDataFrame()
            return objs[0] if objs else MockDataFrame()

        def merge(self, left, right, *args, **kwargs):
            return left if hasattr(left, "empty") else MockDataFrame()

        def to_datetime(self, *args, **kwargs):
            return datetime.now(UTC)

        def date_range(self, *args, **kwargs):
            return [datetime.now(UTC)]

    pd = MockPandas()

# ============================================================================
# Scikit-learn Import and Mock Implementation
# ============================================================================

try:
    import sklearn
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
    logger.debug("Scikit-learn imported successfully")
except ImportError:
    logger.warning("Scikit-learn not available, using MockSklearn fallback")

    class BaseEstimator:
        """Mock sklearn BaseEstimator."""

    class TransformerMixin:
        """Mock sklearn TransformerMixin."""

    class MockLinearRegression(BaseEstimator):
        """Mock LinearRegression implementation."""

        def __init__(self):
            self.coef_ = []
            self.intercept_ = 0.0

        def fit(self, X, y):
            return self

        def predict(self, X):
            if hasattr(X, "__len__"):
                return [0.0] * len(X)
            return [0.0]

        def score(self, X, y):
            return 0.5

    class MockRandomForestRegressor(BaseEstimator):
        """Mock RandomForestRegressor implementation."""

        def __init__(self, *args, **kwargs):
            self.feature_importances_ = []

        def fit(self, X, y):
            return self

        def predict(self, X):
            if hasattr(X, "__len__"):
                return [0.0] * len(X)
            return [0.0]

        def score(self, X, y):
            return 0.5

    class MockStandardScaler(BaseEstimator, TransformerMixin):
        """Mock StandardScaler implementation."""

        def __init__(self):
            self.mean_ = 0.0
            self.scale_ = 1.0

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

        def fit_transform(self, X, y=None):
            return self.fit(X, y).transform(X)

        def inverse_transform(self, X):
            return X

    def train_test_split(*arrays, **kwargs):
        """Mock train_test_split function."""
        test_size = kwargs.get("test_size", 0.25)
        train_size = 1.0 - test_size

        if len(arrays) == 2:
            X, y = arrays
            split_idx = int(len(X) * train_size) if hasattr(X, "__len__") else 0
            return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

        # Return empty splits for simplicity
        return [[] for _ in range(len(arrays) * 2)]

    def mean_squared_error(y_true, y_pred):
        """Mock mean_squared_error function."""
        return 0.0

    def r2_score(y_true, y_pred):
        """Mock r2_score function."""
        return 0.5

    # Assign mock classes to module names
    LinearRegression = MockLinearRegression
    RandomForestRegressor = MockRandomForestRegressor
    StandardScaler = MockStandardScaler

    # Mock sklearn module
    class MockSklearn:
        class base:
            BaseEstimator = BaseEstimator
            TransformerMixin = TransformerMixin

        class linear_model:
            LinearRegression = MockLinearRegression

        class ensemble:
            RandomForestRegressor = MockRandomForestRegressor

        class preprocessing:
            StandardScaler = MockStandardScaler

        class model_selection:
            train_test_split = train_test_split

        class metrics:
            mean_squared_error = mean_squared_error
            r2_score = r2_score

    sklearn = MockSklearn()

# ============================================================================
# Technical Analysis Libraries Import and Mock Implementation
# ============================================================================

# Try ta library first (preferred for compatibility)
TA_AVAILABLE = False
try:
    import ta

    TA_AVAILABLE = True
    logger.debug("TA library imported successfully")
except ImportError:
    logger.warning("TA library not available, will try pandas-ta fallback")
    ta = None

# Try pandas-ta as fallback
if not TA_AVAILABLE:
    try:
        import pandas_ta as ta

        PANDAS_TA_AVAILABLE = True
        logger.debug("pandas-ta imported successfully as fallback")
    except ImportError:
        logger.warning("pandas-ta not available, will try TA-Lib fallback")
        ta = None

# Try TA-Lib as final fallback
if not TA_AVAILABLE and not PANDAS_TA_AVAILABLE:
    try:
        import talib

        TALIB_AVAILABLE = True
        logger.debug("TA-Lib imported successfully as final fallback")
    except ImportError:
        logger.warning("TA-Lib not available, using MockTalib fallback")
        talib = None

# Create unified TA interface with mocks if needed
if not TA_AVAILABLE and not PANDAS_TA_AVAILABLE and not TALIB_AVAILABLE:

    class MockTalib:
        """Mock Technical Analysis library implementation."""

        @staticmethod
        def SMA(data, timeperiod=20):
            """Simple Moving Average."""
            if not hasattr(data, "__len__") or len(data) < timeperiod:
                return [None] * len(data) if hasattr(data, "__len__") else None

            result = [None] * (timeperiod - 1)
            for i in range(timeperiod - 1, len(data)):
                window = data[i - timeperiod + 1 : i + 1]
                result.append(sum(window) / len(window))
            return result

        @staticmethod
        def EMA(data, timeperiod=20):
            """Exponential Moving Average."""
            if not hasattr(data, "__len__") or not data:
                return []

            alpha = 2.0 / (timeperiod + 1)
            result = [data[0]]

            for i in range(1, len(data)):
                ema = alpha * data[i] + (1 - alpha) * result[-1]
                result.append(ema)

            return result

        @staticmethod
        def RSI(data, timeperiod=14):
            """Relative Strength Index."""
            if not hasattr(data, "__len__") or len(data) < timeperiod + 1:
                return [50.0] * len(data) if hasattr(data, "__len__") else 50.0

            # Simplified RSI calculation
            result = [None] * timeperiod
            for i in range(timeperiod, len(data)):
                # Mock RSI oscillating around 50
                result.append(50.0 + (i % 20 - 10) * 2)

            return result

        @staticmethod
        def MACD(data, fastperiod=12, slowperiod=26, signalperiod=9):
            """Moving Average Convergence Divergence."""
            if not hasattr(data, "__len__"):
                return [], [], []

            # Mock MACD values
            length = len(data)
            macd = [0.0] * length
            signal = [0.0] * length
            histogram = [0.0] * length

            for i in range(length):
                # Simple mock oscillation
                macd[i] = (i % 20 - 10) * 0.1
                signal[i] = macd[i] * 0.8
                histogram[i] = macd[i] - signal[i]

            return macd, signal, histogram

        @staticmethod
        def BBANDS(data, timeperiod=20, nbdevup=2, nbdevdn=2):
            """Bollinger Bands."""
            if not hasattr(data, "__len__"):
                return [], [], []

            sma = MockTalib.SMA(data, timeperiod)
            upper = [val + 2.0 if val is not None else None for val in sma]
            lower = [val - 2.0 if val is not None else None for val in sma]

            return upper, sma, lower

        @staticmethod
        def ATR(high, low, close, timeperiod=14):
            """Average True Range."""
            if not all(hasattr(x, "__len__") for x in [high, low, close]):
                return []

            length = min(len(high), len(low), len(close))
            return [1.0] * length  # Mock constant ATR

        @staticmethod
        def STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3):
            """Stochastic Oscillator."""
            if not all(hasattr(x, "__len__") for x in [high, low, close]):
                return [], []

            length = min(len(high), len(low), len(close))
            slowk = [(i % 100) for i in range(length)]  # Mock %K
            slowd = [(i % 100) for i in range(length)]  # Mock %D

            return slowk, slowd

    # Set the mock as our TA library
    talib = MockTalib()
    ta = MockTalib()  # Also available as 'ta' for pandas-ta compatibility
else:
    # Create a compatibility layer for the ta library to provide TA-Lib interface
    if TA_AVAILABLE:

        class TalibCompatLayer:
            """Compatibility layer to provide TA-Lib interface using ta library."""

            @staticmethod
            def SMA(close, timeperiod=30):
                """Simple Moving Average - TA-Lib compatible interface."""
                if not hasattr(close, "__iter__"):
                    return float("nan")

                # Convert to pandas Series if needed
                if not hasattr(close, "rolling"):  # Check for pandas Series methods
                    import pandas as pd

                    close = pd.Series(close)

                result = ta.trend.sma_indicator(close, window=timeperiod, fillna=False)
                return result.tolist() if hasattr(result, "tolist") else result

            @staticmethod
            def EMA(close, timeperiod=30):
                """Exponential Moving Average - TA-Lib compatible interface."""
                if not hasattr(close, "__iter__"):
                    return float("nan")

                if not hasattr(close, "rolling"):
                    import pandas as pd

                    close = pd.Series(close)

                result = ta.trend.ema_indicator(close, window=timeperiod, fillna=False)
                return result.tolist() if hasattr(result, "tolist") else result

            @staticmethod
            def RSI(close, timeperiod=14):
                """Relative Strength Index - TA-Lib compatible interface."""
                if not hasattr(close, "__iter__"):
                    return float("nan")

                if not hasattr(close, "rolling"):
                    import pandas as pd

                    close = pd.Series(close)

                result = ta.momentum.rsi(close, window=timeperiod, fillna=False)
                return result.tolist() if hasattr(result, "tolist") else result

            @staticmethod
            def MACD(close, fastperiod=12, slowperiod=26, signalperiod=9):
                """MACD - TA-Lib compatible interface."""
                if not hasattr(close, "__iter__"):
                    empty = [float("nan")]
                    return empty, empty, empty

                if not hasattr(close, "rolling"):
                    import pandas as pd

                    close = pd.Series(close)

                macd_line = ta.trend.macd(
                    close, window_slow=slowperiod, window_fast=fastperiod, fillna=False
                )
                signal_line = ta.trend.macd_signal(
                    close,
                    window_slow=slowperiod,
                    window_fast=fastperiod,
                    window_sign=signalperiod,
                    fillna=False,
                )
                histogram = ta.trend.macd_diff(
                    close,
                    window_slow=slowperiod,
                    window_fast=fastperiod,
                    window_sign=signalperiod,
                    fillna=False,
                )

                macd_list = (
                    macd_line.tolist() if hasattr(macd_line, "tolist") else macd_line
                )
                signal_list = (
                    signal_line.tolist()
                    if hasattr(signal_line, "tolist")
                    else signal_line
                )
                hist_list = (
                    histogram.tolist() if hasattr(histogram, "tolist") else histogram
                )

                return macd_list, signal_list, hist_list

            @staticmethod
            def BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
                """Bollinger Bands - TA-Lib compatible interface."""
                if not hasattr(close, "__iter__"):
                    empty = [float("nan")]
                    return empty, empty, empty

                if not hasattr(close, "rolling"):
                    import pandas as pd

                    close = pd.Series(close)

                upper = ta.volatility.bollinger_hband(
                    close, window=timeperiod, window_dev=nbdevup, fillna=False
                )
                middle = ta.trend.sma_indicator(close, window=timeperiod, fillna=False)
                lower = ta.volatility.bollinger_lband(
                    close, window=timeperiod, window_dev=nbdevdn, fillna=False
                )

                upper_list = upper.tolist() if hasattr(upper, "tolist") else upper
                middle_list = middle.tolist() if hasattr(middle, "tolist") else middle
                lower_list = lower.tolist() if hasattr(lower, "tolist") else lower

                return upper_list, middle_list, lower_list

            @staticmethod
            def ATR(high, low, close, timeperiod=14):
                """Average True Range - TA-Lib compatible interface."""
                if not hasattr(close, "__iter__"):
                    return [float("nan")]

                if not hasattr(close, "rolling"):
                    import pandas as pd

                    high = pd.Series(high)
                    low = pd.Series(low)
                    close = pd.Series(close)

                result = ta.volatility.average_true_range(
                    high, low, close, window=timeperiod, fillna=False
                )
                return result.tolist() if hasattr(result, "tolist") else result

            @staticmethod
            def STOCH(
                high,
                low,
                close,
                fastk_period=5,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=3,
                slowd_matype=0,
            ):
                """Stochastic Oscillator - TA-Lib compatible interface."""
                if not hasattr(close, "__iter__"):
                    empty = [float("nan")]
                    return empty, empty

                if not hasattr(close, "rolling"):
                    import pandas as pd

                    high = pd.Series(high)
                    low = pd.Series(low)
                    close = pd.Series(close)

                # Map TA-Lib parameters to ta library parameters
                window = fastk_period
                smooth_window = slowk_period

                slowk = ta.momentum.stoch(
                    high,
                    low,
                    close,
                    window=window,
                    smooth_window=smooth_window,
                    fillna=False,
                )
                slowd = ta.momentum.stoch_signal(
                    high,
                    low,
                    close,
                    window=window,
                    smooth_window=smooth_window,
                    fillna=False,
                )

                slowk_list = slowk.tolist() if hasattr(slowk, "tolist") else slowk
                slowd_list = slowd.tolist() if hasattr(slowd, "tolist") else slowd

                return slowk_list, slowd_list

            # Enhanced indicators from ta library
            @staticmethod
            def ADX(high, low, close, timeperiod=14):
                """Average Directional Index - TA-Lib compatible interface."""
                if not hasattr(close, "__iter__"):
                    return [float("nan")]

                if not hasattr(close, "rolling"):
                    import pandas as pd

                    high = pd.Series(high)
                    low = pd.Series(low)
                    close = pd.Series(close)

                result = ta.trend.adx(high, low, close, window=timeperiod, fillna=False)
                return result.tolist() if hasattr(result, "tolist") else result

            @staticmethod
            def WILLR(high, low, close, timeperiod=14):
                """Williams %R - TA-Lib compatible interface."""
                if not hasattr(close, "__iter__"):
                    return [float("nan")]

                if not hasattr(close, "rolling"):
                    import pandas as pd

                    high = pd.Series(high)
                    low = pd.Series(low)
                    close = pd.Series(close)

                result = ta.momentum.williams_r(
                    high, low, close, lbp=timeperiod, fillna=False
                )
                return result.tolist() if hasattr(result, "tolist") else result

            @staticmethod
            def CCI(high, low, close, timeperiod=20):
                """Commodity Channel Index - TA-Lib compatible interface."""
                if not hasattr(close, "__iter__"):
                    return [float("nan")]

                if not hasattr(close, "rolling"):
                    import pandas as pd

                    high = pd.Series(high)
                    low = pd.Series(low)
                    close = pd.Series(close)

                result = ta.trend.cci(high, low, close, window=timeperiod, fillna=False)
                return result.tolist() if hasattr(result, "tolist") else result

            @staticmethod
            def OBV(close, volume):
                """On-Balance Volume - TA-Lib compatible interface."""
                if not hasattr(close, "__iter__"):
                    return [float("nan")]

                if not hasattr(close, "rolling"):
                    import pandas as pd

                    close = pd.Series(close)
                    volume = pd.Series(volume)

                result = ta.volume.on_balance_volume(close, volume, fillna=False)
                return result.tolist() if hasattr(result, "tolist") else result

            @staticmethod
            def VWAP(high, low, close, volume, timeperiod=14):
                """Volume Weighted Average Price - TA-Lib compatible interface."""
                if not hasattr(close, "__iter__"):
                    return [float("nan")]

                if not hasattr(close, "rolling"):
                    import pandas as pd

                    high = pd.Series(high)
                    low = pd.Series(low)
                    close = pd.Series(close)
                    volume = pd.Series(volume)

                result = ta.volume.volume_weighted_average_price(
                    high, low, close, volume, window=timeperiod, fillna=False
                )
                return result.tolist() if hasattr(result, "tolist") else result

            @staticmethod
            def AD(high, low, close, volume):
                """Accumulation/Distribution Line - TA-Lib compatible interface."""
                if not hasattr(close, "__iter__"):
                    return [float("nan")]

                if not hasattr(close, "rolling"):
                    import pandas as pd

                    high = pd.Series(high)
                    low = pd.Series(low)
                    close = pd.Series(close)
                    volume = pd.Series(volume)

                result = ta.volume.acc_dist_index(
                    high, low, close, volume, fillna=False
                )
                return result.tolist() if hasattr(result, "tolist") else result

            @staticmethod
            def BBANDWIDTH(close, timeperiod=20, nbdev=2):
                """Bollinger Band Width - Enhanced indicator."""
                if not hasattr(close, "__iter__"):
                    return [float("nan")]

                if not hasattr(close, "rolling"):
                    import pandas as pd

                    close = pd.Series(close)

                result = ta.volatility.bollinger_wband(
                    close, window=timeperiod, window_dev=nbdev, fillna=False
                )
                return result.tolist() if hasattr(result, "tolist") else result

            @staticmethod
            def DONCHIAN_HIGH(high, timeperiod=20):
                """Donchian Channel Upper Band - Enhanced indicator."""
                if not hasattr(high, "__iter__"):
                    return [float("nan")]

                if not hasattr(high, "rolling"):
                    import pandas as pd

                    high = pd.Series(high)

                result = ta.volatility.donchian_channel_hband(
                    high, high, high, window=timeperiod, fillna=False
                )
                return result.tolist() if hasattr(result, "tolist") else result

            @staticmethod
            def DONCHIAN_LOW(low, timeperiod=20):
                """Donchian Channel Lower Band - Enhanced indicator."""
                if not hasattr(low, "__iter__"):
                    return [float("nan")]

                if not hasattr(low, "rolling"):
                    import pandas as pd

                    low = pd.Series(low)

                result = ta.volatility.donchian_channel_lband(
                    low, low, low, window=timeperiod, fillna=False
                )
                return result.tolist() if hasattr(result, "tolist") else result

        # Use the compatibility layer
        talib = TalibCompatLayer()


# Create unified interface
def get_ta_lib():
    """Get the available technical analysis library."""
    if TA_AVAILABLE:
        return talib  # Return the compatibility layer
    elif PANDAS_TA_AVAILABLE:
        return ta
    elif TALIB_AVAILABLE:
        return talib
    else:
        return talib  # MockTalib


# Ensure talib is always available as a module-level variable
if not TA_AVAILABLE and not PANDAS_TA_AVAILABLE and not TALIB_AVAILABLE:
    # Using MockTalib - make sure it's accessible
    pass
elif PANDAS_TA_AVAILABLE and not TA_AVAILABLE and not TALIB_AVAILABLE:
    # AI-AGENT-REF: Ensure talib is available when using pandas-ta fallback
    talib = ta  # Use pandas-ta as talib for compatibility

# ============================================================================
# Exports and Public Interface
# ============================================================================

__all__ = [
    # Core libraries
    "np",
    "pd",
    "sklearn",
    # Technical analysis
    "ta",
    "talib",
    "get_ta_lib",
    # Availability flags
    "NUMPY_AVAILABLE",
    "PANDAS_AVAILABLE",
    "SKLEARN_AVAILABLE",
    "TA_AVAILABLE",
    "TALIB_AVAILABLE",
    "PANDAS_TA_AVAILABLE",
    # Sklearn components
    "BaseEstimator",
    "TransformerMixin",
    "LinearRegression",
    "RandomForestRegressor",
    "StandardScaler",
    "train_test_split",
    "mean_squared_error",
    "r2_score",
]


# Log summary of what was loaded
def _log_import_summary():
    """Log summary of successfully imported vs mocked dependencies."""
    available = []
    mocked = []

    if NUMPY_AVAILABLE:
        available.append("NumPy")
    else:
        mocked.append("NumPy")

    if PANDAS_AVAILABLE:
        available.append("Pandas")
    else:
        mocked.append("Pandas")

    if SKLEARN_AVAILABLE:
        available.append("Scikit-learn")
    else:
        mocked.append("Scikit-learn")

    if TA_AVAILABLE:
        available.append("TA")
    elif PANDAS_TA_AVAILABLE:
        available.append("pandas-ta")
    elif TALIB_AVAILABLE:
        available.append("TA-Lib")
    else:
        mocked.append("Technical Analysis libraries")

    if available:
        logger.info(f"Successfully imported: {', '.join(available)}")

    if mocked:
        logger.warning(f"Using mock implementations for: {', '.join(mocked)}")


# Call summary logging
_log_import_summary()
