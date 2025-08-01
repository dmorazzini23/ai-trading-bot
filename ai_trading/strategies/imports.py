"""
Centralized import management for ai_trading modules.

Provides graceful fallbacks for dependencies that may not be available
in testing environments, ensuring the bot can function in minimal setups.
"""

import logging
import os

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
        nan = float('nan')
        inf = float('inf')
        
        def array(self, *args, **kwargs):
            return list(args[0]) if args else []
            
        def mean(self, arr):
            return sum(arr) / len(arr) if arr else 0
            
        def std(self, arr):
            if not arr:
                return 0
            mean_val = sum(arr) / len(arr)
            return (sum((x - mean_val)**2 for x in arr) / len(arr))**0.5
            
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
            return x ** 0.5
            
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
    
    # Import the mock pandas from utils
    import sys
    import os
    try:
        # Try to import from parent directory (utils.py)
        utils_path = os.path.dirname(os.path.dirname(__file__))
        sys.path.insert(0, utils_path)
        from utils import pd
    except ImportError:
        # Create a minimal fallback if utils is not available
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

# TA-Lib fallback
try:
    import talib
    TALIB_AVAILABLE = True
    logger.info("TA-Lib loaded successfully for optimized technical analysis")
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning(
        "TA-Lib not available - using fallback implementation. "
        "For enhanced technical analysis, install with `pip install TA-Lib` "
        "(and system package `libta-lib0-dev`)."
    )
    
    class MockTalib:
        """Mock TA-Lib implementation providing basic technical indicators."""
        
        @staticmethod
        def SMA(prices, timeperiod=30):
            """Simple Moving Average fallback implementation."""
            if not prices or len(prices) < timeperiod:
                return [float('nan')] * len(prices) if prices else []
            
            result = [float('nan')] * (timeperiod - 1)
            for i in range(timeperiod - 1, len(prices)):
                window = prices[i - timeperiod + 1:i + 1]
                result.append(sum(window) / len(window))
            return result
        
        @staticmethod
        def EMA(prices, timeperiod=30):
            """Exponential Moving Average fallback implementation."""
            if not prices:
                return []
            
            result = [float('nan')] * len(prices)
            if len(prices) >= timeperiod:
                # Start with SMA for first value
                first_sma = sum(prices[:timeperiod]) / timeperiod
                result[timeperiod - 1] = first_sma
                
                # Calculate EMA for remaining values
                multiplier = 2.0 / (timeperiod + 1)
                for i in range(timeperiod, len(prices)):
                    result[i] = (prices[i] * multiplier) + (result[i - 1] * (1 - multiplier))
            
            return result
        
        @staticmethod
        def RSI(prices, timeperiod=14):
            """Relative Strength Index fallback implementation."""
            if not prices or len(prices) < timeperiod + 1:
                return [float('nan')] * len(prices) if prices else []
            
            result = [float('nan')] * timeperiod
            gains = []
            losses = []
            
            # Calculate initial gains and losses
            for i in range(1, timeperiod + 1):
                change = prices[i] - prices[i - 1]
                gains.append(max(change, 0))
                losses.append(max(-change, 0))
            
            avg_gain = sum(gains) / timeperiod
            avg_loss = sum(losses) / timeperiod
            
            # Calculate RSI values
            for i in range(timeperiod, len(prices)):
                if i > timeperiod:
                    change = prices[i] - prices[i - 1]
                    gain = max(change, 0)
                    loss = max(-change, 0)
                    avg_gain = (avg_gain * (timeperiod - 1) + gain) / timeperiod
                    avg_loss = (avg_loss * (timeperiod - 1) + loss) / timeperiod
                
                if avg_loss == 0:
                    result.append(100)
                else:
                    rs = avg_gain / avg_loss
                    result.append(100 - (100 / (1 + rs)))
            
            return result
        
        @staticmethod
        def MACD(prices, fastperiod=12, slowperiod=26, signalperiod=9):
            """MACD fallback implementation."""
            if not prices or len(prices) < slowperiod:
                empty = [float('nan')] * len(prices) if prices else []
                return empty, empty, empty
            
            # Calculate EMAs
            ema_fast = MockTalib.EMA(prices, fastperiod)
            ema_slow = MockTalib.EMA(prices, slowperiod)
            
            # Calculate MACD line
            macd_line = []
            for i in range(len(prices)):
                if i < slowperiod - 1:
                    macd_line.append(float('nan'))
                else:
                    macd_line.append(ema_fast[i] - ema_slow[i])
            
            # Calculate signal line (EMA of MACD)
            # Create a clean macd series for signal calculation
            clean_macd = []
            for x in macd_line:
                if isinstance(x, float) and x != x:  # NaN check
                    clean_macd.append(0.0)  # Use 0 for NaN values temporarily
                else:
                    clean_macd.append(x)
            
            signal_line = MockTalib.EMA(clean_macd, signalperiod)
            
            # Restore NaN values where MACD was NaN
            for i in range(len(macd_line)):
                if isinstance(macd_line[i], float) and macd_line[i] != macd_line[i]:  # NaN check
                    signal_line[i] = float('nan')
            
            # Calculate histogram
            histogram = []
            for i in range(len(macd_line)):
                if isinstance(macd_line[i], float) and macd_line[i] != macd_line[i]:  # NaN check
                    histogram.append(float('nan'))
                elif isinstance(signal_line[i], float) and signal_line[i] != signal_line[i]:  # NaN check
                    histogram.append(float('nan'))
                else:
                    histogram.append(macd_line[i] - signal_line[i])
            
            return macd_line, signal_line, histogram
        
        @staticmethod
        def BBANDS(prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
            """Bollinger Bands fallback implementation."""
            if not prices or len(prices) < timeperiod:
                empty = [float('nan')] * len(prices) if prices else []
                return empty, empty, empty
            
            # Calculate SMA (middle band)
            sma = MockTalib.SMA(prices, timeperiod)
            
            # Calculate standard deviation and bands
            upper_band = []
            lower_band = []
            
            for i in range(len(prices)):
                if i < timeperiod - 1:
                    upper_band.append(float('nan'))
                    lower_band.append(float('nan'))
                else:
                    window = prices[i - timeperiod + 1:i + 1]
                    mean_val = sma[i]
                    variance = sum((x - mean_val) ** 2 for x in window) / timeperiod
                    std_dev = variance ** 0.5
                    
                    upper_band.append(mean_val + (nbdevup * std_dev))
                    lower_band.append(mean_val - (nbdevdn * std_dev))
            
            return upper_band, sma, lower_band
        
        @staticmethod
        def ATR(high, low, close, timeperiod=14):
            """Average True Range fallback implementation."""
            if not high or not low or not close or len(high) < 2:
                return [float('nan')] * len(high) if high else []
            
            # Calculate True Range
            true_ranges = [float('nan')]  # First value is NaN
            for i in range(1, len(high)):
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i - 1])
                tr3 = abs(low[i] - close[i - 1])
                true_ranges.append(max(tr1, tr2, tr3))
            
            # Calculate ATR (SMA of True Range)
            atr_values = []
            for i in range(len(true_ranges)):
                if i < timeperiod:
                    atr_values.append(float('nan'))
                else:
                    window = true_ranges[i - timeperiod + 1:i + 1]
                    atr_values.append(sum(window) / len(window))
            
            return atr_values
        
        @staticmethod
        def STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0):
            """Stochastic Oscillator fallback implementation."""
            if not high or not low or not close or len(high) < fastk_period:
                empty = [float('nan')] * len(high) if high else []
                return empty, empty
            
            # Calculate %K
            k_values = []
            for i in range(len(high)):
                if i < fastk_period - 1:
                    k_values.append(float('nan'))
                else:
                    window_high = max(high[i - fastk_period + 1:i + 1])
                    window_low = min(low[i - fastk_period + 1:i + 1])
                    if window_high == window_low:
                        k_values.append(50.0)  # Avoid division by zero
                    else:
                        k_values.append(100 * (close[i] - window_low) / (window_high - window_low))
            
            # Calculate %D (SMA of %K)
            d_values = MockTalib.SMA(k_values, slowd_period)
            
            return k_values, d_values
    
    talib = MockTalib()

# Export commonly used items
__all__ = [
    'np', 'pd', 'metrics', 'RandomForestClassifier', 'train_test_split', 'talib',
    'NUMPY_AVAILABLE', 'PANDAS_AVAILABLE', 'SKLEARN_AVAILABLE', 'TALIB_AVAILABLE'
]