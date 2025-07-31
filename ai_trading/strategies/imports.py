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
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available - using fallback implementation")
    
    class MockTalib:
        @staticmethod
        def SMA(arr, timeperiod=20):
            if len(arr) < timeperiod:
                return [np.nan] * len(arr)
            result = []
            for i in range(len(arr)):
                if i < timeperiod - 1:
                    result.append(np.nan)
                else:
                    result.append(sum(arr[i-timeperiod+1:i+1]) / timeperiod)
            return result
            
        @staticmethod
        def EMA(arr, timeperiod=20):
            return MockTalib.SMA(arr, timeperiod)  # Simplified fallback
            
        @staticmethod
        def RSI(arr, timeperiod=14):
            return [50.0] * len(arr)  # Mock RSI at neutral
            
        @staticmethod
        def MACD(arr, fastperiod=12, slowperiod=26, signalperiod=9):
            mock_line = [0.0] * len(arr)
            mock_signal = [0.0] * len(arr)
            mock_hist = [0.0] * len(arr)
            return mock_line, mock_signal, mock_hist
    
    talib = MockTalib()

# Export commonly used items
__all__ = [
    'np', 'pd', 'metrics', 'RandomForestClassifier', 'train_test_split', 'talib',
    'NUMPY_AVAILABLE', 'PANDAS_AVAILABLE', 'SKLEARN_AVAILABLE', 'TALIB_AVAILABLE'
]