# Centralized Import Management

The AI Trading Bot now includes a centralized import management system that provides graceful fallbacks for dependencies that may not be available in testing or minimal environments.

## Overview

The `ai_trading/imports.py` module handles imports for all AI trading dependencies with comprehensive fallback implementations, ensuring the bot can function even when dependencies are missing.

## Features

### 1. Dependency Management with Fallbacks

- **NumPy**: Mathematical operations with MockNumpy fallback
- **Pandas**: DataFrame/Series operations with MockPandas fallback  
- **Scikit-learn**: Machine learning utilities with MockSklearn fallback
- **TA-Lib/pandas-ta**: Technical analysis indicators with MockTalib fallback

### 2. Availability Flags

```python
from ai_trading.imports import (
    NUMPY_AVAILABLE,
    PANDAS_AVAILABLE, 
    SKLEARN_AVAILABLE,
    TALIB_AVAILABLE,
    PANDAS_TA_AVAILABLE
)
```

### 3. Unified Interface

```python
from ai_trading.imports import np, pd, get_ta_lib, LinearRegression

# Works regardless of whether real libraries are available
data = np.array([1, 2, 3, 4, 5])
df = pd.DataFrame({'price': data})
ta_lib = get_ta_lib()
sma = ta_lib.SMA(data, timeperiod=3)
```

## Mock Implementations

### MockNumpy
- Basic mathematical operations (mean, std, sum, max, min)
- Array creation and manipulation
- Mathematical constants (pi, e, nan, inf)
- Logical operations (isnan, isinf, isfinite)

### MockPandas
- DataFrame and Series creation
- Basic operations (mean, std, rolling, shift)
- Data manipulation (dropna, fillna, concat)
- I/O operations (read_csv, to_dict)

### MockSklearn
- Model classes (LinearRegression, RandomForestRegressor)
- Preprocessing (StandardScaler)
- Model selection (train_test_split)
- Metrics (mean_squared_error, r2_score)

### MockTalib
- Moving averages (SMA, EMA)
- Momentum indicators (RSI, MACD)
- Volatility indicators (BBANDS, ATR)
- Stochastic oscillators (STOCH)

## Technical Analysis Indicators

The mock TA library provides realistic implementations of major indicators:

- **SMA**: Simple Moving Average
- **EMA**: Exponential Moving Average  
- **RSI**: Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
- **BBANDS**: Bollinger Bands
- **ATR**: Average True Range
- **STOCH**: Stochastic Oscillator

## Usage Examples

### Basic Import
```python
from ai_trading.imports import np, pd, get_ta_lib

# NumPy operations
arr = np.array([1, 2, 3, 4, 5])
mean_val = np.mean(arr)

# Pandas operations
df = pd.DataFrame({'price': [100, 102, 101, 105]})
rolling_mean = df['price'].rolling(3).mean()

# Technical analysis
ta_lib = get_ta_lib()
sma = ta_lib.SMA([100, 102, 101, 105, 103], timeperiod=3)
```

### Machine Learning
```python
from ai_trading.imports import LinearRegression, StandardScaler

# Create and train model
X = [[1, 2], [3, 4], [5, 6]]
y = [10, 20, 30]

lr = LinearRegression()
lr.fit(X, y)
predictions = lr.predict([[7, 8]])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Checking Availability
```python
from ai_trading.imports import NUMPY_AVAILABLE, PANDAS_AVAILABLE

if NUMPY_AVAILABLE:
    print("Using real NumPy")
else:
    print("Using NumPy mock - some operations may be simplified")
```

## Benefits

1. **Robust Testing**: Tests can run without heavy dependencies
2. **Flexible Development**: Development environments don't need all libraries
3. **Gradual Migration**: Easy transition from mock to real implementations
4. **Production Safety**: Fallbacks prevent crashes in minimal deployments
5. **CI/CD Friendly**: Faster builds with optional dependencies

## Integration

The centralized import system integrates seamlessly with existing code:

1. Replace direct imports with imports from `ai_trading.imports`
2. Use availability flags to enable/disable features
3. Log warnings when fallbacks are used
4. Maintain same interfaces across mock and real implementations

## Logging

The system provides comprehensive logging:
- Info messages for successful imports
- Warning messages when using fallbacks
- Import summary showing what's available vs mocked

## AI-AGENT-REF Notes

This implementation follows the repository's guidelines:
- Uses centralized logger module for all output
- Maintains stability of core trading logic
- Provides incremental enhancement without breaking changes
- Includes comprehensive mock implementations for testing