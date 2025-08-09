"""
Simple tests for centralized import management system.

AI-AGENT-REF: Basic smoke tests for centralized imports
"""

import pytest


def test_imports_module_loads():
    """Test that the imports module loads successfully."""
    from ai_trading.imports import (
        NUMPY_AVAILABLE, PANDAS_AVAILABLE, SKLEARN_AVAILABLE,
        TALIB_AVAILABLE, PANDAS_TA_AVAILABLE
    )
    
    # All availability flags should be boolean
    assert isinstance(NUMPY_AVAILABLE, bool)
    assert isinstance(PANDAS_AVAILABLE, bool)
    assert isinstance(SKLEARN_AVAILABLE, bool)
    assert isinstance(TALIB_AVAILABLE, bool)
    assert isinstance(PANDAS_TA_AVAILABLE, bool)


def test_numpy_basic_operations():
    """Test basic numpy operations work."""
    from ai_trading.imports import np
    
    # Test array creation and basic math
    arr = np.array([1, 2, 3, 4, 5])
    mean_val = np.mean(arr)
    assert isinstance(mean_val, (int, float))
    
    # Test constants
    assert hasattr(np, 'nan')
    assert hasattr(np, 'pi')


def test_pandas_basic_operations():
    """Test basic pandas operations work."""
    from ai_trading.imports import pd
    
    # Test DataFrame creation
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    assert hasattr(df, 'shape')
    assert hasattr(df, 'empty')
    
    # Test Series creation
    series = pd.Series([1, 2, 3])
    assert hasattr(series, 'mean')


def test_sklearn_basic_operations():
    """Test basic sklearn operations work."""
    from ai_trading.imports import LinearRegression, StandardScaler
    
    # Test model creation and basic operations
    lr = LinearRegression()
    lr.fit([[1, 2], [3, 4]], [1, 2])
    predictions = lr.predict([[5, 6]])
    
    # Test scaler
    scaler = StandardScaler()
    scaled = scaler.fit_transform([[1, 2], [3, 4]])
    
    # Just verify they don't crash
    assert True


def test_talib_basic_operations():
    """Test basic TA operations work."""
    from ai_trading.imports import get_ta_lib
    
    ta_lib = get_ta_lib()
    data = [10, 11, 12, 11, 10, 9, 10, 11, 12, 13]
    
    # Test basic indicators
    sma = ta_lib.SMA(data, timeperiod=3)
    ema = ta_lib.EMA(data, timeperiod=3)
    rsi = ta_lib.RSI(data, timeperiod=5)
    
    # Verify they return list-like objects
    assert hasattr(sma, '__len__')
    assert hasattr(ema, '__len__')
    assert hasattr(rsi, '__len__')


def test_module_exports():
    """Test that expected exports are available."""
    import ai_trading.imports as imports
    
    required_exports = ['np', 'pd', 'get_ta_lib', 'talib']
    for export in required_exports:
        assert hasattr(imports, export), f"Missing export: {export}"


def test_technical_indicators():
    """Test various technical indicators work."""
    from ai_trading.imports import get_ta_lib
    
    ta_lib = get_ta_lib()
    
    # Prepare test data
    data = list(range(1, 21))  # [1, 2, 3, ..., 20]
    high = [x + 1 for x in data]
    low = [x - 1 for x in data]
    close = data
    
    # Test all major indicators
    sma = ta_lib.SMA(data, timeperiod=5)
    ema = ta_lib.EMA(data, timeperiod=5)
    rsi = ta_lib.RSI(data, timeperiod=5)
    
    # Test MACD
    macd, signal, histogram = ta_lib.MACD(data)
    assert all(hasattr(x, '__len__') for x in [macd, signal, histogram])
    
    # Test Bollinger Bands
    upper, middle, lower = ta_lib.BBANDS(data)
    assert all(hasattr(x, '__len__') for x in [upper, middle, lower])
    
    # Test ATR
    atr = ta_lib.ATR(high, low, close, timeperiod=5)
    assert hasattr(atr, '__len__')
    
    # Test Stochastic
    slowk, slowd = ta_lib.STOCH(high, low, close)
    assert hasattr(slowk, '__len__')
    assert hasattr(slowd, '__len__')


if __name__ == '__main__':
    pytest.main([__file__])