"""
Comprehensive integration tests for centralized imports functionality.

This tests the actual behavior in a mixed environment where some 
dependencies might be available while others are not.

AI-AGENT-REF: Robust tests for production environment compatibility
"""

import pytest
import importlib.util


def test_centralized_imports_functionality():
    """Test that centralized imports work in current environment."""
    # Import the module directly to avoid ai_trading package issues
    spec = importlib.util.spec_from_file_location(
        'centralized_imports', 'ai_trading/imports.py'
    )
    imports_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imports_module)
    
    # Test availability flags
    assert hasattr(imports_module, 'NUMPY_AVAILABLE')
    assert hasattr(imports_module, 'PANDAS_AVAILABLE')
    assert hasattr(imports_module, 'SKLEARN_AVAILABLE')
    assert hasattr(imports_module, 'TALIB_AVAILABLE')
    assert hasattr(imports_module, 'PANDAS_TA_AVAILABLE')
    
    # Test core imports
    assert hasattr(imports_module, 'np')
    assert hasattr(imports_module, 'pd')
    assert hasattr(imports_module, 'sklearn')
    assert hasattr(imports_module, 'talib')
    assert hasattr(imports_module, 'get_ta_lib')


def test_numpy_mock_functionality():
    """Test NumPy mock provides expected interface."""
    spec = importlib.util.spec_from_file_location(
        'centralized_imports', 'ai_trading/imports.py'
    )
    imports_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imports_module)
    
    np = imports_module.np
    
    # Test constants
    assert hasattr(np, 'pi')
    assert hasattr(np, 'e')
    assert hasattr(np, 'nan')
    assert hasattr(np, 'inf')
    
    # Test array operations
    arr = np.array([1, 2, 3, 4, 5])
    assert hasattr(arr, '__len__') or hasattr(arr, '__iter__')
    
    # Test mathematical operations
    mean_val = np.mean([1, 2, 3, 4, 5])
    assert isinstance(mean_val, (int, float))
    
    std_val = np.std([1, 2, 3, 4, 5])
    assert isinstance(std_val, (int, float))


def test_pandas_mock_functionality():
    """Test Pandas mock provides expected interface."""
    spec = importlib.util.spec_from_file_location(
        'centralized_imports', 'ai_trading/imports.py'
    )
    imports_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imports_module)
    
    pd = imports_module.pd
    
    # Test DataFrame creation
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    assert hasattr(df, 'shape')
    assert hasattr(df, 'empty')
    assert hasattr(df, 'head')
    assert hasattr(df, 'mean')
    
    # Test Series creation
    series = pd.Series([1, 2, 3, 4, 5])
    assert hasattr(series, 'mean')
    assert hasattr(series, 'std')
    assert hasattr(series, 'shift')
    assert hasattr(series, 'rolling')
    
    # Test rolling operations
    rolling = series.rolling(3)
    rolling_mean = rolling.mean()
    assert hasattr(rolling_mean, 'data') or hasattr(rolling_mean, '__len__')


def test_sklearn_mock_functionality():
    """Test Scikit-learn mock provides expected interface."""
    spec = importlib.util.spec_from_file_location(
        'centralized_imports', 'ai_trading/imports.py'
    )
    imports_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imports_module)
    
    # Test model classes
    LinearRegression = imports_module.LinearRegression
    StandardScaler = imports_module.StandardScaler
    train_test_split = imports_module.train_test_split
    
    # Test LinearRegression
    lr = LinearRegression()
    X = [[1, 2], [3, 4], [5, 6]]
    y = [1, 2, 3]
    
    # These should not raise exceptions
    lr.fit(X, y)
    predictions = lr.predict(X)
    score = lr.score(X, y)
    
    assert hasattr(predictions, '__len__') or hasattr(predictions, '__iter__')
    assert isinstance(score, (int, float))
    
    # Test StandardScaler
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    inverse = scaler.inverse_transform(scaled)
    
    # Test train_test_split
    split_result = train_test_split(X, y, test_size=0.3)
    assert len(split_result) == 4  # X_train, X_test, y_train, y_test


def test_talib_mock_functionality():
    """Test TA-Lib mock provides expected interface."""
    spec = importlib.util.spec_from_file_location(
        'centralized_imports', 'ai_trading/imports.py'
    )
    imports_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imports_module)
    
    get_ta_lib = imports_module.get_ta_lib
    ta_lib = get_ta_lib()
    
    # Test data - enough for all indicators
    data = list(range(1, 31))  # [1, 2, 3, ..., 30]
    high = [x + 1 for x in data]
    low = [x - 1 for x in data]
    close = data
    
    # Test all major indicators
    sma = ta_lib.SMA(data, timeperiod=5)
    assert hasattr(sma, '__len__')
    assert len(sma) == len(data)
    
    ema = ta_lib.EMA(data, timeperiod=5)
    assert hasattr(ema, '__len__')
    assert len(ema) == len(data)
    
    rsi = ta_lib.RSI(data, timeperiod=14)
    assert hasattr(rsi, '__len__')
    
    # Test MACD
    macd, signal, histogram = ta_lib.MACD(data)
    assert all(hasattr(x, '__len__') for x in [macd, signal, histogram])
    assert all(len(x) == len(data) for x in [macd, signal, histogram])
    
    # Test Bollinger Bands
    upper, middle, lower = ta_lib.BBANDS(data)
    assert all(hasattr(x, '__len__') for x in [upper, middle, lower])
    assert all(len(x) == len(data) for x in [upper, middle, lower])
    
    # Test ATR
    atr = ta_lib.ATR(high, low, close, timeperiod=14)
    assert hasattr(atr, '__len__')
    assert len(atr) == len(data)
    
    # Test Stochastic
    slowk, slowd = ta_lib.STOCH(high, low, close)
    assert hasattr(slowk, '__len__')
    assert hasattr(slowd, '__len__')
    assert len(slowk) == len(data)
    assert len(slowd) == len(data)


def test_availability_flags_consistency():
    """Test that availability flags are consistent with actual imports."""
    spec = importlib.util.spec_from_file_location(
        'centralized_imports', 'ai_trading/imports.py'
    )
    imports_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imports_module)
    
    # Get availability flags
    numpy_available = imports_module.NUMPY_AVAILABLE
    pandas_available = imports_module.PANDAS_AVAILABLE
    sklearn_available = imports_module.SKLEARN_AVAILABLE
    talib_available = imports_module.TALIB_AVAILABLE
    pandas_ta_available = imports_module.PANDAS_TA_AVAILABLE
    
    # All should be boolean
    assert isinstance(numpy_available, bool)
    assert isinstance(pandas_available, bool)
    assert isinstance(sklearn_available, bool)
    assert isinstance(talib_available, bool)
    assert isinstance(pandas_ta_available, bool)
    
    # At least one TA library should be available (even if mocked)
    assert hasattr(imports_module, 'talib')
    assert hasattr(imports_module, 'ta')


def test_exports_in_all():
    """Test that __all__ exports are available."""
    spec = importlib.util.spec_from_file_location(
        'centralized_imports', 'ai_trading/imports.py'
    )
    imports_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imports_module)
    
    if hasattr(imports_module, '__all__'):
        for export_name in imports_module.__all__:
            assert hasattr(imports_module, export_name), \
                f"Module missing export listed in __all__: {export_name}"


def test_mock_edge_cases():
    """Test edge cases in mock implementations."""
    spec = importlib.util.spec_from_file_location(
        'centralized_imports', 'ai_trading/imports.py'
    )
    imports_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imports_module)
    
    np = imports_module.np
    pd = imports_module.pd
    ta_lib = imports_module.get_ta_lib()
    
    # Test empty data
    empty_result = np.mean([])
    assert isinstance(empty_result, (int, float))
    
    # Test single value
    single_result = np.mean([5])
    assert isinstance(single_result, (int, float))
    
    # Test pandas with empty data
    empty_df = pd.DataFrame()
    assert hasattr(empty_df, 'empty')
    
    # Test TA with insufficient data
    short_data = [1, 2]
    sma_short = ta_lib.SMA(short_data, timeperiod=5)
    assert hasattr(sma_short, '__len__')


if __name__ == '__main__':
    pytest.main([__file__])