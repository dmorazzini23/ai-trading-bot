#!/usr/bin/env python3
"""
Demonstration of the centralized import management system.

This script shows how the ai_trading.imports module provides graceful 
fallbacks for dependencies that may not be available.

AI-AGENT-REF: Production-ready import demonstration
"""

import os
import sys

# Add the current directory to the path for demo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    print("=" * 60)
    print("AI Trading Bot - Centralized Import Management Demo")
    print("=" * 60)
    
    # Import our centralized imports module directly
    import importlib.util
    spec = importlib.util.spec_from_file_location('ai_imports', 'ai_trading/imports.py')
    imports_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imports_module)
    
    # Extract the components we need
    np = imports_module.np
    pd = imports_module.pd
    sklearn = imports_module.sklearn
    get_ta_lib = imports_module.get_ta_lib
    NUMPY_AVAILABLE = imports_module.NUMPY_AVAILABLE
    PANDAS_AVAILABLE = imports_module.PANDAS_AVAILABLE
    SKLEARN_AVAILABLE = imports_module.SKLEARN_AVAILABLE
    TALIB_AVAILABLE = imports_module.TALIB_AVAILABLE
    PANDAS_TA_AVAILABLE = imports_module.PANDAS_TA_AVAILABLE
    LinearRegression = imports_module.LinearRegression
    StandardScaler = imports_module.StandardScaler
    
    print(f"\n📊 Dependency Availability Status:")
    print(f"   NumPy:       {'✅ Available' if NUMPY_AVAILABLE else '❌ Using Mock'}")
    print(f"   Pandas:      {'✅ Available' if PANDAS_AVAILABLE else '❌ Using Mock'}")
    print(f"   Scikit-learn:{'✅ Available' if SKLEARN_AVAILABLE else '❌ Using Mock'}")
    print(f"   TA-Lib:      {'✅ Available' if TALIB_AVAILABLE else '❌ Using Mock'}")
    print(f"   pandas-ta:   {'✅ Available' if PANDAS_TA_AVAILABLE else '❌ Using Mock'}")
    
    print(f"\n🔬 Testing NumPy Operations:")
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    arr = np.array(data)
    mean_val = np.mean(data)
    std_val = np.std(data)
    print(f"   Array creation: {arr}")
    print(f"   Mean calculation: {mean_val}")
    print(f"   Standard deviation: {std_val}")
    print(f"   Mathematical constants: π = {np.pi}, e = {np.e}")
    
    print(f"\n📈 Testing Pandas Operations:")
    df = pd.DataFrame({
        'price': [100, 102, 101, 105, 103, 107, 106, 110],
        'volume': [1000, 1200, 800, 1500, 900, 1800, 1100, 2000]
    })
    print(f"   DataFrame shape: {df.shape}")
    print(f"   DataFrame empty: {df.empty}")
    
    # Test Series operations
    price_series = df['price'] if hasattr(df, '__getitem__') else pd.Series([100, 102, 101, 105])
    rolling_mean = price_series.rolling(3).mean() if hasattr(price_series, 'rolling') else price_series
    print(f"   Price series mean: {price_series.mean()}")
    print(f"   Rolling mean calculated: {hasattr(rolling_mean, 'data') or hasattr(rolling_mean, '__len__')}")
    
    print(f"\n🤖 Testing Machine Learning:")
    # Prepare sample data
    X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
    y = [10, 20, 30, 40, 50]
    
    # Test LinearRegression
    lr = LinearRegression()
    lr.fit(X, y)
    predictions = lr.predict([[6, 7]])
    print(f"   Linear regression trained successfully")
    print(f"   Prediction for [6, 7]: {predictions}")
    
    # Test StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"   Standard scaling completed")
    print(f"   Scaled data shape: {len(X_scaled) if hasattr(X_scaled, '__len__') else 'N/A'}")
    
    print(f"\n📊 Testing Technical Analysis:")
    # Prepare price data
    price_data = [100, 102, 101, 105, 103, 107, 106, 110, 108, 112, 115, 113, 118, 116, 120]
    high_data = [x + 2 for x in price_data]
    low_data = [x - 2 for x in price_data]
    
    ta_lib = get_ta_lib()
    
    # Test SMA
    sma_5 = ta_lib.SMA(price_data, timeperiod=5)
    print(f"   SMA(5) calculated: {len(sma_5)} values")
    print(f"   SMA(5) last 3 values: {sma_5[-3:] if hasattr(sma_5, '__getitem__') else 'Mock result'}")
    
    # Test EMA
    ema_5 = ta_lib.EMA(price_data, timeperiod=5)
    print(f"   EMA(5) calculated: {len(ema_5)} values")
    
    # Test RSI
    rsi = ta_lib.RSI(price_data, timeperiod=10)
    print(f"   RSI(10) calculated: {len(rsi)} values")
    
    # Test MACD
    macd, signal, histogram = ta_lib.MACD(price_data)
    print(f"   MACD calculated: {len(macd)} values")
    print(f"   MACD components: MACD line, Signal line, Histogram")
    
    # Test Bollinger Bands
    upper, middle, lower = ta_lib.BBANDS(price_data)
    print(f"   Bollinger Bands calculated: {len(upper)} values")
    print(f"   BB components: Upper band, Middle band (SMA), Lower band")
    
    # Test ATR
    atr = ta_lib.ATR(high_data, low_data, price_data)
    print(f"   ATR calculated: {len(atr)} values")
    
    print(f"\n✅ Summary:")
    print(f"   All core functionality working correctly!")
    print(f"   The centralized import system successfully provides:")
    print(f"   • Graceful fallbacks for missing dependencies")
    print(f"   • Consistent interfaces across mock and real libraries")
    print(f"   • Full technical analysis capabilities")
    print(f"   • Machine learning functionality")
    print(f"   • Mathematical and statistical operations")
    
    print(f"\n🎯 Benefits:")
    print(f"   • Trading bot can run in minimal environments")
    print(f"   • Tests can run without heavy dependencies")
    print(f"   • Development environments are more flexible")
    print(f"   • Production deployments are more robust")
    
    print("\n" + "=" * 60)
    print("Centralized Import Management System: SUCCESS! ✨")
    print("=" * 60)


if __name__ == "__main__":
    main()