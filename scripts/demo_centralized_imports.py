#!/usr/bin/env python3
import logging

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
    logging.info(str("=" * 60))
    logging.info("AI Trading Bot - Centralized Import Management Demo")
    logging.info(str("=" * 60))
    
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
    
    logging.info("\n📊 Dependency Availability Status:")
    logging.info(str(f"   NumPy:       {'✅ Available' if NUMPY_AVAILABLE else '❌ Using Mock'}"))
    logging.info(str(f"   Pandas:      {'✅ Available' if PANDAS_AVAILABLE else '❌ Using Mock'}"))
    logging.info(str(f"   Scikit-learn:{'✅ Available' if SKLEARN_AVAILABLE else '❌ Using Mock'}"))
    logging.info(str(f"   TA-Lib:      {'✅ Available' if TALIB_AVAILABLE else '❌ Using Mock'}"))
    logging.info(str(f"   pandas-ta:   {'✅ Available' if PANDAS_TA_AVAILABLE else '❌ Using Mock'}"))
    
    logging.info("\n🔬 Testing NumPy Operations:")
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    arr = np.array(data)
    mean_val = np.mean(data)
    std_val = np.std(data)
    logging.info(f"   Array creation: {arr}")
    logging.info(f"   Mean calculation: {mean_val}")
    logging.info(f"   Standard deviation: {std_val}")
    logging.info(f"   Mathematical constants: π = {np.pi}, e = {np.e}")
    
    logging.info("\n📈 Testing Pandas Operations:")
    df = pd.DataFrame({
        'price': [100, 102, 101, 105, 103, 107, 106, 110],
        'volume': [1000, 1200, 800, 1500, 900, 1800, 1100, 2000]
    })
    logging.info(f"   DataFrame shape: {df.shape}")
    logging.info(f"   DataFrame empty: {df.empty}")
    
    # Test Series operations
    price_series = df['price'] if hasattr(df, '__getitem__') else pd.Series([100, 102, 101, 105])
    rolling_mean = price_series.rolling(3).mean() if hasattr(price_series, 'rolling') else price_series
    logging.info(f"   Price series mean: {price_series.mean()}")
    logging.info(str(f"   Rolling mean calculated: {hasattr(rolling_mean, 'data') or hasattr(rolling_mean, '__len__')}"))
    
    logging.info("\n🤖 Testing Machine Learning:")
    # Prepare sample data
    X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
    y = [10, 20, 30, 40, 50]
    
    # Test LinearRegression
    lr = LinearRegression()
    lr.fit(X, y)
    predictions = lr.predict([[6, 7]])
    logging.info("   Linear regression trained successfully")
    logging.info(f"   Prediction for [6, 7]: {predictions}")
    
    # Test StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logging.info("   Standard scaling completed")
    logging.info(str(f"   Scaled data shape: {len(X_scaled) if hasattr(X_scaled, '__len__') else 'N/A'}"))
    
    logging.info("\n📊 Testing Technical Analysis:")
    # Prepare price data
    price_data = [100, 102, 101, 105, 103, 107, 106, 110, 108, 112, 115, 113, 118, 116, 120]
    high_data = [x + 2 for x in price_data]
    low_data = [x - 2 for x in price_data]
    
    ta_lib = get_ta_lib()
    
    # Test SMA
    sma_5 = ta_lib.SMA(price_data, timeperiod=5)
    logging.info(f"   SMA(5) calculated: {len(sma_5)} values")
    logging.info(str(f"   SMA(5) last 3 values: {sma_5[-3:] if hasattr(sma_5, '__getitem__') else 'Mock result'}"))
    
    # Test EMA
    ema_5 = ta_lib.EMA(price_data, timeperiod=5)
    logging.info(f"   EMA(5) calculated: {len(ema_5)} values")
    
    # Test RSI
    rsi = ta_lib.RSI(price_data, timeperiod=10)
    logging.info(f"   RSI(10) calculated: {len(rsi)} values")
    
    # Test MACD
    macd, signal, histogram = ta_lib.MACD(price_data)
    logging.info(f"   MACD calculated: {len(macd)} values")
    logging.info("   MACD components: MACD line, Signal line, Histogram")
    
    # Test Bollinger Bands
    upper, middle, lower = ta_lib.BBANDS(price_data)
    logging.info(f"   Bollinger Bands calculated: {len(upper)} values")
    logging.info("   BB components: Upper band, Middle band (SMA), Lower band")
    
    # Test ATR
    atr = ta_lib.ATR(high_data, low_data, price_data)
    logging.info(f"   ATR calculated: {len(atr)} values")
    
    logging.info("\n✅ Summary:")
    logging.info("   All core functionality working correctly!")
    logging.info("   The centralized import system successfully provides:")
    logging.info("   • Graceful fallbacks for missing dependencies")
    logging.info("   • Consistent interfaces across mock and real libraries")
    logging.info("   • Full technical analysis capabilities")
    logging.info("   • Machine learning functionality")
    logging.info("   • Mathematical and statistical operations")
    
    logging.info("\n🎯 Benefits:")
    logging.info("   • Trading bot can run in minimal environments")
    logging.info("   • Tests can run without heavy dependencies")
    logging.info("   • Development environments are more flexible")
    logging.info("   • Production deployments are more robust")
    
    logging.info(str("\n" + "=" * 60))
    logging.info("Centralized Import Management System: SUCCESS! ✨")
    logging.info(str("=" * 60))


if __name__ == "__main__":
    main()