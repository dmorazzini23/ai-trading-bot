# 🔧 Troubleshooting Guide

## Overview

This guide provides comprehensive troubleshooting procedures, common issues, debugging steps, and performance optimization tips for the AI Trading Bot.

## Table of Contents

- [Quick Diagnostic Checklist](#quick-diagnostic-checklist)
- [Common Issues](#common-issues)
- [Debugging Procedures](#debugging-procedures)
- [Error Code Reference](#error-code-reference)
- [Performance Issues](#performance-issues)
- [API Connectivity Problems](#api-connectivity-problems)
- [Data Issues](#data-issues)
- [Trading Logic Problems](#trading-logic-problems)
- [System Resource Issues](#system-resource-issues)
- [FAQ](#faq)

## Quick Diagnostic Checklist

### Initial Health Check

Run this checklist before diving into detailed troubleshooting:

```bash
# 1. Check service status
sudo systemctl status ai-trading.service

# 2. Verify configuration
python validate_env.py
python verify_config.py

# 3. Test API connectivity
python health_check.py

# 4. Check logs for errors
tail -n 50 logs/scheduler.log | grep -i error

# 5. Verify Python environment
python --version  # Should be 3.12.3
pip list | grep -E "(pandas|numpy|alpaca)"

# 6. Check system resources
df -h  # Disk space
free -h  # Memory usage
top -p $(pgrep python)  # CPU usage
```

### Log Analysis Commands

```bash
# Recent errors
grep -i error logs/scheduler.log | tail -20

# API failures
grep -i "api.*fail\|timeout\|connection" logs/scheduler.log

# Trade execution issues
grep -i "trade\|order\|execution" logs/scheduler.log | tail -10

# Performance issues
grep -i "slow\|timeout\|memory" logs/scheduler.log

# Configuration problems
grep -i "config\|missing\|invalid" logs/scheduler.log
```

## Common Issues

### 1. Bot Won't Start

**Symptoms:**
- Service fails to start
- Immediate exit after launch
- ImportError or ModuleNotFoundError

**Solutions:**

```bash
# Check Python version
python --version
# Should output: Python 3.12.3

# Verify virtual environment
which python
# Should point to venv/bin/python

# Reinstall dependencies
python -m pip install -U pip
pip install -e .

# Check for missing environment variables
python -c "
import os
required = ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'BOT_MODE']
missing = [var for var in required if not os.getenv(var)]
if missing:
    print(f'Missing: {missing}')
else:
    print('All required variables present')
"
```

**Configuration Issues:**

```bash
# Validate .env file exists and is readable
ls -la .env
cat .env | grep -v "^#" | grep "="

# Check for common configuration errors
python -c "
import config
try:
    cfg = config.load_config()
    print('Configuration loaded successfully')
except Exception as e:
    print(f'Configuration error: {e}')
"
```

### Module attribute errors

**Error:** `module 'ai_trading.config.management' has no attribute 'get_env'` (or `reload_env`).

**Fix:** Update to the latest code and ensure `ai_trading/config/management.py` exports both helpers.

### Health endpoint returns 500

**Symptoms:** `curl -s http://127.0.0.1:9001/health` shows 500 or non-JSON.

**Fix:** The `/health` route must catch exceptions and return `{"ok": false}` on degradation. Verify no unhandled errors in the handler.

### 2. API Connection Failures

**Symptoms:**
- "Connection refused" errors
- "Authentication failed" messages
- Timeouts during data fetching

**Alpaca API Issues:**

```bash
# Test Alpaca connection
python -c "
import alpaca_trade_api as tradeapi
import os

try:
    api = tradeapi.REST(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_SECRET_KEY'),
        os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    )
    account = api.get_account()
    print(f'Connected to Alpaca. Account: {account.id}')
    print(f'Buying power: ${account.buying_power}')
except Exception as e:
    print(f'Alpaca connection failed: {e}')
"

# Check API key validity
curl -u "${ALPACA_API_KEY}:${ALPACA_SECRET_KEY}" \
  https://paper-api.alpaca.markets/v2/account
```

**Network Connectivity:**

```bash
# Test internet connectivity
ping -c 3 google.com

# Test specific endpoints
curl -I https://api.alpaca.markets/v2/account
curl -I https://finnhub.io/api/v1/quote

# Check DNS resolution
nslookup api.alpaca.markets
nslookup finnhub.io

# Test with different DNS servers
nslookup api.alpaca.markets 8.8.8.8
```

### 3. Data Fetching Problems

**Symptoms:**
- Empty dataframes
- Stale data warnings
- Data provider fallback messages

**Data Provider Diagnostics:**

```python
# test_data_providers.py
from ai_trading import data_fetcher
import pandas as pd
from datetime import UTC, datetime, timedelta

def test_data_provider(symbol='SPY', timeframe='1h'):
    """Test all data providers for a symbol."""
    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=5)
    
    providers = ['alpaca', 'finnhub', 'yahoo']
    
    for provider in providers:
        try:
            print(f"\nTesting {provider}...")
            data = data_fetcher.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                provider=provider
            )
            
            if data is not None and not data.empty:
                print(f"✓ {provider}: Got {len(data)} rows")
                print(f"  Latest: {data.index[-1]}")
                print(f"  Columns: {list(data.columns)}")
            else:
                print(f"✗ {provider}: No data returned")
                
        except Exception as e:
            print(f"✗ {provider}: Error - {e}")

if __name__ == "__main__":
    test_data_provider()
```

**Market Hours Issues:**

```python
# check_market_hours.py
import pandas_market_calendars as mcal
from datetime import UTC, datetime

def check_market_status():
    """Check if market is currently open."""
    nyse = mcal.get_calendar('NYSE')
    now = datetime.now(UTC)
    
    # Check if today is a trading day
    schedule = nyse.schedule(start_date=now.date(), end_date=now.date())
    
    if schedule.empty:
        print("Market is closed today (holiday/weekend)")
        return False
    
    # Check if within trading hours
    market_open = schedule.iloc[0]['market_open'].tz_convert('US/Eastern')
    market_close = schedule.iloc[0]['market_close'].tz_convert('US/Eastern')
    
    current_et = pd.Timestamp.now(tz='US/Eastern')
    
    if market_open <= current_et <= market_close:
        print(f"Market is OPEN (closes at {market_close.strftime('%H:%M ET')})")
        return True
    else:
        print(f"Market is CLOSED (opens at {market_open.strftime('%H:%M ET')})")
        return False

if __name__ == "__main__":
    check_market_status()
```

### 4. Trade Execution Failures

**Symptoms:**
- Orders rejected by broker
- Insufficient buying power errors
- Position size calculation errors

**Order Validation Issues:**

```python
# debug_order_validation.py
import trade_execution
import alpaca_trade_api as tradeapi
import os

def debug_order_issue(symbol, quantity, side):
    """Debug order execution issues."""
    
    print(f"Debugging order: {symbol} {side} {quantity}")
    
    # Check account status
    api = tradeapi.REST(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_SECRET_KEY'),
        os.getenv('ALPACA_BASE_URL')
    )
    
    account = api.get_account()
    print(f"Account status: {account.status}")
    print(f"Trading blocked: {account.trading_blocked}")
    print(f"Buying power: ${account.buying_power}")
    
    # Check position limits
    positions = api.list_positions()
    current_position = next((p for p in positions if p.symbol == symbol), None)
    
    if current_position:
        print(f"Current position: {current_position.qty} shares")
        print(f"Market value: ${current_position.market_value}")
    else:
        print("No current position")
    
    # Validate order
    try:
        is_valid, error_msg = trade_execution.validate_order(
            symbol, quantity, side, {}
        )
        print(f"Order validation: {'PASS' if is_valid else 'FAIL'}")
        if not is_valid:
            print(f"Error: {error_msg}")
    except Exception as e:
        print(f"Validation error: {e}")

# Example usage
if __name__ == "__main__":
    debug_order_issue('AAPL', 10, 'buy')
```

**Position Sizing Problems:**

```python
# debug_position_sizing.py
import ai_trading.risk.engine as risk_engine
import pandas as pd

def debug_position_sizing(symbol, signal_strength, account_equity):
    """Debug position sizing calculations."""
    
    print(f"Debugging position sizing for {symbol}")
    print(f"Signal strength: {signal_strength}")
    print(f"Account equity: ${account_equity}")
    
    try:
        # Get volatility data
        data = data_fetcher.get_historical_data(
            symbol, '1d', 
            (pd.Timestamp.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d'),
            pd.Timestamp.now().strftime('%Y-%m-%d')
        )
        
        returns = data['close'].pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5)  # Annualized
        
        print(f"30-day volatility: {volatility:.4f}")
        
        # Calculate position size
        position_size = risk_engine.calculate_position_size(
            symbol=symbol,
            signal_strength=signal_strength,
            account_equity=account_equity,
            volatility=volatility
        )
        
        print(f"Calculated position size: {position_size} shares")
        
        # Check against limits
        max_position_value = account_equity * 0.05  # 5% max
        current_price = data['close'].iloc[-1]
        max_shares = max_position_value / current_price
        
        print(f"Current price: ${current_price:.2f}")
        print(f"Max allowed shares (5%): {max_shares:.0f}")
        print(f"Position within limits: {position_size <= max_shares}")
        
    except Exception as e:
        print(f"Position sizing error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_position_sizing('SPY', 0.7, 50000)
```

## Debugging Procedures

### 1. Enable Debug Logging

```python
# Enable debug logging temporarily
import logging
import os

# Set environment variable
os.environ['LOG_LEVEL'] = 'DEBUG'

# Configure logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.debug("Debug logging enabled")
```

### 2. Performance Profiling

```python
# profile_bot.py
import cProfile
import pstats
import io
from ai_trading import main

def profile_bot():
    """Profile bot performance."""
    pr = cProfile.Profile()
    pr.enable()
    
    # Run bot for a short period
    try:
        main.run_trading_cycle()
    except KeyboardInterrupt:
        pass
    
    pr.disable()
    
    # Generate report
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative').print_stats(20)
    
    print(s.getvalue())
    
    # Save to file
    ps.dump_stats('profile_results.prof')

if __name__ == "__main__":
    profile_bot()
```

### 3. Memory Usage Analysis

```python
# memory_profiler.py
import psutil
import os
import time
from memory_profiler import profile

@profile
def analyze_memory_usage():
    """Analyze memory usage of bot components."""
    process = psutil.Process(os.getpid())
    
    print(f"Initial memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    # Import heavy modules
    import pandas as pd
    import numpy as np
    print(f"After pandas/numpy: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    import bot_engine
    print(f"After bot_engine: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    # Load test data
    data = pd.DataFrame({
        'close': np.random.randn(10000),
        'volume': np.random.randint(1000, 10000, 10000)
    })
    print(f"After test data: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    del data
    print(f"After cleanup: {process.memory_info().rss / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    analyze_memory_usage()
```

### 4. Network Diagnostics

```python
# network_diagnostics.py
import requests
import time
import statistics

def test_api_latency(url, num_tests=10):
    """Test API response times."""
    latencies = []
    
    for i in range(num_tests):
        start_time = time.time()
        try:
            response = requests.get(url, timeout=10)
            latency = (time.time() - start_time) * 1000  # ms
            latencies.append(latency)
            print(f"Test {i+1}: {latency:.2f}ms - Status: {response.status_code}")
        except Exception as e:
            print(f"Test {i+1}: FAILED - {e}")
        
        time.sleep(1)
    
    if latencies:
        print(f"\nLatency Statistics:")
        print(f"Average: {statistics.mean(latencies):.2f}ms")
        print(f"Median: {statistics.median(latencies):.2f}ms")
        print(f"Min: {min(latencies):.2f}ms")
        print(f"Max: {max(latencies):.2f}ms")

def run_network_diagnostics():
    """Run comprehensive network diagnostics."""
    
    endpoints = [
        'https://api.alpaca.markets/v2/account',
        'https://finnhub.io/api/v1/quote?symbol=AAPL',
        'https://query1.finance.yahoo.com/v8/finance/chart/SPY'
    ]
    
    for endpoint in endpoints:
        print(f"\nTesting {endpoint}:")
        test_api_latency(endpoint)

if __name__ == "__main__":
    run_network_diagnostics()
```

## Error Code Reference

### Application Error Codes

| Code | Description | Severity | Solution |
|------|-------------|----------|----------|
| `CONFIG_001` | Missing required environment variable | Critical | Set missing environment variables |
| `CONFIG_002` | Invalid configuration value | Critical | Correct configuration values |
| `API_001` | Alpaca API authentication failed | Critical | Check API keys and permissions |
| `API_002` | Alpaca API rate limit exceeded | Warning | Implement backoff strategy |
| `API_003` | Data provider unavailable | Warning | Switch to backup provider |
| `DATA_001` | Insufficient historical data | Warning | Extend data collection period |
| `DATA_002` | Data quality issues detected | Warning | Validate and clean data |
| `TRADE_001` | Insufficient buying power | Warning | Reduce position size |
| `TRADE_002` | Order rejected by broker | Warning | Check order parameters |
| `RISK_001` | Position exceeds risk limits | Warning | Reduce position size |
| `RISK_002` | Portfolio heat too high | Warning | Close some positions |
| `SYS_001` | Low disk space | Warning | Clean up old logs and data |
| `SYS_002` | High memory usage | Warning | Restart application |

### HTTP Error Codes

| Code | Description | Action |
|------|-------------|--------|
| 400 | Bad Request | Check request parameters |
| 401 | Unauthorized | Verify API credentials |
| 403 | Forbidden | Check API permissions |
| 404 | Not Found | Verify endpoint URL |
| 422 | Unprocessable Entity | Validate request data |
| 429 | Too Many Requests | Implement rate limiting |
| 500 | Internal Server Error | Check server logs |
| 502 | Bad Gateway | Check upstream services |
| 503 | Service Unavailable | Wait and retry |

## Performance Issues

### 1. Slow Indicator Calculations

**Symptoms:**
- Long processing times
- High CPU usage
- Memory consumption spikes

**Solutions:**

```python
# Optimize indicator calculations
import concurrent.futures
import pandas_ta as ta

def optimize_indicators(data):
    """Optimized indicator calculation."""
    
    # Use vectorized operations
    data['sma_20'] = data['close'].rolling(20).mean()
    data['ema_12'] = data['close'].ewm(span=12).mean()
    
    # Parallel calculation for heavy indicators
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit heavy calculations
        rsi_future = executor.submit(ta.rsi, data['close'])
        macd_future = executor.submit(ta.macd, data['close'])
        bbands_future = executor.submit(ta.bbands, data['close'])
        
        # Collect results
        data['rsi'] = rsi_future.result()
        macd_result = macd_future.result()
        bbands_result = bbands_future.result()
        
        # Combine results
        data = pd.concat([data, macd_result, bbands_result], axis=1)
    
    return data
```

### 2. Memory Leaks

**Detection:**

```python
# memory_monitor.py
import psutil
import gc
import threading
import time

class MemoryMonitor:
    def __init__(self, threshold_mb=1000):
        self.threshold_mb = threshold_mb
        self.monitoring = False
        
    def start_monitoring(self):
        """Start memory monitoring in background."""
        self.monitoring = True
        thread = threading.Thread(target=self._monitor_loop)
        thread.daemon = True
        thread.start()
        
    def _monitor_loop(self):
        """Monitor memory usage continuously."""
        while self.monitoring:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.threshold_mb:
                print(f"WARNING: Memory usage {memory_mb:.2f}MB exceeds threshold")
                # Force garbage collection
                gc.collect()
                
            time.sleep(60)  # Check every minute
            
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring = False

# Usage
monitor = MemoryMonitor(threshold_mb=500)
monitor.start_monitoring()
```

### 3. Database Performance

**Optimization:**

```python
# database_optimization.py
import sqlite3
import pandas as pd

def optimize_database():
    """Optimize database performance."""
    
    conn = sqlite3.connect('trading_data.db')
    cursor = conn.cursor()
    
    # Create indexes
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_trades_symbol 
    ON trades(symbol)
    """)
    
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_trades_timestamp 
    ON trades(timestamp)
    """)
    
    # Analyze tables
    cursor.execute("ANALYZE")
    
    # Vacuum database
    cursor.execute("VACUUM")
    
    conn.close()
    print("Database optimization completed")

def batch_insert_trades(trades_data):
    """Efficient batch insert for trades."""
    
    conn = sqlite3.connect('trading_data.db')
    
    # Use executemany for batch inserts
    cursor = conn.cursor()
    cursor.executemany("""
    INSERT INTO trades (symbol, side, quantity, price, timestamp)
    VALUES (?, ?, ?, ?, ?)
    """, trades_data)
    
    conn.commit()
    conn.close()
```

## API Connectivity Problems

### 1. Alpaca API Issues

**Connection Testing:**

```bash
# Test Alpaca API connectivity
curl -X GET \
  -H "APCA-API-KEY-ID: ${ALPACA_API_KEY}" \
  -H "APCA-API-SECRET-KEY: ${ALPACA_SECRET_KEY}" \
  "${ALPACA_BASE_URL}/v2/account"

# Check market data permissions
curl -X GET \
  -H "APCA-API-KEY-ID: ${ALPACA_API_KEY}" \
  -H "APCA-API-SECRET-KEY: ${ALPACA_SECRET_KEY}" \
  "${ALPACA_BASE_URL}/v2/stocks/AAPL/quotes/latest"
```

**Common Alpaca Errors:**

| Error | Cause | Solution |
|-------|-------|----------|
| 40110000 | Invalid API key | Check API key format |
| 40210000 | Account restricted | Contact Alpaca support |
| 42210000 | Insufficient funds | Check buying power |
| 42910000 | Rate limit exceeded | Implement backoff |

### 2. Data Provider Fallback

```python
# data_provider_fallback.py
import logging

class DataProviderManager:
    def __init__(self):
        self.providers = ['alpaca', 'finnhub', 'yahoo']
        self.current_provider = 0
        
    def get_data_with_fallback(self, symbol, timeframe, start_date, end_date):
        """Get data with automatic provider fallback."""
        
        for i, provider in enumerate(self.providers):
            try:
                logging.info(f"Attempting data fetch with {provider}")
                
                data = self._fetch_data(provider, symbol, timeframe, start_date, end_date)
                
                if data is not None and not data.empty:
                    self.current_provider = i
                    logging.info(f"Successfully fetched data from {provider}")
                    return data
                    
            except Exception as e:
                logging.warning(f"Provider {provider} failed: {e}")
                continue
        
        logging.error("All data providers failed")
        return None
        
    def _fetch_data(self, provider, symbol, timeframe, start_date, end_date):
        """Fetch data from specific provider."""
        if provider == 'alpaca':
            return self._fetch_alpaca_data(symbol, timeframe, start_date, end_date)
        elif provider == 'finnhub':
            return self._fetch_finnhub_data(symbol, timeframe, start_date, end_date)
        elif provider == 'yahoo':
            return self._fetch_yahoo_data(symbol, timeframe, start_date, end_date)
```

## FAQ

### Q: Bot is not making any trades

**A:** Check the following:
1. Market is open (`check_market_hours.py`)
2. Signals are being generated (`grep -i signal logs/scheduler.log`)
3. Risk limits are not too restrictive
4. Sufficient buying power available
5. No position limits preventing trades

### Q: High memory usage after running for hours

**A:** This is likely due to:
1. Data accumulation in memory - implement periodic cleanup
2. ML model memory leaks - reload models periodically
3. Pandas DataFrame fragmentation - use `df.copy()` when modifying

```python
# Memory cleanup routine
import gc

def cleanup_memory():
    """Periodic memory cleanup."""
    gc.collect()  # Force garbage collection
    
    # Clear old data from memory
    if hasattr(self, 'historical_data'):
        # Keep only last 1000 rows
        for symbol in self.historical_data:
            if len(self.historical_data[symbol]) > 1000:
                self.historical_data[symbol] = self.historical_data[symbol].tail(1000).copy()
```

### Q: Orders are being rejected with "Insufficient funds"

**A:** Common causes:
1. Buying power calculation is incorrect
2. Pending orders using available funds
3. Account restrictions
4. Position size too large for available cash

```python
# Check available funds
def check_buying_power():
    account = api.get_account()
    print(f"Buying power: ${account.buying_power}")
    print(f"Cash: ${account.cash}")
    print(f"Portfolio value: ${account.portfolio_value}")
    
    # Check pending orders
    orders = api.list_orders(status='open')
    pending_value = sum(float(order.qty) * float(order.limit_price or 0) 
                       for order in orders if order.side == 'buy')
    print(f"Pending orders value: ${pending_value}")
```

### Q: Indicators calculating incorrectly

**A:** Verify:
1. Data quality and completeness
2. Correct timeframe alignment
3. Sufficient historical data for indicator periods
4. No forward-looking bias in calculations

```python
# Debug indicator calculation
def debug_indicators(data, symbol):
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Missing values: {data.isnull().sum()}")
    
    # Check for data gaps
    expected_periods = len(pd.date_range(data.index[0], data.index[-1], freq='1H'))
    actual_periods = len(data)
    if expected_periods != actual_periods:
        print(f"WARNING: Data gaps detected. Expected {expected_periods}, got {actual_periods}")
```

### Q: Bot stops running after a few hours

**A:** Check for:
1. Unhandled exceptions in main loop
2. Memory exhaustion causing crash
3. Network connectivity issues
4. API rate limits causing permanent failures

**Add robust error handling:**

```python
def robust_trading_loop():
    """Trading loop with comprehensive error handling."""
    retry_count = 0
    max_retries = 3
    
    while True:
        try:
            run_trading_cycle()
            retry_count = 0  # Reset on success
            
        except Exception as e:
            retry_count += 1
            logging.error(f"Trading cycle failed (attempt {retry_count}): {e}")
            
            if retry_count >= max_retries:
                logging.critical("Max retries exceeded, shutting down")
                break
            
            # Exponential backoff
            sleep_time = 2 ** retry_count
            logging.info(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
```

This troubleshooting guide provides comprehensive procedures for diagnosing and resolving common issues with the AI Trading Bot.