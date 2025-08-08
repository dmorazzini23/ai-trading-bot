# 🚀 AI Trading Bot

[![CI](https://github.com/dmorazzini23/ai-trading-bot/actions/workflows/python-app.yml/badge.svg)](https://github.com/dmorazzini23/ai-trading-bot/actions/workflows/python-app.yml)
[![codecov](https://codecov.io/gh/dmorazzini23/ai-trading-bot/branch/main/graph/badge.svg)](https://codecov.io/gh/dmorazzini23/ai-trading-bot)
[![deploy](https://github.com/dmorazzini23/ai-trading-bot/actions/workflows/deploy.yml/badge.svg)](https://github.com/dmorazzini23/ai-trading-bot/actions/workflows/deploy.yml)
[![Python 3.12.3](https://img.shields.io/badge/python-3.12.3-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated **AI-powered algorithmic trading system** that combines machine learning, technical analysis, and risk management for automated trading through Alpaca Markets.

## 📦 Package Import Paths

### Preferred (Package-first)
```python
from ai_trading.signals import generate_position_hold_signals
from ai_trading.data_fetcher import get_minute_df
from ai_trading.trade_execution import recent_buys
from ai_trading import indicators
```

### Legacy (Deprecated - will be removed in a future release)
```python
from signals import generate_position_hold_signals  # ⚠️ Deprecated
from data_fetcher import get_minute_df             # ⚠️ Deprecated  
from trade_execution import recent_buys            # ⚠️ Deprecated
import indicators                                   # ⚠️ Deprecated
```

**Migration Note**: Root imports are deprecated and will be removed in a future release. Please migrate to use `ai_trading.<module>` imports for better package isolation and future compatibility.

## ✨ Key Features

### 🧠 Advanced AI & Machine Learning
- **Multi-timeframe Analysis**: 1m, 5m, 15m, 1h, 1d data processing
- **Machine Learning Models**: LightGBM, scikit-learn, and reinforcement learning
- **Signal Generation**: Momentum, mean reversion, moving averages, regime detection
- **Meta-learning**: Adaptive strategy selection based on market conditions

### 📊 Technical Analysis Engine
- **50+ Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, and custom indicators
- **Parallel Processing**: Concurrent indicator calculations for optimal performance
- **Multi-horizon Signals**: Aggregated signals across different timeframes
- **Regime Detection**: Hidden Markov Models for market state identification

### 🛡️ Robust Risk Management
- **Kelly Criterion**: Optimal position sizing based on signal confidence
- **Volatility Scaling**: Dynamic position adjustment based on market volatility
- **Portfolio Heat**: Total risk exposure monitoring and limits
- **Drawdown Protection**: Automatic position reduction during adverse periods

### 🔄 Live Trading & Execution
- **Alpaca Integration**: Seamless broker connectivity with paper/live trading
- **Smart Order Routing**: Multiple data providers with automatic failover
- **Slippage Tracking**: Real-time execution cost monitoring
- **Async Execution**: Non-blocking order processing and monitoring

### 📈 Performance & Monitoring
- **Real-time Dashboard**: Web-based monitoring and control interface
- **Comprehensive Logging**: Detailed trade and performance logging
- **Metrics Collection**: Prometheus-compatible performance metrics
- **Health Monitoring**: System health checks and alerting

## 🎯 Recent Improvements

This update introduces several critical enhancements:

1. **🔧 StrategyAllocator Bug Fix**: Resolved hold protection logic issues
2. **⚡ Enhanced Signal Sensitivity**: Reduced delta threshold from 0.3 to 0.02
3. **📏 ATR-Based Sizing**: Improved position sizing using Average True Range
4. **✅ Signal Confirmation**: Multi-bar signal validation for higher accuracy
5. **🎛️ Dynamic Risk Management**: Adaptive exposure limits based on performance
6. **🌐 Expanded Trading Universe**: Portfolio expanded from 5 to 24+ high-quality stocks
7. **🔧 Enhanced Technical Analysis**: Integrated ta library v0.11.0 for reliable cross-platform indicators

### 📊 Trading Universe

The bot now monitors a diversified portfolio of 24+ symbols across multiple sectors:

- **Technology Leaders**: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, AMD, META, NFLX, CRM
- **Growth Stocks**: UBER, SHOP, SQ, PLTR  
- **Market ETFs**: SPY, QQQ, IWM
- **Blue Chip**: JPM, JNJ, PG, KO
- **Energy**: XOM, CVX
- **International**: BABA

Symbols are automatically screened based on volume (>100K daily) and volatility (ATR) ranking to ensure optimal trading opportunities.

## 📚 Documentation

- **[🏗️ Architecture Guide](ARCHITECTURE.md)** - System architecture and design patterns
- **[📖 API Documentation](API_DOCUMENTATION.md)** - Complete API reference and examples
- **[🚀 Deployment Guide](DEPLOYMENT.md)** - Production deployment and scaling
- **[🔧 Troubleshooting](TROUBLESHOOTING.md)** - Debugging procedures and FAQ
- **[💼 Development Setup](docs/DEVELOPMENT.md)** - Development environment setup

**Targets Python 3.12.3** for optimal performance and compatibility.

---

## ⚙️ Installation

### Prerequisites

- **Python 3.12.3** (exact version required)
- **Git** for version control
- **4GB+ RAM** (8GB+ recommended for production)
- **Stable internet connection** for market data

### Dependencies

This trading bot requires both **Python packages** and **system libraries** for optimal performance:

#### Required System Dependencies
- **Python 3.8+**: Core runtime environment
- **Git**: For repository management and updates

#### Python Dependencies
All Python packages are specified in `requirements.txt`, including:
- **ta==0.11.0**: Professional technical analysis library with 150+ indicators
- **pandas**, **numpy**: Data processing and numerical computations
- **scikit-learn**: Machine learning algorithms
- **alpaca-trade-api**: Broker integration

**Note**: The ta library provides cross-platform compatibility without requiring system-level C library installations.

### Quick Start

```bash
# Clone the repository
git clone https://github.com/dmorazzini23/ai-trading-bot.git
cd ai-trading-bot

# Install Python dependencies (ta library included)
pip install -r requirements.txt

# Alternative automated setup
make install-dev
make validate-env

# Verify installation
python --version  # Should output: Python 3.12.3
python health_check.py
```

### Manual Installation

If you prefer manual setup or encounter issues with the automated process:

```bash
# Create and activate virtual environment
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip and install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install development tools (optional)
pip install -r requirements-dev.txt

# Verify installation
pip list | grep -E "(pandas|numpy|alpaca|scikit-learn)"
```

### Technical Analysis Library

The bot uses the `ta` library v0.11.0 for professional-grade technical analysis, providing 150+ indicators with excellent cross-platform compatibility.

#### Features
- **150+ Technical Indicators**: Complete coverage of trend, momentum, volatility, and volume indicators
- **Cross-Platform**: Works on Windows, macOS, and Linux without C library dependencies
- **High Performance**: Optimized pandas-native operations
- **Professional Grade**: Used in production trading systems worldwide

#### Available Indicators
- **Trend**: SMA, EMA, MACD, ADX, Bollinger Bands
- **Momentum**: RSI, Stochastic, Williams %R, CCI
- **Volatility**: ATR, Bollinger Band Width, Donchian Channel
- **Volume**: OBV, VWAP, Accumulation/Distribution Line

#### Usage Examples
```python
from ai_trading.imports import get_ta_lib

ta_lib = get_ta_lib()

# Traditional TA-Lib compatible interface
sma = ta_lib.SMA(close_prices, timeperiod=20)
rsi = ta_lib.RSI(close_prices, timeperiod=14)

# Direct ta library interface for advanced usage
from ai_trading.strategies.imports import ta
sma_direct = ta.trend.sma_indicator(close_series, window=20)
```

#### Installation
The ta library is automatically installed with the bot dependencies:
```bash
pip install -r requirements.txt
```

**Note**: No additional system dependencies or C libraries required. The ta library provides reliable cross-platform technical analysis out of the box.

### Docker Installation

For containerized deployment:

```bash
# Build the Docker image
docker build -t ai-trading-bot:latest .

# Run with environment file
docker run -d --name ai-trading-bot \
  --env-file .env \
  -v $(pwd)/logs:/app/logs \
  -p 5000:5000 \
  ai-trading-bot:latest

# Check status
docker logs ai-trading-bot
```

### Troubleshooting Installation

**Common Issues:**

1. **Python Version Mismatch**: Ensure exactly Python 3.12.3
   ```bash
   python --version
   which python
   ```

2. **Dependency Conflicts**: Clean install in fresh environment
   ```bash
   rm -rf venv
   python3.12 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Missing System Dependencies** (Linux/macOS):
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install python3.12-dev build-essential
   
   # macOS
   brew install python@3.12
   ```

4. **Technical Analysis Library Issues**:
   ```bash
   # Verify ta library installation
   python -c "import ta; print('ta library version:', ta.__version__ if hasattr(ta, '__version__') else 'installed')"
   
   # Check bot's technical analysis status
   python -c "from ai_trading.strategies.imports import TA_AVAILABLE; print(f'TA available: {TA_AVAILABLE}')"
   
   # Reinstall if needed
   pip install --upgrade ta==0.11.0
   ```

### Development Setup

For comprehensive development environment setup, see [**docs/DEVELOPMENT.md**](docs/DEVELOPMENT.md).

---

---

## 🚀 Usage

### Quick Start Trading

```bash
# Start the bot with default settings
python -m ai_trading

# Or use the convenience script
./start.sh
```

### 📈 Backtesting & Optimization

#### Basic Backtesting

```bash
# Run backtest on popular ETFs
python backtester.py --symbols SPY,QQQ,IWM --start 2024-01-01 --end 2024-12-31

# Test with custom parameters
python backtester.py \
  --symbols AAPL,MSFT,GOOGL \
  --start 2023-01-01 \
  --end 2024-01-01 \
  --initial-capital 100000 \
  --max-position-pct 0.08
```

#### Advanced Backtesting

```bash
# Run with specific strategy
python backtester.py \
  --symbols SPY \
  --strategy momentum \
  --timeframes 1h,1d \
  --optimize-hyperparams

# Multi-timeframe analysis
python backtester.py \
  --symbols SPY,AAPL,TSLA \
  --timeframes 5m,15m,1h,1d \
  --lookback-days 90
```

**Results:** The backtester performs grid search optimization and saves the best parameters to `best_hyperparams.json`.

#### Hyperparameter Optimization

The bot automatically uses optimized parameters when available:

1. **`best_hyperparams.json`** - Optimized parameters (preferred)
2. **`hyperparams.json`** - Default parameters (fallback)

```bash
# View current parameters
cat best_hyperparams.json

# Manual optimization
python algorithm_optimizer.py --symbols SPY --iterations 100
```

### 🎛️ Trading Modes

#### Paper Trading (Recommended for beginners)

```bash
# Set environment for paper trading
export ALPACA_BASE_URL=https://paper-api.alpaca.markets
export BOT_MODE=paper

# Start bot
python -m ai_trading
```

#### Live Trading (Production)

```bash
# ⚠️  CAUTION: Real money trading
export ALPACA_BASE_URL=https://api.alpaca.markets
export BOT_MODE=production
export MAX_POSITION_PCT=0.03  # Conservative sizing

# Start with extra safety checks
python -m ai_trading --confirm-live-trading
```

### 📊 Monitoring & Control

#### Real-time Dashboard

```bash
# Start web dashboard
python monitoring_dashboard.py

# Access at http://localhost:5000
# View real-time performance, positions, and logs
```

#### Command Line Monitoring

```bash
# Follow live logs
tail -F logs/scheduler.log

# Watch specific events
tail -F logs/scheduler.log | grep -E "(BUY|SELL|ERROR)"

# Performance summary
python -c "
import pandas as pd
trades = pd.read_csv('trades.csv')
print(f'Total trades: {len(trades)}')
print(f'Win rate: {(trades[\"pnl\"] > 0).mean():.1%}')
print(f'Total PnL: ${trades[\"pnl\"].sum():.2f}')
"
```

#### Health Checks

```bash
# System health check
python health_check.py

# API connectivity test
python -c "
from data_fetcher import test_all_providers
test_all_providers(['SPY'])
"

# Risk limits check
python risk_engine.py --check-limits
```

### 🔄 Advanced Usage

#### Custom Strategy Development

```python
# strategies/custom_strategy.py
from strategies.base import BaseStrategy

class CustomMomentumStrategy(BaseStrategy):
    def __init__(self, rsi_period=14, rsi_oversold=30, rsi_overbought=70):
        super().__init__()
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
    
    def generate_signals(self, data):
        """Generate custom trading signals."""
        signals = {}
        
        # Calculate RSI
        rsi = self.calculate_rsi(data['close'], self.rsi_period)
        
        # Generate signals
        if rsi.iloc[-1] < self.rsi_oversold:
            signals['action'] = 'BUY'
            signals['strength'] = (self.rsi_oversold - rsi.iloc[-1]) / self.rsi_oversold
        elif rsi.iloc[-1] > self.rsi_overbought:
            signals['action'] = 'SELL'
            signals['strength'] = (rsi.iloc[-1] - self.rsi_overbought) / (100 - self.rsi_overbought)
        else:
            signals['action'] = 'HOLD'
            signals['strength'] = 0
            
        return signals

# Usage
python -m ai_trading --strategy custom_momentum
```

#### Programmatic Trading

```python
# example_bot_usage.py
import asyncio
from bot_engine import run_all_trades_worker
from trade_execution import execute_order_async

async def custom_trading_logic():
    """Example of programmatic trading."""
    
    # Run analysis on specific symbols
    symbols = ['SPY', 'QQQ', 'IWM']
    results = run_all_trades_worker(symbols, dry_run=True)
    
    # Execute trades based on results
    for symbol, analysis in results.items():
        if analysis['signal_strength'] > 0.8:
            await execute_order_async(
                symbol=symbol,
                quantity=analysis['recommended_shares'],
                side='buy'
            )

# Run the custom logic
asyncio.run(custom_trading_logic())
```

### 🛠️ Development & Testing

#### Running Tests

```bash
# Quick test suite (excludes slow tests)
pytest -n auto --disable-warnings

# Full test suite
pytest --maxfail=1 --disable-warnings --strict-markers

# Test with coverage
pytest --cov=ai_trading --cov-fail-under=80 --cov-report=html

# Specific test categories
pytest -m "not slow"  # Fast tests only
pytest -m "smoke"     # Smoke tests
pytest -m "integration"  # Integration tests
```

#### Performance Analysis

```bash
# Profile bot performance
python -m cProfile -m ai_trading > profile_output.txt

# Memory profiling
python -m memory_profiler bot_engine.py

# Benchmark indicators
python profile_indicators.py
```

#### Data Management

```bash
# Fetch and cache data for development
python data_fetcher.py --cache --symbols SPY,AAPL --days 30

# Clean up old data
python cleanup.py --older-than 30days

# Validate data quality
python data_validator.py --check-all
```

### 🔧 Troubleshooting

For common issues and solutions, see [**TROUBLESHOOTING.md**](TROUBLESHOOTING.md).

**Quick Fixes:**

```bash
# Reset and restart
./cleanup_pycache.sh
python health_check.py
python -m ai_trading

# Check logs for errors
grep -i error logs/scheduler.log | tail -10

# Validate configuration
python validate_env.py
python verify_config.py
```

---

## 🔑 Configuration

### 🚨 Critical: API Keys Setup

**You must configure your broker API keys before running the bot.**

#### Step-by-Step Setup

1. **Get Alpaca API Credentials**
   - Sign up at [Alpaca Markets](https://app.alpaca.markets/signup)
   - Navigate to [API Management](https://app.alpaca.markets/paper/dashboard/overview)
   - Generate API Key and Secret Key
   - **Important**: Start with Paper Trading for testing

2. **Configure Environment Variables**
   ```bash
   # Copy template and edit
   cp .env.example .env
   
   # Edit .env file with your credentials
   nano .env  # or use your preferred editor
   ```

3. **Required Configuration**
   ```bash
   # Alpaca API Configuration
   ALPACA_API_KEY=your_actual_api_key_here
   ALPACA_SECRET_KEY=your_actual_secret_key_here
   ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Paper trading
   # ALPACA_BASE_URL=https://api.alpaca.markets     # Live trading (DANGER!)
   
   # Bot Configuration
   BOT_MODE=balanced                    # Trading mode: conservative, balanced, aggressive
   BOT_LOG_FILE=logs/scheduler.log     # Log file location
   LOG_LEVEL=INFO                      # DEBUG, INFO, WARNING, ERROR
   
   # Risk Management
   MAX_POSITION_PCT=0.05               # Maximum 5% per position
   MAX_PORTFOLIO_HEAT=0.15             # Maximum 15% total risk
   ENABLE_STOP_LOSS=true               # Enable stop-loss orders
   ```

### 📋 Configuration Reference

#### Core Trading Parameters

| Parameter | Default | Description | Range |
|-----------|---------|-------------|-------|
| `BOT_MODE` | `balanced` | Trading aggressiveness | `conservative`, `balanced`, `aggressive` |
| `SCHEDULER_SLEEP_SECONDS` | `60` | Delay between trading cycles | `30-300` seconds |
| `MAX_POSITION_PCT` | `0.05` | Maximum position size (% of equity) | `0.01-0.20` |
| `MAX_PORTFOLIO_HEAT` | `0.15` | Maximum total portfolio risk | `0.05-0.30` |
| `SIGNAL_THRESHOLD` | `0.7` | Minimum signal strength for trades | `0.1-1.0` |

#### Data and Market Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PRIMARY_DATA_PROVIDER` | `alpaca` | Primary data source |
| `BACKUP_DATA_PROVIDER` | `yahoo` | Fallback data source |
| `MARKET_DATA_TIMEOUT` | `30` | API timeout in seconds |
| `CACHE_MARKET_DATA` | `true` | Enable data caching |
| `TRADING_HOURS_ONLY` | `true` | Trade only during market hours |

#### Machine Learning Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `USE_RL_AGENT` | `false` | Enable reinforcement learning |
| `RL_MODEL_PATH` | `models/rl_agent.pkl` | RL model file path |
| `ML_MODEL_RETRAIN_DAYS` | `7` | Days between model retraining |
| `FEATURE_ENGINEERING` | `true` | Enable advanced features |

#### Advanced Settings

```bash
# Performance Optimization
USE_PARALLEL_PROCESSING=true
MAX_WORKER_THREADS=4
MEMORY_LIMIT_MB=2048

# Security
ENABLE_AUDIT_LOGGING=true
API_RATE_LIMIT=100

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=8000
WEBHOOK_URL=https://your-webhook-url.com/trading-alerts
```

### 🔒 Security Best Practices

1. **Never commit API keys to version control**
2. **Use environment-specific .env files**
3. **Rotate API keys regularly**
4. **Enable two-factor authentication**
5. **Monitor API usage and limits**

### 🧪 Testing Configuration

```bash
# .env.testing
BOT_MODE=paper
ALPACA_BASE_URL=https://paper-api.alpaca.markets
DRY_RUN=true
LOG_LEVEL=DEBUG
MAX_POSITION_PCT=0.01  # Very small positions for testing
```

### ✅ Configuration Validation

```bash
# Validate your configuration
python validate_env.py

# Check API connectivity
python -c "
import alpaca_trade_api as tradeapi
import os
api = tradeapi.REST(
    os.getenv('ALPACA_API_KEY'),
    os.getenv('ALPACA_SECRET_KEY'),
    os.getenv('ALPACA_BASE_URL')
)
account = api.get_account()
print(f'✅ Connected! Account: {account.id}')
print(f'💰 Buying Power: ${account.buying_power}')
"
```

📖 **For detailed API key setup instructions, see: [docs/API_KEY_SETUP.md](docs/API_KEY_SETUP.md)**

---

## 📡 Data Sources & Market Data

The bot uses a multi-provider approach for reliable market data access:

### Provider Priority

1. **🥇 Alpaca Markets** (Primary)
   - Real-time market data
   - Historical bars (1min to 1day)
   - Integrated with trading account
   - Low latency execution data

2. **🥈 Finnhub** (Secondary - Paid)
   - Alternative data source
   - Real-time quotes and fundamentals
   - News sentiment analysis
   - Economic calendar

3. **🥉 Yahoo Finance** (Fallback - Free)
   - Backup data provider
   - Historical data only
   - No real-time capabilities
   - Used when other providers fail

### Data Configuration

```bash
# Configure data providers in .env
PRIMARY_DATA_PROVIDER=alpaca
SECONDARY_DATA_PROVIDER=finnhub
FALLBACK_DATA_PROVIDER=yahoo

# Finnhub API (optional, for enhanced data)
FINNHUB_API_KEY=your_finnhub_api_key

# Data quality settings
ENABLE_DATA_VALIDATION=true
MAX_DATA_AGE_MINUTES=5
REQUIRE_MINIMUM_VOLUME=10000
```

### Supported Timeframes

| Timeframe | Alpaca | Finnhub | Yahoo | Usage |
|-----------|--------|---------|-------|-------|
| `1min` | ✅ | ✅ | ❌ | Scalping, real-time |
| `5min` | ✅ | ✅ | ❌ | Short-term signals |
| `15min` | ✅ | ✅ | ❌ | Intraday analysis |
| `1hour` | ✅ | ✅ | ✅ | Medium-term trends |
| `1day` | ✅ | ✅ | ✅ | Long-term analysis |

### Data Quality Monitoring

The bot automatically monitors data quality and switches providers when issues are detected:

```python
# Automatic data provider switching
if data_quality_score < 0.8:
    switch_to_backup_provider()
    log_data_quality_issue()
```

---

## 📝 Logging & Monitoring

### Log Structure

```
logs/
├── scheduler.log     # Main bot operations and decisions
├── trades.log        # Trade execution details
├── performance.log   # Performance metrics and analysis
├── error.log         # Error tracking and debugging
└── api.log          # API calls and responses
```

### Log Configuration

```bash
# Logging settings in .env
BOT_LOG_FILE=logs/scheduler.log
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR
LOG_ROTATION_SIZE=100MB          # Rotate when file reaches size
LOG_RETENTION_DAYS=30            # Keep logs for 30 days
LOG_FORMAT=detailed              # minimal, standard, detailed
```

### Real-time Log Monitoring

```bash
# Follow main activity
tail -F logs/scheduler.log

# Monitor trades only
tail -F logs/trades.log | grep -E "(BUY|SELL)"

# Watch for errors
tail -F logs/error.log

# Filter specific symbols
tail -F logs/scheduler.log | grep "AAPL"
```

### Log Analysis Examples

```bash
# Trading performance today
grep "$(date +%Y-%m-%d)" logs/trades.log | wc -l

# Error rate in last hour
grep "$(date -d '1 hour ago' '+%Y-%m-%d %H')" logs/error.log | wc -l

# Most traded symbols
grep -o "symbol='[A-Z]*'" logs/trades.log | sort | uniq -c | sort -nr
```

### Alerting Configuration

```bash
# Email alerts for critical events
ALERT_EMAIL=your-email@example.com
ALERT_ON_ERRORS=true
ALERT_ON_LARGE_DRAWDOWN=true
ALERT_DRAWDOWN_THRESHOLD=0.05

# Webhook notifications
WEBHOOK_URL=https://hooks.slack.com/your-webhook
WEBHOOK_ON_TRADES=true
WEBHOOK_ON_SYSTEM_EVENTS=true
```

---

## 🔥 Production Deployment

### Pre-Deployment Setup

The bot automatically initializes required files and directories on first run:

- **Trade Log**: Creates `data/trades.csv` with proper permissions (0o664) for trade auditing
- **Data Directory**: Auto-creates the `data/` directory if it doesn't exist
- **Permissions**: Ensures the bot user can read/write trade logs for audit compliance

### Systemd Service

Deploy as a persistent system service:

```bash
# Install service file
sudo cp ai-trading-scheduler.service /etc/systemd/system/

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable ai-trading-scheduler
sudo systemctl start ai-trading-scheduler

# Monitor service
sudo systemctl status ai-trading-scheduler
journalctl -u ai-trading-scheduler -f
```

### Service Configuration

```ini
# ai-trading-scheduler.service
[Unit]
Description=AI Trading Bot Scheduler
After=network.target

[Service]
Type=simple
User=ai-trading
Group=ai-trading
WorkingDirectory=/opt/ai-trading-bot
Environment=PATH=/opt/ai-trading-bot/venv/bin
ExecStart=/opt/ai-trading-bot/venv/bin/python -m ai_trading
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Docker Production Deployment

```bash
# Build production image
docker build -t ai-trading-bot:prod .

# Run with production settings
docker run -d \
  --name ai-trading-prod \
  --restart unless-stopped \
  --env-file .env.production \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  ai-trading-bot:prod

# Monitor container
docker logs -f ai-trading-prod
```

### Health Checks & Monitoring

```bash
# Built-in health endpoint
curl http://localhost:5000/health

# System resource monitoring
python monitoring_dashboard.py &

# Performance metrics
python metrics.py --export-prometheus
```

---

## 🔄 Daily Operations

### Automatic Retraining

The bot includes automatic model retraining capabilities:

```bash
# Enable/disable automatic retraining
export ENABLE_DAILY_RETRAIN=true   # Enable (default)
export DISABLE_DAILY_RETRAIN=1     # Disable

# Manual retraining
python retrain.py --symbols SPY,QQQ --days 90

# Schedule retraining
echo "0 2 * * * /opt/ai-trading-bot/venv/bin/python retrain.py" | crontab -
```

### Maintenance Tasks

```bash
# Daily maintenance script
#!/bin/bash
# daily_maintenance.sh

# Rotate logs
python logger_rotator.py

# Cleanup old data
find data/ -name "*.csv" -mtime +30 -delete

# Backup configuration
cp .env backup/.env.$(date +%Y%m%d)

# Health check
python health_check.py || echo "Health check failed" | mail -s "Bot Alert" admin@example.com

# Performance report
python performance_optimizer.py --generate-report
```

### Performance Monitoring

```bash
# Generate performance report
python performance_optimizer.py --report

# View trading statistics
python -c "
import pandas as pd
trades = pd.read_csv('trades.csv')
print('=== Trading Performance ===')
print(f'Total trades: {len(trades)}')
print(f'Win rate: {(trades[\"pnl\"] > 0).mean():.1%}')
print(f'Average trade: ${trades[\"pnl\"].mean():.2f}')
print(f'Best trade: ${trades[\"pnl\"].max():.2f}')
print(f'Worst trade: ${trades[\"pnl\"].min():.2f}')
print(f'Total PnL: ${trades[\"pnl\"].sum():.2f}')
"
```

---

## 🧪 Development & Testing

### Test Suite Overview

The bot includes a comprehensive testing framework with multiple test categories:

```bash
# Quick development tests (recommended during development)
pytest -n auto --disable-warnings -m "not slow"

# Full test suite (CI/CD)
pytest --maxfail=1 --disable-warnings --strict-markers

# Coverage testing (minimum 80% required)
pytest --cov=ai_trading --cov-fail-under=80 --cov-report=html

# Open coverage report
xdg-open htmlcov/index.html  # Linux
open htmlcov/index.html      # macOS
```

### Test Categories

| Marker | Description | Runtime | Usage |
|--------|-------------|---------|-------|
| `smoke` | Basic functionality tests | <30s | Quick validation |
| `integration` | End-to-end workflow tests | 1-5min | Feature validation |
| `slow` | Heavy computation/ML tests | 5-30min | Full validation |
| `benchmark` | Performance measurement | Variable | Optimization |

```bash
# Run specific test categories
pytest -m smoke          # Fast smoke tests
pytest -m integration    # Integration tests
pytest -m "not slow"     # Exclude slow tests
pytest -m benchmark      # Performance tests
```

### Development Workflow

```bash
# 1. Setup development environment
make install-dev
pre-commit install  # Install git hooks

# 2. Run quick tests during development
pytest -n auto -m "not slow" --maxfail=3

# 3. Check code quality
flake8 .
mypy ai_trading
black --check .

# 4. Run full test suite before committing
make test-all

# 5. Performance profiling (when needed)
python -m cProfile -m ai_trading
pyinstrument python -m ai_trading
```

### Creating Tests

```python
# tests/test_new_feature.py
import pytest
from unittest.mock import patch, MagicMock
from ai_trading.new_feature import NewFeature

class TestNewFeature:
    """Test suite for new feature."""
    
    @pytest.fixture
    def sample_data(self):
        """Provide test data."""
        return {
            'symbol': 'TEST',
            'price': 100.0,
            'volume': 1000
        }
    
    def test_basic_functionality(self, sample_data):
        """Test basic feature functionality."""
        feature = NewFeature()
        result = feature.process(sample_data)
        
        assert result is not None
        assert result['status'] == 'success'
    
    @patch('ai_trading.new_feature.external_api_call')
    def test_with_mocked_api(self, mock_api, sample_data):
        """Test with mocked external dependencies."""
        mock_api.return_value = {'data': 'mocked'}
        
        feature = NewFeature()
        result = feature.process_with_api(sample_data)
        
        mock_api.assert_called_once()
        assert result['data'] == 'mocked'
    
    @pytest.mark.slow
    def test_heavy_computation(self):
        """Test computationally intensive operations."""
        # Mark slow tests that take significant time
        pass
```

---

## ⚡ Performance Optimization

### Profiling Tools

```bash
# CPU profiling
python -m cProfile -o profile.stats -m ai_trading
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"

# Memory profiling
pip install memory-profiler
python -m memory_profiler bot_engine.py

# Line-by-line profiling
kernprof -l -v bot_engine.py

# Real-time performance monitoring
python performance_optimizer.py --monitor --duration 3600
```

### Optimization Techniques

1. **Parallel Processing**
   ```python
   # Enable parallel indicator calculations
   USE_PARALLEL_PROCESSING=true
   MAX_WORKER_THREADS=4
   ```

2. **Data Caching**
   ```python
   # Cache frequently accessed data
   ENABLE_DATA_CACHE=true
   CACHE_TTL_SECONDS=300
   ```

3. **Memory Management**
   ```python
   # Periodic cleanup
   ENABLE_MEMORY_CLEANUP=true
   CLEANUP_INTERVAL_MINUTES=60
   ```

### Performance Metrics

Monitor key performance indicators:

```python
# Monitor with Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

trade_counter = Counter('trades_total', 'Total trades executed')
execution_time = Histogram('trade_execution_seconds', 'Trade execution time')
memory_usage = Gauge('memory_usage_bytes', 'Current memory usage')
```

---

## 🤝 Contributing

### Development Guidelines

1. **Follow PEP 8** coding standards
2. **Add comprehensive tests** for new features
3. **Update documentation** for API changes
4. **Use type hints** throughout the codebase
5. **Follow the existing logging patterns**

### Code Quality Standards

```bash
# Automated code quality checks
make lint        # Run all linters
make format      # Auto-format code
make type-check  # Run mypy type checking
make security    # Security vulnerability scan
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes with tests
4. Run the full test suite: `make test-all`
5. Update documentation as needed
6. Submit a pull request with clear description

### AI-Powered Maintenance

This project follows the [**AI-Only Maintenance Policy**](AGENTS.md):

- Exclusively maintained by Dom with AI assistance (Codex/GPT-4o)
- All automated refactoring and enhancements follow strict safety guidelines
- Core trading logic files have special protection against full rewrites
- Comprehensive testing discipline is maintained

**For AI agents working on this repository**: Strictly follow the directives in [`AGENTS.md`](./AGENTS.md) to maintain trading safety, logging integrity, and system reliability.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🆘 Support & Community

### Getting Help

1. **📚 Documentation**: Check our comprehensive guides
   - [Architecture Guide](ARCHITECTURE.md)
   - [API Documentation](API_DOCUMENTATION.md)  
   - [Deployment Guide](DEPLOYMENT.md)
   - [Troubleshooting Guide](TROUBLESHOOTING.md)

2. **🐛 Issues**: Report bugs via [GitHub Issues](https://github.com/dmorazzini23/ai-trading-bot/issues)

3. **💬 Discussions**: Join [GitHub Discussions](https://github.com/dmorazzini23/ai-trading-bot/discussions)

### Quick Links

- **🏠 Homepage**: [AI Trading Bot Repository](https://github.com/dmorazzini23/ai-trading-bot)
- **📖 Wiki**: [Project Wiki](https://github.com/dmorazzini23/ai-trading-bot/wiki)
- **🚀 Releases**: [Latest Releases](https://github.com/dmorazzini23/ai-trading-bot/releases)
- **📊 Issues**: [Issue Tracker](https://github.com/dmorazzini23/ai-trading-bot/issues)

### Acknowledgments

- **Alpaca Markets** for providing trading infrastructure
- **Python Community** for excellent libraries and tools
- **Open Source Contributors** who make projects like this possible

---

## ⚠️ Disclaimer

**IMPORTANT RISK DISCLOSURE:**

- This software is for educational and research purposes
- Trading involves substantial risk of loss
- Past performance does not guarantee future results
- Never trade with money you cannot afford to lose
- The authors are not responsible for any financial losses
- Always thoroughly test strategies before using real money
- Consider consulting with a financial advisor

**Use at your own risk. Start with paper trading.**

---

*Happy Trading! 🎯📈*
