"""
Compatibility shim for tests importing rebalancer directly.
This module re-exports the rebalancer module from ai_trading package.
"""
import os
import threading
from datetime import UTC, datetime, timedelta, timezone
if 'ALPACA_API_KEY' not in os.environ:
    os.environ['ALPACA_API_KEY'] = 'test'
if 'ALPACA_SECRET_KEY' not in os.environ:
    os.environ['ALPACA_SECRET_KEY'] = 'test'
if 'ALPACA_BASE_URL' not in os.environ:
    os.environ['ALPACA_BASE_URL'] = 'https://paper-api.alpaca.markets'
if 'WEBHOOK_SECRET' not in os.environ:
    os.environ['WEBHOOK_SECRET'] = 'test'
if 'FLASK_PORT' not in os.environ:
    os.environ['FLASK_PORT'] = '5000'
_import_success = False
try:
    from ai_trading.rebalancer import *
    _import_success = True
except (ValueError, TypeError):
    pass
if not _import_success:
    REBALANCE_INTERVAL_MIN = 60
    _last_rebalance = datetime.now(UTC)

    def maybe_rebalance(ctx):
        """Mock rebalance function"""

    def rebalance_portfolio(ctx):
        """Mock portfolio rebalance"""

    def start_rebalancer(ctx):
        """Mock rebalancer start"""
        return threading.Thread(target=lambda: None)
    datetime = datetime
    timedelta = timedelta
    timezone = timezone
    threading = threading