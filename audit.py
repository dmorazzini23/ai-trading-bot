import csv
import logging
import os
import uuid
from datetime import datetime, timezone

from validate_env import settings

TRADE_LOG_FILE = settings.TRADE_LOG_FILE

logger = logging.getLogger(__name__)
_fields = ["id", "timestamp", "symbol", "side", "qty", "price", "mode", "result"]


def log_trade(self, trade):
    # AI-AGENT-REF: track trades and enforce drawdown limit
    self.trades.append(trade)
    total_pnl = sum([t['pnl'] for t in self.trades])
    if total_pnl < -0.1 * self.starting_capital:
        print('🚨 Max drawdown hit. Trading halted.')
        raise SystemExit

