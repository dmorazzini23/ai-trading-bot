"""
Workers module for AI trading bot.
Provides worker functions for trade execution.
"""
from __future__ import annotations

# Import the actual worker from bot_engine 
from ai_trading.core.bot_engine import run_all_trades_worker

__all__ = ["run_all_trades_worker"]