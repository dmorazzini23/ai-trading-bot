"""Path utilities for centralized data and log directory management."""
from __future__ import annotations
from pathlib import Path

from ai_trading.config.management import get_env


def data_dir() -> Path:
    """Get the data directory path from environment or default."""
    return Path(get_env("AI_TRADING_DATA_DIR", "var/data", cast=str)).resolve()


def log_dir() -> Path:
    """Get the log directory path from environment or default."""
    return Path(get_env("AI_TRADING_LOG_DIR", "var/logs", cast=str)).resolve()
