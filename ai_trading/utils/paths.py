"""Path utilities for centralized data and log directory management."""
from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    """Get the data directory path from environment or default."""
    return Path(os.getenv("AI_TRADING_DATA_DIR", "var/data")).resolve()


def log_dir() -> Path:
    """Get the log directory path from environment or default."""
    return Path(os.getenv("AI_TRADING_LOG_DIR", "var/logs")).resolve()