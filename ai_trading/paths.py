"""
Runtime paths for AI Trading Bot.

Defines writable data, log, and cache directories with environment overrides.
Creates directories at import time.
"""
import os
from pathlib import Path


def _ensure_dir(path: Path) -> Path:
    """Ensure directory exists and return the Path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# Base runtime directory - defaults to current working directory
BASE_DIR = Path(os.getenv("AI_TRADING_BASE_DIR", os.getcwd()))

# Data directory for models, signals, weights, etc.
DATA_DIR = _ensure_dir(Path(os.getenv("AI_TRADING_DATA_DIR", BASE_DIR / "data")))

# Log directory for trade logs, application logs, etc.
LOG_DIR = _ensure_dir(Path(os.getenv("AI_TRADING_LOG_DIR", BASE_DIR / "logs")))

# Cache directory for temporary files, caches, etc.
CACHE_DIR = _ensure_dir(Path(os.getenv("AI_TRADING_CACHE_DIR", BASE_DIR / "cache")))

# AI-AGENT-REF: Created runtime paths module for proper directory separation