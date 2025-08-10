"""
Runtime paths for AI Trading Bot.

Defines writable data, log, and cache directories with environment overrides.
Creates directories at import time.
"""
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
APP_NAME = "ai-trading-bot"

def _ensure_dir(path: Path) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        # In restricted environments (e.g., systemd with ProtectSystem/ProtectHome),
        # creation may be disallowed. Log the issue but return the path
        logger.debug("Directory creation failed for %s: %s", path, e)
    return path

def _first_env_path(*names: str) -> Path | None:
    for n in names:
        v = os.getenv(n)
        if v:
            # systemd can provide colon-separated list; take the first
            return Path(v.split(":")[0])
    return None

def _default_state_dir() -> Path:
    if os.geteuid() == 0:
        return Path("/var/lib") / APP_NAME
    return Path.home() / ".local" / "share" / APP_NAME

def _default_cache_dir() -> Path:
    if os.geteuid() == 0:
        return Path("/var/cache") / APP_NAME
    xdg = os.getenv("XDG_CACHE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".cache") / APP_NAME

def _default_log_dir() -> Path:
    if os.geteuid() == 0:
        return Path("/var/log") / APP_NAME
    return _default_state_dir() / "logs"

DATA_DIR = _ensure_dir(
    _first_env_path("AI_TRADING_DATA_DIR", "STATE_DIRECTORY") or _default_state_dir()
)
CACHE_DIR = _ensure_dir(
    _first_env_path("AI_TRADING_CACHE_DIR", "CACHE_DIRECTORY") or _default_cache_dir()
)
LOG_DIR = _ensure_dir(
    _first_env_path("AI_TRADING_LOG_DIR", "LOGS_DIRECTORY") or _default_log_dir()
)

MODELS_DIR = _ensure_dir(Path(os.getenv("AI_TRADING_MODELS_DIR", DATA_DIR / "models")))
OUTPUT_DIR = _ensure_dir(Path(os.getenv("AI_TRADING_OUTPUT_DIR", DATA_DIR / "output")))
DB_PATH = Path(os.getenv("AI_TRADING_DB_PATH", DATA_DIR / "trades.db"))
SLIPPAGE_LOG_PATH = Path(os.getenv("SLIPPAGE_LOG_PATH", LOG_DIR / "slippage.csv"))
TICKERS_FILE_PATH = Path(os.getenv("TICKERS_FILE_PATH", DATA_DIR / "tickers.csv"))