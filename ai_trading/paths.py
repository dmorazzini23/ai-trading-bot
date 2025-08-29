"""Runtime paths for AI Trading Bot.

Defines writable data, log, and cache directories with environment overrides.
Creates directories at import time.
"""
from ai_trading.logging import get_logger
import errno
import os
import tempfile
from pathlib import Path
logger = get_logger(__name__)
BASE_DIR = Path(__file__).resolve().parents[1]
APP_NAME = 'ai-trading-bot'

def _ensure_dir(path: Path) -> Path:
    """Create *path* with 0700 perms, falling back on read-only errors."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        try:
            if os.access(path, os.W_OK):
                path.chmod(0o700)
            else:
                logger.debug("Skipping chmod for non-writable path %s", path)
        except OSError as perm_err:  # best effort permission fix
            logger.debug("chmod failed for %s: %s", path, perm_err)
        return path
    except OSError as e:
        if e.errno == errno.EROFS:
            fallback = Path(tempfile.gettempdir()) / APP_NAME
            fallback.mkdir(parents=True, exist_ok=True)
            try:
                if os.access(fallback, os.W_OK):
                    fallback.chmod(0o700)
                else:
                    logger.debug("Skipping chmod for non-writable path %s", fallback)
            except OSError as perm_err:  # pragma: no cover - unlikely
                logger.debug("chmod failed for %s: %s", fallback, perm_err)
            logger.error(
                "Persistent directory %s is not writable (%s); using temp dir %s. Set AI_TRADING_*_DIR env vars to a writable location.",
                path,
                e,
                fallback,
            )
            return fallback
        logger.debug("Directory creation failed for %s: %s", path, e)
        return path

def _first_env_path(*names: str) -> Path | None:
    for n in names:
        v = os.getenv(n)
        if v:
            return Path(v.split(':')[0])
    return None

def _default_state_dir() -> Path:
    if os.geteuid() == 0:
        return Path('/var/lib') / APP_NAME
    return Path.home() / '.local' / 'share' / APP_NAME

def _default_cache_dir() -> Path:
    if os.geteuid() == 0:
        return Path('/var/cache') / APP_NAME
    xdg = os.getenv('XDG_CACHE_HOME')
    return (Path(xdg) if xdg else Path.home() / '.cache') / APP_NAME

def _default_log_dir() -> Path:
    if os.geteuid() == 0:
        return Path('/var/log') / APP_NAME
    return _default_state_dir() / 'logs'


def _resolve_cache_dir() -> Path:
    """Return cache directory from env or defaults."""
    env_path = _first_env_path('AI_TRADING_CACHE_DIR', 'CACHE_DIRECTORY')
    return _ensure_dir(env_path or _default_cache_dir())


DATA_DIR = _ensure_dir(
    _first_env_path('AI_TRADING_DATA_DIR', 'STATE_DIRECTORY') or _default_state_dir()
)
CACHE_DIR = _resolve_cache_dir()
LOG_DIR = _ensure_dir(
    _first_env_path('AI_TRADING_LOG_DIR', 'LOGS_DIRECTORY') or _default_log_dir()
)
MODELS_DIR = _ensure_dir(Path(os.getenv('AI_TRADING_MODELS_DIR', DATA_DIR / 'models')))
OUTPUT_DIR = _ensure_dir(Path(os.getenv('AI_TRADING_OUTPUT_DIR', DATA_DIR / 'output')))
DB_PATH = Path(os.getenv('AI_TRADING_DB_PATH', DATA_DIR / 'trades.db'))
SLIPPAGE_LOG_PATH = Path(os.getenv('SLIPPAGE_LOG_PATH', LOG_DIR / 'slippage.csv'))
TICKERS_FILE_PATH = Path(os.getenv('TICKERS_FILE_PATH', DATA_DIR / 'tickers.csv'))
