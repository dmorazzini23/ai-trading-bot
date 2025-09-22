"""Runtime path helpers for AI Trading Bot.

All writable directories are derived from systemd-provided environment
variables.  Directories are created with ``0700`` permissions when missing and
validated for writability on import so startup fails fast on misconfiguration.
"""

from __future__ import annotations

import contextlib
import errno
import os
import tempfile
from pathlib import Path

from ai_trading.logging import get_logger

logger = get_logger(__name__)
APP_NAME = "ai-trading-bot"


def _ensure_dir(path: Path) -> Path:
    """Ensure *path* exists, is absolute, and writable with 0700 perms."""

    if not path.is_absolute():
        raise RuntimeError(f"Runtime directory must be absolute: {path}")
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        if exc.errno in (errno.EROFS, errno.EACCES, errno.EPERM):
            raise RuntimeError(f"Directory {path} is not writable: {exc}") from exc
        raise
    if not os.access(path, os.W_OK):
        raise RuntimeError(f"Directory {path} is not writable")
    with contextlib.suppress(PermissionError):
        path.chmod(0o700)
    return path


def _fallback_tmp_dir() -> Path:
    base = os.getenv("TMPDIR") or tempfile.gettempdir()
    fallback = Path(base).expanduser() / APP_NAME
    fallback.mkdir(mode=0o700, parents=True, exist_ok=True)
    with contextlib.suppress(PermissionError):
        fallback.chmod(0o700)
    return fallback


def _resolve_from_env(names: tuple[str, ...], default: Path) -> Path:
    for name in names:
        raw = os.getenv(name)
        if not raw:
            continue
        candidate = Path(raw.split(":")[0]).expanduser()
        try:
            return _ensure_dir(candidate)
        except RuntimeError as exc:
            raise RuntimeError(f"Environment path {name} invalid: {exc}") from exc
    try:
        return _ensure_dir(default)
    except RuntimeError as exc:
        logger.error("Default path %s unusable: %s", default, exc)
        return _fallback_tmp_dir()


def _default_data_dir() -> Path:
    return Path("/var/lib") / APP_NAME


def _default_cache_dir() -> Path:
    cache_home = os.getenv("XDG_CACHE_HOME")
    if cache_home:
        return Path(cache_home)
    return Path("/var/cache") / APP_NAME


def _default_log_dir() -> Path:
    return Path("/var/log") / APP_NAME


DATA_DIR = _resolve_from_env(("AI_TRADING_DATA_DIR", "STATE_DIRECTORY"), _default_data_dir())
CACHE_DIR = _resolve_from_env(("AI_TRADING_CACHE_DIR", "CACHE_DIRECTORY", "XDG_CACHE_HOME"), _default_cache_dir())
LOG_DIR = _resolve_from_env(("AI_TRADING_LOG_DIR", "LOGS_DIRECTORY"), _default_log_dir())


def _resolve_optional_dir(env: str, default: Path) -> Path:
    raw = os.getenv(env)
    if not raw:
        return _ensure_dir(default)
    candidate = Path(raw).expanduser()
    try:
        return _ensure_dir(candidate)
    except RuntimeError as exc:
        raise RuntimeError(f"Environment path {env} invalid: {exc}") from exc


MODELS_DIR = _resolve_optional_dir("AI_TRADING_MODELS_DIR", DATA_DIR / "models")
OUTPUT_DIR = _resolve_optional_dir("AI_TRADING_OUTPUT_DIR", DATA_DIR / "output")

DB_PATH = Path(os.getenv("AI_TRADING_DB_PATH", str(DATA_DIR / "trades.db")))
SLIPPAGE_LOG_PATH = Path(os.getenv("SLIPPAGE_LOG_PATH", str(LOG_DIR / "slippage.csv")))
TICKERS_FILE_PATH = Path(os.getenv("TICKERS_FILE_PATH", str(DATA_DIR / "tickers.csv")))


_INITIALIZED = False


def ensure_runtime_paths() -> None:
    """Idempotently verify runtime directories."""

    global _INITIALIZED
    if _INITIALIZED:
        return
    for path in (DATA_DIR, CACHE_DIR, LOG_DIR, MODELS_DIR, OUTPUT_DIR):
        _ensure_dir(path)
    _INITIALIZED = True


ensure_runtime_paths()
