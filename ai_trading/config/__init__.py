from ai_trading.logging import get_logger
import os
import threading
from typing import Any

from .locks import LockWithTimeout
# AI-AGENT-REF: re-export config management helpers without triggering optional deps
from .management import TradingConfig, derive_cap_from_settings
from ai_trading.validation.require_env import _require_env_vars, require_env_vars
logger = get_logger(__name__)
_LOCK_TIMEOUT = 30
_ENV_LOCK = LockWithTimeout(_LOCK_TIMEOUT)
_lock_state = threading.local()
_CONFIG_LOGGED = False
LIQUIDITY_SPREAD_THRESHOLD = 0.01
LIQUIDITY_VOL_THRESHOLD = 1000
LIQUIDITY_REDUCTION_AGGRESSIVE = 0.75
LIQUIDITY_REDUCTION_MODERATE = 0.9
ORDER_TIMEOUT_SECONDS = 300
ORDER_STALE_CLEANUP_INTERVAL = 60
ORDER_FILL_RATE_TARGET = 0.8
MAX_DRAWDOWN_THRESHOLD = 0.08
MODE_PARAMETERS = {'conservative': 0.85, 'balanced': 0.75, 'aggressive': 0.65}
SENTIMENT_API_KEY = os.getenv('SENTIMENT_API_KEY')
SENTIMENT_API_URL = os.getenv('SENTIMENT_API_URL')
SENTIMENT_ENHANCED_CACHING = True
SENTIMENT_RECOVERY_TIMEOUT_SECS = 3600
SENTIMENT_FALLBACK_SOURCES = []
META_LEARNING_BOOTSTRAP_ENABLED = True
META_LEARNING_MIN_TRADES_REDUCED = True
SENTIMENT_SUCCESS_RATE_TARGET = 0.8
META_LEARNING_BOOTSTRAP_WIN_RATE = 0.55

def __getattr__(name: str):
    if name == 'TradingConfig':
        from .management import TradingConfig as _TC
        return _TC
    if name in {'Settings', 'broker_keys', 'get_settings'}:
        from .settings import Settings as _S, broker_keys as _bk, get_settings as _gs

        return {'Settings': _S, 'broker_keys': _bk, 'get_settings': _gs}[name]
    if name in {'AlpacaConfig', 'get_alpaca_config'}:
        from .alpaca import AlpacaConfig as _AC, get_alpaca_config as _gac

        return {'AlpacaConfig': _AC, 'get_alpaca_config': _gac}[name]
    raise AttributeError(name)

def _is_lock_held_by_current_thread() -> bool:
    return bool(getattr(_lock_state, 'held', False))

def _set_lock_held_by_current_thread(val: bool) -> None:
    _lock_state.held = bool(val)

def reload_env() -> None:
    """Reload .env if python-dotenv is present; ignore failures."""
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except (KeyError, ValueError, TypeError):
        pass

def get_env(name: str, default: Any=None, *, reload: bool=False, required: bool=False) -> Any:
    """Return env var; if reload=True, call reload_env() first."""
    if reload:
        reload_env()
    val = os.getenv(name, default)
    if required and val is None:
        raise RuntimeError(f'Missing required env var: {name}')
    return val


def _perform_env_validation() -> None:
    from .management import validate_required_env

    validate_required_env()

def validate_environment() -> None:
    """Validate required environment variables with deadlock-safe locking."""
    if _is_lock_held_by_current_thread():
        _perform_env_validation()
        return
    with _ENV_LOCK:
        _set_lock_held_by_current_thread(True)
        try:
            try:
                from .settings import get_settings as _gs

                _gs.cache_clear()
            except (Exception):
                pass
            _perform_env_validation()
        finally:
            _set_lock_held_by_current_thread(False)

def validate_alpaca_credentials() -> None:
    from .management import validate_required_env

    validate_required_env(("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_API_URL"))

def validate_env_vars() -> None:
    return validate_environment()

def log_config(masked_keys: list[str] | None=None, secrets_to_redact: list[str] | None=None) -> dict:
    """
    Return a sanitized snapshot of current config for diagnostics.
    MUST NOT log or print in tests.
    """
    global _CONFIG_LOGGED
    _CONFIG_LOGGED = True
    from .settings import get_settings as _gs

    s = _gs()
    conf = {'ALPACA_API_KEY': '***' if s.alpaca_api_key else '', 'ALPACA_SECRET_KEY': '***REDACTED***' if s.alpaca_secret_key else '', 'ALPACA_API_URL': s.alpaca_base_url or '', 'CAPITAL_CAP': getattr(s, 'capital_cap', None) or 0.25, 'CONF_THRESHOLD': getattr(s, 'conf_threshold', None) or 0.75, 'DAILY_LOSS_LIMIT': getattr(s, 'daily_loss_limit', None) or 0.03}
    if masked_keys is None and secrets_to_redact is not None:
        masked_keys = secrets_to_redact
    if masked_keys:
        for key in masked_keys:
            if key in conf:
                conf[key] = '***'
    return conf
__all__ = ['Settings', 'get_settings', 'broker_keys', 'get_alpaca_config', 'AlpacaConfig', 'TradingConfig', 'derive_cap_from_settings', 'get_env', '_require_env_vars', 'require_env_vars', 'reload_env', 'validate_environment', 'validate_alpaca_credentials', 'validate_env_vars', 'log_config', 'ORDER_FILL_RATE_TARGET', 'MAX_DRAWDOWN_THRESHOLD', 'MODE_PARAMETERS', 'SENTIMENT_API_KEY', 'SENTIMENT_API_URL', 'SENTIMENT_ENHANCED_CACHING', 'SENTIMENT_RECOVERY_TIMEOUT_SECS', 'SENTIMENT_FALLBACK_SOURCES', 'META_LEARNING_BOOTSTRAP_ENABLED', 'META_LEARNING_MIN_TRADES_REDUCED', 'SENTIMENT_SUCCESS_RATE_TARGET', 'META_LEARNING_BOOTSTRAP_WIN_RATE']
