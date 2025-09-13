from ai_trading.logging import get_logger
import os
import threading
from typing import Any

from .locks import LockWithTimeout
# AI-AGENT-REF: re-export config management helpers without triggering optional deps
from .management import TradingConfig, derive_cap_from_settings
from .settings import Settings, broker_keys, get_settings
from .alpaca import AlpacaConfig, get_alpaca_config
from .aliases import resolve_trading_mode
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


def _parse_float_env(val: str) -> float:
    """Parse a float environment value, ignoring inline comments."""
    token = val.split("#", 1)[0].strip()
    return float(token)


def _require_float_env(name: str) -> float:
    """Fetch a required environment variable and convert to float."""
    val = os.getenv(name)
    if val is None or val == "":
        raise RuntimeError(f"Missing required env var: {name}")
    try:
        return _parse_float_env(val)
    except ValueError as e:
        raise RuntimeError(f"Invalid value for {name}: {val}") from e


def _optional_float_env(name: str, default: float) -> float:
    """Fetch an optional float environment variable."""
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return _parse_float_env(val)
    except ValueError as e:
        raise RuntimeError(f"Invalid value for {name}: {val}") from e


MAX_DRAWDOWN_THRESHOLD = _require_float_env("MAX_DRAWDOWN_THRESHOLD")

_MODE_PRESETS = {
    "conservative": {
        "kelly_fraction": 0.25,
        "conf_threshold": 0.85,
        "daily_loss_limit": 0.03,
        "capital_cap": 0.20,
        "max_position_size": 5000.0,
    },
    "balanced": {
        "kelly_fraction": 0.6,
        "conf_threshold": 0.75,
        "daily_loss_limit": 0.05,
        "capital_cap": 0.25,
        "max_position_size": 8000.0,
    },
    "aggressive": {
        "kelly_fraction": 0.75,
        "conf_threshold": 0.65,
        "daily_loss_limit": 0.08,
        "capital_cap": 0.30,
        "max_position_size": 12000.0,
    },
}

TRADING_MODE = resolve_trading_mode("balanced").lower()
if TRADING_MODE not in _MODE_PRESETS:
    raise RuntimeError(f"Invalid TRADING_MODE: {TRADING_MODE}")

_mode_defaults = _MODE_PRESETS[TRADING_MODE]
KELLY_FRACTION = _optional_float_env("KELLY_FRACTION", _mode_defaults["kelly_fraction"])
CONF_THRESHOLD = _optional_float_env("CONF_THRESHOLD", _mode_defaults["conf_threshold"])
MAX_POSITION_SIZE = _optional_float_env(
    "MAX_POSITION_SIZE", _mode_defaults["max_position_size"]
)
DAILY_LOSS_LIMIT = _optional_float_env(
    "DAILY_LOSS_LIMIT", _mode_defaults["daily_loss_limit"]
)
CAPITAL_CAP = _optional_float_env("CAPITAL_CAP", _mode_defaults["capital_cap"])

MODE_PARAMETERS = {k: v["conf_threshold"] for k, v in _MODE_PRESETS.items()}
SENTIMENT_API_KEY = os.getenv('SENTIMENT_API_KEY')
SENTIMENT_API_URL = os.getenv('SENTIMENT_API_URL')
SENTIMENT_ENHANCED_CACHING = True
SENTIMENT_RECOVERY_TIMEOUT_SECS = 3600
SENTIMENT_FALLBACK_SOURCES = []
META_LEARNING_BOOTSTRAP_ENABLED = True
META_LEARNING_MIN_TRADES_REDUCED = 10
SENTIMENT_SUCCESS_RATE_TARGET = 0.90
META_LEARNING_BOOTSTRAP_WIN_RATE = 0.55

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
    # Respect module-level overrides used in tests; fall back to process env
    env = dict(os.environ)
    for k_mod, k_env in (
        ("ALPACA_API_KEY", "ALPACA_API_KEY"),
        ("ALPACA_SECRET_KEY", "ALPACA_SECRET_KEY"),
        ("ALPACA_BASE_URL", "ALPACA_API_URL"),
    ):
        try:
            val = globals().get(k_mod)
        except Exception:
            val = None
        if val is not None:
            env[k_env] = str(val)
    validate_required_env(("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_API_URL"), env=env)

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
# Public API surface intentionally small; rarely used constants remain module attrs
# but are omitted here to discourage imports from this namespace.
__all__ = [
    '_require_env_vars',
    'AlpacaConfig',
    'MAX_DRAWDOWN_THRESHOLD',
    'META_LEARNING_BOOTSTRAP_ENABLED',
    'META_LEARNING_BOOTSTRAP_WIN_RATE',
    'META_LEARNING_MIN_TRADES_REDUCED',
    'MODE_PARAMETERS',
    'ORDER_FILL_RATE_TARGET',
    'SENTIMENT_API_KEY',
    'SENTIMENT_API_URL',
    'SENTIMENT_ENHANCED_CACHING',
    'SENTIMENT_FALLBACK_SOURCES',
    'SENTIMENT_RECOVERY_TIMEOUT_SECS',
    'SENTIMENT_SUCCESS_RATE_TARGET',
    'Settings',
    'TradingConfig',
    'broker_keys',
    'derive_cap_from_settings',
    'get_alpaca_config',
    'get_env',
    'get_settings',
    'log_config',
    'reload_env',
    'require_env_vars',
    'validate_alpaca_credentials',
    'validate_environment',
    'validate_env_vars',
]
