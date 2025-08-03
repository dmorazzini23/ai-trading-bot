import logging
import os
import threading
import time
from pathlib import Path

# Optional import: avoid import error when dotenv is missing.
try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover - when python-dotenv is not installed
    def load_dotenv(*args, **kwargs):  # type: ignore[override]
        """Fallback no-op for ``load_dotenv``.

        This stub allows the configuration module to be imported in
        environments where ``python-dotenv`` is not installed.  Tests
        that rely on environment variables should set them directly via
        ``os.environ``.
        """
        return False
logger = logging.getLogger(__name__)

# AI-AGENT-REF: Add thread-safe configuration validation locking with timeout support
_CONFIG_VALIDATION_LOCK = threading.Lock()
_LOCK_TIMEOUT = 30  # seconds
_validation_in_progress = threading.local()


def _is_lock_held_by_current_thread():
    """Check if the current thread already holds the validation lock."""
    return getattr(_validation_in_progress, 'has_lock', False)


def _set_lock_held_by_current_thread(held):
    """Mark that the current thread holds/releases the validation lock."""
    _validation_in_progress.has_lock = held

# AI-AGENT-REF: robust import handling for pydantic-settings to prevent hangs
try:
    from pydantic_settings import BaseSettings
    _PYDANTIC_AVAILABLE = True
except ImportError:
    logger.warning("pydantic-settings not available, using fallback")
    _PYDANTIC_AVAILABLE = False
    BaseSettings = object  # Minimal fallback

# Import validate_env with fallback handling
try:
    from validate_env import settings as env_settings
except Exception as e:
    logger.warning("validate_env import failed: %s, using fallback", e)
    # Create a minimal fallback settings object
    class _FallbackSettings:
        ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
        ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
        ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex")
        FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
        FUNDAMENTAL_API_KEY = os.getenv("FUNDAMENTAL_API_KEY")
        NEWS_API_KEY = os.getenv("NEWS_API_KEY")
        IEX_API_TOKEN = os.getenv("IEX_API_TOKEN")
        BOT_MODE = os.getenv("BOT_MODE", "balanced")
        DOLLAR_RISK_LIMIT = float(os.getenv("DOLLAR_RISK_LIMIT", "0.02"))
        BUY_THRESHOLD = float(os.getenv("BUY_THRESHOLD", "0.5"))
        WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
        TRADE_LOG_FILE = os.getenv("TRADE_LOG_FILE", "test_trades.csv")  # AI-AGENT-REF: add missing TRADE_LOG_FILE
        SEED = int(os.getenv("SEED", "42"))  # AI-AGENT-REF: add missing SEED 
        RATE_LIMIT_BUDGET = int(os.getenv("RATE_LIMIT_BUDGET", "190"))  # AI-AGENT-REF: add missing RATE_LIMIT_BUDGET
        REBALANCE_INTERVAL_MIN = int(os.getenv("REBALANCE_INTERVAL_MIN", "60"))  # AI-AGENT-REF: add missing value
        SHADOW_MODE = os.getenv("SHADOW_MODE", "True").lower() in ("true", "1")  # AI-AGENT-REF: add missing value
        DISABLE_DAILY_RETRAIN = os.getenv("DISABLE_DAILY_RETRAIN", "False").lower() in ("true", "1")  # AI-AGENT-REF: add missing value
        # Add other commonly accessed attributes as needed
        def __getattr__(self, name):
            # Fallback for any missing attributes
            return os.getenv(name)
    env_settings = _FallbackSettings()

if _PYDANTIC_AVAILABLE:
    class Settings(BaseSettings):
        """Runtime configuration via environment."""

    settings = Settings()
else:
    # Use a minimal fallback when pydantic is not available
    settings = type('Settings', (), {})()

ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"
# Load environment variables once at import. Individual scripts
# should call ``reload_env()`` to refresh values if needed.
load_dotenv(ENV_PATH)

# Relax environment variable validation when running under pytest so the
# configuration module can be imported without all production variables set.
TESTING = os.getenv("PYTEST_CURRENT_TEST") is not None or os.getenv("TESTING")

required_env_vars = [
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "ALPACA_BASE_URL",
    "WEBHOOK_SECRET",
]
if not TESTING:
    required_env_vars.append("FLASK_PORT")

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    if not TESTING:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")
    else:
        logger.warning("Missing environment variables in test mode: %s", missing_vars)

# Validate critical numeric environment variables
try:
    if os.getenv('FLASK_PORT'):
        port = int(os.getenv('FLASK_PORT'))
        if not (1024 <= port <= 65535):
            raise ValueError(f'FLASK_PORT must be between 1024 and 65535, got {port}')
except ValueError as e:
    if not TESTING:
        raise RuntimeError(f'Invalid FLASK_PORT: {e}')
REQUIRED_ENV_VARS = [
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "ALPACA_BASE_URL",
    "WEBHOOK_SECRET",
]


def get_env(
    key: str,
    default: str | None = None,
    *,
    reload: bool = False,
    required: bool = False,
) -> str | None:
    """Return environment variable ``key``.

    Parameters
    ----------
    key : str
        Name of the variable.
    default : str | None, optional
        Value returned if the variable is missing.
    reload : bool, optional
        Reload ``.env`` before checking when ``True``.
    required : bool, optional
        If ``True`` and the variable is missing, raise ``RuntimeError``.
    """
    if reload:
        reload_env()
    value = os.environ.get(key, default)
    if required and value is None:
        logger.error("Required environment variable '%s' is missing", key)
        raise RuntimeError(f"Required environment variable '{key}' is missing")
    return value


def reload_env() -> None:
    """Reload environment variables from the .env file if it exists."""
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=True)


from types import MappingProxyType


def mask_secret(value: str, show_last: int = 4) -> str:
    """Return ``value`` with all but the last ``show_last`` characters masked."""
    if value is None:
        return ""
    return "*" * max(0, len(value) - show_last) + value[-show_last:]


_CONFIG_LOGGED = False


def log_config(keys: list[str] | None = None) -> None:
    """Log selected configuration values with secrets hidden."""
    global _CONFIG_LOGGED
    if _CONFIG_LOGGED:
        return
    if keys is None:
        keys = REQUIRED_ENV_VARS
    cfg = {}
    for k in keys:
        val = os.getenv(k, "")
        if "KEY" in k or "SECRET" in k:
            # AI-AGENT-REF: avoid logging secret values entirely
            val = "<hidden>"
        cfg[k] = val
    _CONFIG_LOGGED = True


def _require_env_vars(*keys: str) -> None:
    missing = [v for v in keys if not os.environ.get(v)]
    if missing:
        logger.critical("Missing required environment variables: %s", missing)
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )


def validate_environment() -> None:
    """Validate that mandatory environment variables are present."""
    start_time = time.time()
    logger.info("Starting environment validation")
    
    # Check if we already hold the lock to prevent deadlock
    if _is_lock_held_by_current_thread():
        logger.debug("Validation lock already held by current thread, proceeding without re-acquiring")
        _validate_environment_core()
        return
    
    # Attempt to acquire lock with timeout
    logger.debug("Attempting to acquire configuration validation lock (timeout: %d seconds)", _LOCK_TIMEOUT)
    lock_acquired = False
    
    try:
        lock_acquired = _CONFIG_VALIDATION_LOCK.acquire(timeout=_LOCK_TIMEOUT)
        if not lock_acquired:
            error_msg = f"Failed to acquire configuration validation lock within {_LOCK_TIMEOUT} seconds"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.debug("Configuration validation lock acquired successfully")
        _set_lock_held_by_current_thread(True)
        
        # Perform the actual validation
        _validate_environment_core()
        
        elapsed = time.time() - start_time
        logger.info("Environment validation completed successfully in %.2f seconds", elapsed)
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("Environment validation failed after %.2f seconds: %s", elapsed, e)
        raise
    finally:
        if lock_acquired:
            _set_lock_held_by_current_thread(False)
            _CONFIG_VALIDATION_LOCK.release()
            logger.debug("Configuration validation lock released")


def _validate_environment_core() -> None:
    """Core environment validation logic without lock handling."""
    logger.debug("Starting core environment validation")
    
    # AI-AGENT-REF: handle missing pydantic gracefully
    if not _PYDANTIC_AVAILABLE:
        logger.warning("Pydantic unavailable, performing basic validation only")
        # Basic validation without pydantic
        basic_vars = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]
        logger.debug("Validating basic environment variables: %s", basic_vars)
        for var in basic_vars:
            if not os.getenv(var):
                error_msg = f"Missing required environment variable: {var}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        logger.info("Basic environment validation completed (pydantic unavailable)")
        return
        
    logger.debug("Validating required environment variables: %s", REQUIRED_ENV_VARS)
    missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        error_msg = f"Missing required environment variables: {', '.join(missing)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.debug("All required environment variables are present")
    
    # Validate API key formats
    logger.debug("Validating API key formats")
    api_key = os.getenv("ALPACA_API_KEY", "")
    if api_key and len(api_key) < 10:
        error_msg = "ALPACA_API_KEY appears to be invalid (too short)"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    if secret_key and len(secret_key) < 10:
        error_msg = "ALPACA_SECRET_KEY appears to be invalid (too short)"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Validate base URL format
    logger.debug("Validating base URL format")
    base_url = os.getenv("ALPACA_BASE_URL", "")
    if base_url and not base_url.startswith(("http://", "https://")):
        error_msg = "ALPACA_BASE_URL must start with http:// or https://"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Validate webhook secret
    logger.debug("Validating webhook secret")
    webhook_secret = os.getenv("WEBHOOK_SECRET", "")
    if webhook_secret and len(webhook_secret) < 8:
        error_msg = "WEBHOOK_SECRET must be at least 8 characters"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.debug("Core environment validation completed successfully")


# AI-AGENT-REF: robust environment variable access
ALPACA_API_KEY = getattr(env_settings, 'ALPACA_API_KEY', os.getenv("ALPACA_API_KEY", ""))
ALPACA_SECRET_KEY = getattr(env_settings, 'ALPACA_SECRET_KEY', os.getenv("ALPACA_SECRET_KEY", ""))
ALPACA_BASE_URL = getattr(env_settings, 'ALPACA_BASE_URL', os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"))
ALPACA_PAPER = "paper" in ALPACA_BASE_URL.lower()
ALPACA_DATA_FEED = getattr(env_settings, 'ALPACA_DATA_FEED', os.getenv("ALPACA_DATA_FEED", "iex"))
FINNHUB_API_KEY = getattr(env_settings, 'FINNHUB_API_KEY', os.getenv("FINNHUB_API_KEY"))
FUNDAMENTAL_API_KEY = getattr(env_settings, 'FUNDAMENTAL_API_KEY', os.getenv("FUNDAMENTAL_API_KEY"))
NEWS_API_KEY = env_settings.NEWS_API_KEY
IEX_API_TOKEN = env_settings.IEX_API_TOKEN
BOT_MODE = env_settings.BOT_MODE
MODEL_PATH = env_settings.MODEL_PATH
RL_MODEL_PATH = env_settings.RL_MODEL_PATH
USE_RL_AGENT = env_settings.USE_RL_AGENT
HALT_FLAG_PATH = env_settings.HALT_FLAG_PATH
MAX_PORTFOLIO_POSITIONS = env_settings.MAX_PORTFOLIO_POSITIONS
LIMIT_ORDER_SLIPPAGE = env_settings.LIMIT_ORDER_SLIPPAGE
HEALTHCHECK_PORT = env_settings.HEALTHCHECK_PORT
RUN_HEALTHCHECK = env_settings.RUN_HEALTHCHECK
BUY_THRESHOLD = env_settings.BUY_THRESHOLD
WEBHOOK_SECRET = env_settings.WEBHOOK_SECRET
WEBHOOK_PORT = env_settings.WEBHOOK_PORT
SLIPPAGE_THRESHOLD = env_settings.SLIPPAGE_THRESHOLD
REBALANCE_INTERVAL_MIN = env_settings.REBALANCE_INTERVAL_MIN
SHADOW_MODE = env_settings.SHADOW_MODE
DISABLE_DAILY_RETRAIN = env_settings.DISABLE_DAILY_RETRAIN
# AI-AGENT-REF: unify trade log path under data/ with None guard
trade_log_file = env_settings.TRADE_LOG_FILE or "test_trades.csv"  # AI-AGENT-REF: fallback if None
TRADE_LOG_FILE = str((ROOT_DIR / trade_log_file).resolve())
EQUITY_EXPOSURE_CAP = float(os.getenv("EQUITY_EXPOSURE_CAP", "2.5"))
PORTFOLIO_EXPOSURE_CAP = float(os.getenv("PORTFOLIO_EXPOSURE_CAP", "2.5"))
SEED = int(os.getenv("SEED", str(env_settings.SEED)))
RATE_LIMIT_BUDGET = int(os.getenv("RATE_LIMIT_BUDGET", str(env_settings.RATE_LIMIT_BUDGET)))
VERBOSE = os.getenv("VERBOSE", "1").lower() not in ("0", "false")
VERBOSE_LOGGING = os.getenv("VERBOSE_LOGGING", "1").lower() not in ("0", "false")
# Minimum delay between scheduler iterations. Recommended 30–60s to
# prevent high CPU usage if errors occur in the trading loop.
SCHEDULER_SLEEP_SECONDS = float(os.getenv("SCHEDULER_SLEEP_SECONDS", "30"))
MIN_HEALTH_ROWS = int(os.getenv("MIN_HEALTH_ROWS", "30"))
MIN_HEALTH_ROWS_DAILY = int(os.getenv("MIN_HEALTH_ROWS_DAILY", "5"))
VOLUME_SPIKE_THRESHOLD = float(os.getenv("VOLUME_SPIKE_THRESHOLD", 1.5))
ML_CONFIDENCE_THRESHOLD = float(os.getenv("ML_CONFIDENCE_THRESHOLD", 0.55))
PYRAMID_LEVELS = {
    "high": float(os.getenv("PYRAMID_HIGH", 0.4)),
    "medium": float(os.getenv("PYRAMID_MEDIUM", 0.25)),
    "low": float(os.getenv("PYRAMID_LOW", 0.15)),
}
FORCE_TRADES: bool = False
"""If True, bypasses all pre-trade halts for testing."""

# AI-AGENT-REF: new adaptive execution and rebalance settings
PARTIAL_FILL_FRAGMENT_THRESHOLD = int(os.getenv("PARTIAL_FILL_FRAGMENT_THRESHOLD", "5"))
PARTIAL_FILL_LOOKBACK = int(os.getenv("PARTIAL_FILL_LOOKBACK", "10"))
PARTIAL_FILL_REDUCTION_RATIO = float(os.getenv("PARTIAL_FILL_REDUCTION_RATIO", "0.2"))
LIQUIDITY_SPREAD_THRESHOLD = float(os.getenv("LIQUIDITY_SPREAD_THRESHOLD", "0.05"))
LIQUIDITY_VOL_THRESHOLD = float(os.getenv("LIQUIDITY_VOL_THRESHOLD", "0.02"))
VOL_REGIME_MULTIPLIER = float(os.getenv("VOL_REGIME_MULTIPLIER", "2.0"))
REBALANCE_DRIFT_THRESHOLD = float(os.getenv("REBALANCE_DRIFT_THRESHOLD", "0.1"))
PORTFOLIO_DRIFT_THRESHOLD = float(os.getenv("PORTFOLIO_DRIFT_THRESHOLD", "0.10"))
TRADE_AUDIT_DIR = os.getenv("TRADE_AUDIT_DIR", "logs/trade_audits")


def set_runtime_config(volume_thr: float, ml_thr: float, pyramid_levels: dict) -> None:
    """Override key strategy parameters at runtime."""
    global VOLUME_SPIKE_THRESHOLD, ML_CONFIDENCE_THRESHOLD, PYRAMID_LEVELS
    VOLUME_SPIKE_THRESHOLD = volume_thr
    ML_CONFIDENCE_THRESHOLD = ml_thr
    PYRAMID_LEVELS = pyramid_levels


# centralize SGDRegressor hyperparameters
SGD_PARAMS = MappingProxyType(
    {
        "loss": "squared_error",
        "learning_rate": "constant",
        "eta0": 0.01,
        "penalty": "l2",
    }
)


def validate_alpaca_credentials() -> None:
    """Ensure required Alpaca credentials are present."""
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY or not ALPACA_BASE_URL:
        logger.error("Missing Alpaca credentials")
        raise RuntimeError(
            "Missing Alpaca credentials. Please set ALPACA_API_KEY, "
            "ALPACA_SECRET_KEY and ALPACA_BASE_URL in your environment"
        )


def validate_env_vars() -> None:
    """Comprehensive environment variable validation."""
    start_time = time.time()
    logger.info("Starting comprehensive environment variable validation")
    
    # Check if we already hold the lock to prevent deadlock
    if _is_lock_held_by_current_thread():
        logger.debug("Validation lock already held by current thread, proceeding without re-acquiring")
        _validate_env_vars_core()
        return
    
    # Attempt to acquire lock with timeout
    logger.debug("Attempting to acquire configuration validation lock (timeout: %d seconds)", _LOCK_TIMEOUT)
    lock_acquired = False
    
    try:
        lock_acquired = _CONFIG_VALIDATION_LOCK.acquire(timeout=_LOCK_TIMEOUT)
        if not lock_acquired:
            error_msg = f"Failed to acquire configuration validation lock within {_LOCK_TIMEOUT} seconds"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.debug("Configuration validation lock acquired successfully")
        _set_lock_held_by_current_thread(True)
        
        # Perform the actual validation
        _validate_env_vars_core()
        
        elapsed = time.time() - start_time
        logger.info("Comprehensive environment validation completed successfully in %.2f seconds", elapsed)
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("Comprehensive environment validation failed after %.2f seconds: %s", elapsed, e)
        raise
    finally:
        if lock_acquired:
            _set_lock_held_by_current_thread(False)
            _CONFIG_VALIDATION_LOCK.release()
            logger.debug("Configuration validation lock released")


def _validate_env_vars_core() -> None:
    """Core environment variable validation logic without lock handling."""
    logger.debug("Loading environment from .env file")
    try:
        load_dotenv()
        logger.debug("Environment loaded successfully")
    except Exception as e:
        logger.warning("Failed to load environment from .env file: %s", e)
    
    logger.debug("Calling core environment validation")
    # Call the core validation function directly to avoid lock re-acquisition
    _validate_environment_core()

    # Additional runtime validations
    logger.debug("Performing additional runtime validations")
    scheduler_sleep = os.getenv("SCHEDULER_SLEEP_SECONDS", "30")
    try:
        sleep_val = int(scheduler_sleep)
        if not (1 <= sleep_val <= 3600):
            raise ValueError("SCHEDULER_SLEEP_SECONDS must be between 1 and 3600")
        logger.debug("SCHEDULER_SLEEP_SECONDS validation passed: %d", sleep_val)
    except ValueError as e:
        error_msg = f"Invalid SCHEDULER_SLEEP_SECONDS: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.debug("All runtime validations completed successfully")


__all__ = [
    "get_env",
    "reload_env",
    "validate_environment",
    "validate_env_vars",
    "validate_alpaca_credentials",
    "mask_secret",
    "log_config",
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "ALPACA_BASE_URL",
    "ALPACA_PAPER",
    "ALPACA_DATA_FEED",
    "FINNHUB_API_KEY",
    "NEWS_API_KEY",
    "WEBHOOK_SECRET",
    "WEBHOOK_PORT",
    "MIN_HEALTH_ROWS",
    "MIN_HEALTH_ROWS_DAILY",
    "VOLUME_SPIKE_THRESHOLD",
    "ML_CONFIDENCE_THRESHOLD",
    "PYRAMID_LEVELS",
    "FORCE_TRADES",
    "VERBOSE",
    "PARTIAL_FILL_FRAGMENT_THRESHOLD",
    "PARTIAL_FILL_LOOKBACK",
    "LIQUIDITY_SPREAD_THRESHOLD",
    "LIQUIDITY_VOL_THRESHOLD",
    "VOL_REGIME_MULTIPLIER",
    "REBALANCE_DRIFT_THRESHOLD",
    "PORTFOLIO_DRIFT_THRESHOLD",
    "TRADE_AUDIT_DIR",
    "EQUITY_EXPOSURE_CAP",
    "PORTFOLIO_EXPOSURE_CAP",
    "SEED",
    "RATE_LIMIT_BUDGET",
    "set_runtime_config",
    "TradingConfig",
    "CONFIG",
]

# alias for backwards compatibility
Config = settings

# --- Trading bot parameters ----------------------------

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class TradingConfig:
    """Centralized configuration for trading parameters."""

    # Risk Management
    max_drawdown_threshold: float = 0.08
    position_size_min_usd: float = 100.0
    kelly_fraction_max: float = 0.25
    max_trades: int = 15

    # Signal Processing
    signal_confirmation_bars: int = 2
    trade_cooldown_min: float = 5.0
    delta_threshold: float = 0.02
    min_confidence: float = 0.6  # AI-AGENT-REF: add missing min_confidence attribute

    # Volatility & ATR
    volatility_lookback_days: int = 20
    atr_multiplier: float = 2.0

    # Exposure Management
    exposure_cap_aggressive: float = 0.8
    exposure_cap_conservative: float = 0.4

    @classmethod
    def from_env(cls) -> "TradingConfig":
        """Load configuration from environment variables."""
        import os

        return cls(
            max_drawdown_threshold=float(os.getenv("MAX_DRAWDOWN_THRESHOLD", "0.08")),
            position_size_min_usd=float(os.getenv("POSITION_SIZE_MIN_USD", "100.0")),
            kelly_fraction_max=float(os.getenv("KELLY_FRACTION_MAX", "0.25")),
            max_trades=int(os.getenv("MAX_TRADES", "15")),
            signal_confirmation_bars=int(os.getenv("SIGNAL_CONFIRMATION_BARS", "2")),
            trade_cooldown_min=float(os.getenv("TRADE_COOLDOWN_MIN", "5.0")),
            delta_threshold=float(os.getenv("DELTA_THRESHOLD", "0.02")),
            min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.6")),  # AI-AGENT-REF: add min_confidence env var
            volatility_lookback_days=int(os.getenv("VOLATILITY_LOOKBACK_DAYS", "20")),
            atr_multiplier=float(os.getenv("ATR_MULTIPLIER", "2.0")),
            exposure_cap_aggressive=float(os.getenv("EXPOSURE_CAP_AGGRESSIVE", "0.8")),
            exposure_cap_conservative=float(os.getenv("EXPOSURE_CAP_CONSERVATIVE", "0.4")),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for optimization algorithms."""
        return {
            "max_drawdown_threshold": self.max_drawdown_threshold,
            "kelly_fraction_max": self.kelly_fraction_max,
            "signal_confirmation_bars": self.signal_confirmation_bars,
            "atr_multiplier": self.atr_multiplier,
            "delta_threshold": self.delta_threshold,
            "min_confidence": self.min_confidence,  # AI-AGENT-REF: add min_confidence to dict conversion
            "exposure_cap_aggressive": self.exposure_cap_aggressive,
            "exposure_cap_conservative": self.exposure_cap_conservative,
        }

    @classmethod
    def from_optimization(cls, params: Dict[str, Any]) -> "TradingConfig":
        """Create configuration from optimization parameters."""
        config = cls.from_env()
        for key, value in params.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


# default trading configuration used across modules
CONFIG = TradingConfig.from_env()
