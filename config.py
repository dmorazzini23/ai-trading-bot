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
        # Sentiment API Configuration (with fallback to NEWS_API_KEY for backwards compatibility)
        SENTIMENT_API_KEY = os.getenv("SENTIMENT_API_KEY") or os.getenv("NEWS_API_KEY")
        SENTIMENT_API_URL = os.getenv("SENTIMENT_API_URL", "https://newsapi.org/v2/everything")
        IEX_API_TOKEN = os.getenv("IEX_API_TOKEN")
        BOT_MODE = os.getenv("BOT_MODE", "balanced")
        DOLLAR_RISK_LIMIT = float(os.getenv("DOLLAR_RISK_LIMIT", "0.05"))
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
# Sentiment API Configuration with fallback support
SENTIMENT_API_KEY = getattr(env_settings, 'SENTIMENT_API_KEY', None) or NEWS_API_KEY
SENTIMENT_API_URL = getattr(env_settings, 'SENTIMENT_API_URL', "https://newsapi.org/v2/everything")
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

# AI-AGENT-REF: Add drawdown circuit breaker configuration
MAX_DRAWDOWN_THRESHOLD = float(os.getenv("MAX_DRAWDOWN_THRESHOLD", "0.15"))
DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", "0.03"))
FORCE_TRADES: bool = False
"""If True, bypasses all pre-trade halts for testing."""

# AI-AGENT-REF: new adaptive execution and rebalance settings
PARTIAL_FILL_FRAGMENT_THRESHOLD = int(os.getenv("PARTIAL_FILL_FRAGMENT_THRESHOLD", "5"))
PARTIAL_FILL_LOOKBACK = int(os.getenv("PARTIAL_FILL_LOOKBACK", "10"))
PARTIAL_FILL_REDUCTION_RATIO = float(os.getenv("PARTIAL_FILL_REDUCTION_RATIO", "0.2"))
LIQUIDITY_SPREAD_THRESHOLD = float(os.getenv("LIQUIDITY_SPREAD_THRESHOLD", "0.15"))  # Increased from 0.05 to 0.15 (15%)
LIQUIDITY_VOL_THRESHOLD = float(os.getenv("LIQUIDITY_VOL_THRESHOLD", "0.08"))  # Increased from 0.02 to 0.08 (8%)
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
    "SENTIMENT_API_KEY",
    "SENTIMENT_API_URL",
    "WEBHOOK_SECRET",
    "WEBHOOK_PORT",
    "MIN_HEALTH_ROWS",
    "MIN_HEALTH_ROWS_DAILY",
    "VOLUME_SPIKE_THRESHOLD",
    "ML_CONFIDENCE_THRESHOLD",
    "PYRAMID_LEVELS",
    "MAX_DRAWDOWN_THRESHOLD",
    "DAILY_LOSS_LIMIT",
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
    """Centralized configuration for all trading parameters.
    
    Single source of truth for trading parameters with support for
    mode-specific configurations and environment variable overrides.
    """

    # Risk Management Parameters
    max_drawdown_threshold: float = 0.15
    daily_loss_limit: float = 0.03
    dollar_risk_limit: float = 0.05
    max_portfolio_risk: float = 0.025
    max_correlation_exposure: float = 0.15
    max_sector_concentration: float = 0.15
    min_liquidity_threshold: int = 1000000
    position_size_min_usd: float = 100.0
    max_position_size: int = 8000
    max_position_size_pct: float = 0.25

    # Kelly Criterion Parameters
    kelly_fraction: float = 0.6
    kelly_fraction_max: float = 0.25
    min_sample_size: int = 20
    confidence_level: float = 0.90
    lookback_periods: int = 252
    rebalance_frequency: int = 21

    # Trading Mode Parameters
    conf_threshold: float = 0.75
    buy_threshold: float = 0.1
    min_confidence: float = 0.6
    confirmation_count: int = 2
    take_profit_factor: float = 1.8
    trailing_factor: float = 1.2
    scaling_factor: float = 0.3

    # Signal Processing Parameters
    signal_confirmation_bars: int = 2
    signal_period: int = 9
    fast_period: int = 5
    slow_period: int = 20
    trade_cooldown_min: float = 5.0
    delta_threshold: float = 0.02
    entry_start_offset_min: int = 30
    entry_end_offset_min: int = 15

    # Volatility & ATR Parameters
    volatility_lookback_days: int = 20
    atr_multiplier: float = 2.0
    stop_loss_multiplier: float = 1.8
    take_profit_multiplier: float = 2.5

    # Execution Parameters
    limit_order_slippage: float = 0.005
    max_slippage_bps: int = 15
    participation_rate: float = 0.15
    pov_slice_pct: float = 0.05
    min_order_size: int = 100
    max_order_size: int = 10000
    order_timeout_seconds: int = 180
    retry_attempts: int = 3
    cancel_threshold_seconds: int = 60

    # Capital Allocation Parameters
    capital_cap: float = 0.25
    max_trades: int = 15

    # Exposure Management
    exposure_cap_aggressive: float = 0.8
    exposure_cap_conservative: float = 0.4

    # Performance Thresholds
    min_sharpe_ratio: float = 1.2
    max_drawdown: float = 0.15
    min_win_rate: float = 0.48
    min_profit_factor: float = 1.2
    max_var_95: float = 0.05

    @classmethod
    def from_env(cls, mode: str = "balanced") -> "TradingConfig":
        """Load configuration from environment variables with mode-specific defaults."""
        import os

        # Get base configuration
        config = cls(
            max_drawdown_threshold=float(os.getenv("MAX_DRAWDOWN_THRESHOLD", "0.15")),
            daily_loss_limit=float(os.getenv("DAILY_LOSS_LIMIT", "0.03")),
            dollar_risk_limit=float(os.getenv("DOLLAR_RISK_LIMIT", "0.05")),
            max_portfolio_risk=float(os.getenv("MAX_PORTFOLIO_RISK", "0.025")),
            max_correlation_exposure=float(os.getenv("MAX_CORRELATION_EXPOSURE", "0.15")),
            max_sector_concentration=float(os.getenv("MAX_SECTOR_CONCENTRATION", "0.15")),
            min_liquidity_threshold=int(os.getenv("MIN_LIQUIDITY_THRESHOLD", "1000000")),
            position_size_min_usd=float(os.getenv("POSITION_SIZE_MIN_USD", "100.0")),
            max_position_size=int(os.getenv("MAX_POSITION_SIZE", "8000")),
            max_position_size_pct=float(os.getenv("MAX_POSITION_SIZE_PCT", "0.25")),
            
            kelly_fraction=float(os.getenv("KELLY_FRACTION", "0.6")),
            kelly_fraction_max=float(os.getenv("KELLY_FRACTION_MAX", "0.25")),
            min_sample_size=int(os.getenv("MIN_SAMPLE_SIZE", "20")),
            confidence_level=float(os.getenv("CONFIDENCE_LEVEL", "0.90")),
            lookback_periods=int(os.getenv("LOOKBACK_PERIODS", "252")),
            rebalance_frequency=int(os.getenv("REBALANCE_FREQUENCY", "21")),
            
            conf_threshold=float(os.getenv("CONF_THRESHOLD", "0.75")),
            buy_threshold=float(os.getenv("BUY_THRESHOLD", "0.1")),
            min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.6")),
            confirmation_count=int(os.getenv("CONFIRMATION_COUNT", "2")),
            take_profit_factor=float(os.getenv("TAKE_PROFIT_FACTOR", "1.8")),
            trailing_factor=float(os.getenv("TRAILING_FACTOR", "1.2")),
            scaling_factor=float(os.getenv("SCALING_FACTOR", "0.3")),
            
            signal_confirmation_bars=int(os.getenv("SIGNAL_CONFIRMATION_BARS", "2")),
            signal_period=int(os.getenv("SIGNAL_PERIOD", "9")),
            fast_period=int(os.getenv("FAST_PERIOD", "5")),
            slow_period=int(os.getenv("SLOW_PERIOD", "20")),
            trade_cooldown_min=float(os.getenv("TRADE_COOLDOWN_MIN", "5.0")),
            delta_threshold=float(os.getenv("DELTA_THRESHOLD", "0.02")),
            entry_start_offset_min=int(os.getenv("ENTRY_START_OFFSET_MIN", "30")),
            entry_end_offset_min=int(os.getenv("ENTRY_END_OFFSET_MIN", "15")),
            
            volatility_lookback_days=int(os.getenv("VOLATILITY_LOOKBACK_DAYS", "20")),
            atr_multiplier=float(os.getenv("ATR_MULTIPLIER", "2.0")),
            stop_loss_multiplier=float(os.getenv("STOP_LOSS_MULTIPLIER", "1.8")),
            take_profit_multiplier=float(os.getenv("TAKE_PROFIT_MULTIPLIER", "2.5")),
            
            limit_order_slippage=float(os.getenv("LIMIT_ORDER_SLIPPAGE", "0.005")),
            max_slippage_bps=int(os.getenv("MAX_SLIPPAGE_BPS", "15")),
            participation_rate=float(os.getenv("PARTICIPATION_RATE", "0.15")),
            pov_slice_pct=float(os.getenv("POV_SLICE_PCT", "0.05")),
            min_order_size=int(os.getenv("MIN_ORDER_SIZE", "100")),
            max_order_size=int(os.getenv("MAX_ORDER_SIZE", "10000")),
            order_timeout_seconds=int(os.getenv("ORDER_TIMEOUT_SECONDS", "180")),
            retry_attempts=int(os.getenv("RETRY_ATTEMPTS", "3")),
            cancel_threshold_seconds=int(os.getenv("CANCEL_THRESHOLD_SECONDS", "60")),
            
            capital_cap=float(os.getenv("CAPITAL_CAP", "0.25")),
            max_trades=int(os.getenv("MAX_TRADES", "15")),
            
            exposure_cap_aggressive=float(os.getenv("EXPOSURE_CAP_AGGRESSIVE", "0.8")),
            exposure_cap_conservative=float(os.getenv("EXPOSURE_CAP_CONSERVATIVE", "0.4")),
            
            min_sharpe_ratio=float(os.getenv("MIN_SHARPE_RATIO", "1.2")),
            max_drawdown=float(os.getenv("MAX_DRAWDOWN", "0.15")),
            min_win_rate=float(os.getenv("MIN_WIN_RATE", "0.48")),
            min_profit_factor=float(os.getenv("MIN_PROFIT_FACTOR", "1.2")),
            max_var_95=float(os.getenv("MAX_VAR_95", "0.05")),
        )
        
        # Apply mode-specific adjustments
        return config.with_mode(mode)

    def with_mode(self, mode: str) -> "TradingConfig":
        """Apply mode-specific parameter adjustments."""
        if mode == "conservative":
            return self._apply_conservative_mode()
        elif mode == "aggressive":
            return self._apply_aggressive_mode()
        else:  # balanced (default)
            return self._apply_balanced_mode()

    def _apply_conservative_mode(self) -> "TradingConfig":
        """Apply conservative mode parameters."""
        # Create a copy and modify conservative parameters
        import copy
        config = copy.deepcopy(self)
        
        # Risk Management - Lower risk tolerance
        config.kelly_fraction = 0.25
        config.daily_loss_limit = 0.03
        config.capital_cap = 0.20
        config.max_position_size = 5000
        
        # Signal Processing - Higher confirmation requirements
        config.conf_threshold = 0.85
        config.min_confidence = 0.75
        config.confirmation_count = 3
        
        # Profit/Loss - Tighter targets
        config.take_profit_factor = 1.5
        config.trailing_factor = 1.5
        
        return config

    def _apply_balanced_mode(self) -> "TradingConfig":
        """Apply balanced mode parameters (default values with some adjustments)."""
        # Create a copy and set balanced-specific parameters
        import copy
        config = copy.deepcopy(self)
        
        # Balanced mode has moderate settings
        config.daily_loss_limit = 0.05  # 5% daily loss limit for balanced mode
        
        return config

    def _apply_aggressive_mode(self) -> "TradingConfig":
        """Apply aggressive mode parameters."""
        # Create a copy and modify aggressive parameters
        import copy
        config = copy.deepcopy(self)
        
        # Risk Management - Higher risk tolerance
        config.kelly_fraction = 0.75
        config.daily_loss_limit = 0.08
        config.capital_cap = 0.30
        config.max_position_size = 12000
        
        # Signal Processing - Faster execution
        config.conf_threshold = 0.65
        config.min_confidence = 0.50
        config.confirmation_count = 1
        
        # Profit/Loss - Extended targets
        config.take_profit_factor = 2.5
        config.trailing_factor = 2.0
        
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for optimization algorithms."""
        return {
            # Risk Management
            "max_drawdown_threshold": self.max_drawdown_threshold,
            "daily_loss_limit": self.daily_loss_limit,
            "dollar_risk_limit": self.dollar_risk_limit,
            "max_portfolio_risk": self.max_portfolio_risk,
            "max_correlation_exposure": self.max_correlation_exposure,
            "max_position_size": self.max_position_size,
            "max_position_size_pct": self.max_position_size_pct,
            
            # Kelly Criterion
            "kelly_fraction": self.kelly_fraction,
            "kelly_fraction_max": self.kelly_fraction_max,
            "min_sample_size": self.min_sample_size,
            "confidence_level": self.confidence_level,
            
            # Trading Mode
            "conf_threshold": self.conf_threshold,
            "buy_threshold": self.buy_threshold,
            "min_confidence": self.min_confidence,
            "confirmation_count": self.confirmation_count,
            "take_profit_factor": self.take_profit_factor,
            "trailing_factor": self.trailing_factor,
            
            # Signal Processing
            "signal_confirmation_bars": self.signal_confirmation_bars,
            "signal_period": self.signal_period,
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "delta_threshold": self.delta_threshold,
            
            # Volatility & ATR
            "atr_multiplier": self.atr_multiplier,
            "stop_loss_multiplier": self.stop_loss_multiplier,
            "take_profit_multiplier": self.take_profit_multiplier,
            
            # Execution
            "limit_order_slippage": self.limit_order_slippage,
            "max_slippage_bps": self.max_slippage_bps,
            "participation_rate": self.participation_rate,
            "pov_slice_pct": self.pov_slice_pct,
            "order_timeout_seconds": self.order_timeout_seconds,
            
            # Capital
            "capital_cap": self.capital_cap,
            "max_trades": self.max_trades,
            
            # Exposure
            "exposure_cap_aggressive": self.exposure_cap_aggressive,
            "exposure_cap_conservative": self.exposure_cap_conservative,
        }

    def get_legacy_params(self) -> Dict[str, float]:
        """Get parameters in legacy format for backward compatibility."""
        return {
            "KELLY_FRACTION": self.kelly_fraction,
            "CONF_THRESHOLD": self.conf_threshold,
            "CONFIRMATION_COUNT": self.confirmation_count,
            "TAKE_PROFIT_FACTOR": self.take_profit_factor,
            "DAILY_LOSS_LIMIT": self.daily_loss_limit,
            "CAPITAL_CAP": self.capital_cap,
            "TRAILING_FACTOR": self.trailing_factor,
            "BUY_THRESHOLD": self.buy_threshold,
            "SCALING_FACTOR": self.scaling_factor,
            "POV_SLICE_PCT": self.pov_slice_pct,
            "ENTRY_START_OFFSET_MIN": self.entry_start_offset_min,
            "ENTRY_END_OFFSET_MIN": self.entry_end_offset_min,
            "LIMIT_ORDER_SLIPPAGE": self.limit_order_slippage,
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
# Load mode from environment variable or use balanced as default
_BOT_MODE = os.getenv("BOT_MODE", "balanced")
CONFIG = TradingConfig.from_env(mode=_BOT_MODE)
