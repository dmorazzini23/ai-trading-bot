import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from validate_env import settings as env_settings

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"
# Load environment variables once at import. Individual scripts
# should call ``reload_env()`` to refresh values if needed.
load_dotenv(ENV_PATH)

# Relax environment variable validation when running under pytest so the
# configuration module can be imported without all production variables set.
TESTING = os.getenv("PYTEST_CURRENT_TEST") is not None or os.getenv("TESTING")

required_env_vars = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_BASE_URL", "WEBHOOK_SECRET"]
if not TESTING:
    required_env_vars.append("FLASK_PORT")

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise RuntimeError(
        f"Missing required environment variables: {', '.join(missing_vars)}"
    )

REQUIRED_ENV_VARS = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_BASE_URL", "WEBHOOK_SECRET"]

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


def log_config(keys: list[str] | None = None) -> None:
    """Log key configuration values with secrets masked."""
    if keys is None:
        keys = REQUIRED_ENV_VARS
    cfg = {}
    for k in keys:
        val = os.getenv(k, "")
        if "KEY" in k or "SECRET" in k:
            val = mask_secret(val)
        cfg[k] = val
    logger.info("Configuration: %s", cfg)


def _require_env_vars(*keys: str) -> None:
    missing = [v for v in keys if not os.environ.get(v)]
    if missing:
        logger.critical("Missing required environment variables: %s", missing)
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )


def validate_environment() -> None:
    """Validate that mandatory environment variables are present."""
    missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )

ALPACA_API_KEY = env_settings.ALPACA_API_KEY
ALPACA_SECRET_KEY = env_settings.ALPACA_SECRET_KEY
ALPACA_BASE_URL = env_settings.ALPACA_BASE_URL
ALPACA_PAPER = "paper" in ALPACA_BASE_URL.lower()
ALPACA_DATA_FEED = env_settings.ALPACA_DATA_FEED
FINNHUB_API_KEY = env_settings.FINNHUB_API_KEY
FUNDAMENTAL_API_KEY = env_settings.FUNDAMENTAL_API_KEY
NEWS_API_KEY = env_settings.NEWS_API_KEY
IEX_API_TOKEN = env_settings.IEX_API_TOKEN
BOT_MODE = env_settings.BOT_MODE
MODEL_PATH = env_settings.MODEL_PATH
HALT_FLAG_PATH = env_settings.HALT_FLAG_PATH
MAX_PORTFOLIO_POSITIONS = env_settings.MAX_PORTFOLIO_POSITIONS
LIMIT_ORDER_SLIPPAGE = env_settings.LIMIT_ORDER_SLIPPAGE
HEALTHCHECK_PORT = env_settings.HEALTHCHECK_PORT
RUN_HEALTHCHECK = env_settings.RUN_HEALTHCHECK
BUY_THRESHOLD = env_settings.BUY_THRESHOLD
WEBHOOK_SECRET = env_settings.WEBHOOK_SECRET
WEBHOOK_PORT = env_settings.WEBHOOK_PORT
SLACK_WEBHOOK = env_settings.SLACK_WEBHOOK
SLIPPAGE_THRESHOLD = env_settings.SLIPPAGE_THRESHOLD
REBALANCE_INTERVAL_MIN = env_settings.REBALANCE_INTERVAL_MIN
SHADOW_MODE = env_settings.SHADOW_MODE
DISABLE_DAILY_RETRAIN = env_settings.DISABLE_DAILY_RETRAIN
TRADE_LOG_FILE = env_settings.TRADE_LOG_FILE
SCHEDULER_SLEEP_SECONDS = float(os.getenv("SCHEDULER_SLEEP_SECONDS", "30"))
MIN_HEALTH_ROWS = int(os.getenv("MIN_HEALTH_ROWS", "30"))
MIN_HEALTH_ROWS_DAILY = int(os.getenv("MIN_HEALTH_ROWS_DAILY", "5"))
FORCE_TRADES: bool = True
"""If True, bypasses all pre-trade halts for testing."""

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
    """Ensure critical environment variables are present.

    This function is intended to be called at startup. It reloads ``.env``
    and raises ``RuntimeError`` if any required variables are missing.
    """
    load_dotenv()
    _require_env_vars(*REQUIRED_ENV_VARS)
    log_config(REQUIRED_ENV_VARS)


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
    "FORCE_TRADES",
]
