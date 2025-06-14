from pathlib import Path
import os
import sys
from dotenv import load_dotenv
import logging

ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"
# Load environment variables once at import. Individual scripts
# should call ``reload_env()`` to refresh values if needed.
load_dotenv(ENV_PATH)

logger = logging.getLogger(__name__)


def get_env(
    key: str, default: str | None = None, *, reload: bool = False, required: bool = False
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


def _require_env_vars(*keys: str) -> None:
    missing = [v for v in keys if not os.environ.get(v)]
    if missing:
        logger.critical("Missing required environment variables: %s", missing)
        sys.exit(1)


_require_env_vars("APCA_API_KEY_ID", "APCA_API_SECRET_KEY")

ALPACA_API_KEY = get_env("ALPACA_API_KEY") or get_env("APCA_API_KEY_ID")
ALPACA_SECRET_KEY = get_env("ALPACA_SECRET_KEY") or get_env("APCA_API_SECRET_KEY")
ALPACA_BASE_URL = get_env("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_PAPER = "paper" in ALPACA_BASE_URL.lower()
FINNHUB_API_KEY = get_env("FINNHUB_API_KEY")
FUNDAMENTAL_API_KEY = get_env("FUNDAMENTAL_API_KEY")
NEWS_API_KEY = get_env("NEWS_API_KEY")
IEX_API_TOKEN = get_env("IEX_API_TOKEN")
SENTRY_DSN = get_env("SENTRY_DSN")
BOT_MODE = get_env("BOT_MODE", "balanced")
MODEL_PATH = get_env("MODEL_PATH", "trained_model.pkl")
HALT_FLAG_PATH = get_env("HALT_FLAG_PATH", "halt.flag")
MAX_PORTFOLIO_POSITIONS = int(get_env("MAX_PORTFOLIO_POSITIONS", "20"))
LIMIT_ORDER_SLIPPAGE = float(get_env("LIMIT_ORDER_SLIPPAGE", "0.005"))
HEALTHCHECK_PORT = int(get_env("HEALTHCHECK_PORT", "8081"))
RUN_HEALTHCHECK = get_env("RUN_HEALTHCHECK", "0")
BUY_THRESHOLD = float(get_env("BUY_THRESHOLD", "0.5"))
WEBHOOK_SECRET = get_env("WEBHOOK_SECRET", "")
WEBHOOK_PORT = int(get_env("WEBHOOK_PORT", "9000"))

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
