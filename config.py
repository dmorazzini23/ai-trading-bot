from pathlib import Path
import os
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"
# Load environment variables once at import. Individual scripts
# should call ``reload_env()`` to refresh values if needed.
load_dotenv(ENV_PATH)


def get_env(key: str, default: str | None = None, *, reload: bool = False):
    """Return environment variable ``key``. Reload .env first if requested."""
    if reload:
        reload_env()
    return os.environ.get(key, default)


def reload_env() -> None:
    """Reload environment variables from the .env file if it exists."""
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=True)

from types import MappingProxyType

required_env_vars = ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY")
missing = [v for v in required_env_vars if v not in os.environ]
if missing:
    raise RuntimeError(f"Missing required environment variables: {missing}")

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
        raise RuntimeError(
            "Missing Alpaca credentials. Please set ALPACA_API_KEY, "
            "ALPACA_SECRET_KEY and ALPACA_BASE_URL in your environment"
        )
