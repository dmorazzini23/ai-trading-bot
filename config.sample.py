import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(ENV_PATH)

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_PAPER = "paper" in ALPACA_BASE_URL.lower()
ALPACA_DATA_FEED = os.environ.get("ALPACA_DATA_FEED", "iex")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
FUNDAMENTAL_API_KEY = os.environ.get("FUNDAMENTAL_API_KEY")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
IEX_API_TOKEN = os.environ.get("IEX_API_TOKEN")
BOT_MODE = os.environ.get("BOT_MODE", "balanced")
MODEL_PATH = os.environ.get("MODEL_PATH", "trained_model.pkl")
HALT_FLAG_PATH = os.environ.get("HALT_FLAG_PATH", "halt.flag")
MAX_PORTFOLIO_POSITIONS = int(os.environ.get("MAX_PORTFOLIO_POSITIONS", "20"))
LIMIT_ORDER_SLIPPAGE = float(os.environ.get("LIMIT_ORDER_SLIPPAGE", "0.005"))
HEALTHCHECK_PORT = int(os.environ.get("HEALTHCHECK_PORT", "8081"))
RUN_HEALTHCHECK = os.environ.get("RUN_HEALTHCHECK", "0")
BUY_THRESHOLD = float(os.environ.get("BUY_THRESHOLD", "0.5"))
DISABLE_DAILY_RETRAIN = os.environ.get("DISABLE_DAILY_RETRAIN", "0")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
WEBHOOK_PORT = int(os.environ.get("WEBHOOK_PORT", "9000"))
FORCE_TRADES = False  # If True, bypasses all pre-trade halts for testing.


def validate_alpaca_credentials() -> None:
    """Ensure required Alpaca credentials are present."""
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY or not ALPACA_BASE_URL:
        raise RuntimeError(
            "Missing Alpaca credentials. Please set ALPACA_API_KEY, "
            "ALPACA_SECRET_KEY and ALPACA_BASE_URL in your environment"
        )
