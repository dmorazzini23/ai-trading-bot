from pathlib import Path
import os
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / '.env'
load_dotenv(ENV_PATH)

required = ['APCA_API_KEY_ID','APCA_API_SECRET_KEY']  # FUNDAMENTAL_API_KEY optional
missing = [v for v in required if v not in os.environ]
if missing:
    raise RuntimeError(f"Missing required environment variables: {missing}")

FUNDAMENTAL_API_KEY = os.getenv('FUNDAMENTAL_API_KEY', None)

ALPACA_API_KEY = os.environ.get('ALPACA_API_KEY') or os.environ.get('APCA_API_KEY_ID')
ALPACA_SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY') or os.environ.get('APCA_API_SECRET_KEY')
ALPACA_BASE_URL = os.environ.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
ALPACA_PAPER = 'paper' in ALPACA_BASE_URL.lower()
FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY')
NEWS_API_KEY = os.environ.get('NEWS_API_KEY')
IEX_API_TOKEN = os.environ.get('IEX_API_TOKEN')
SENTRY_DSN = os.environ.get('SENTRY_DSN')
BOT_MODE = os.environ.get('BOT_MODE', 'balanced')
MODEL_PATH = os.environ.get('MODEL_PATH', 'trained_model.pkl')
HALT_FLAG_PATH = os.environ.get('HALT_FLAG_PATH', 'halt.flag')
MAX_PORTFOLIO_POSITIONS = int(os.environ.get('MAX_PORTFOLIO_POSITIONS', '20'))
LIMIT_ORDER_SLIPPAGE = float(os.environ.get('LIMIT_ORDER_SLIPPAGE', '0.005'))
HEALTHCHECK_PORT = int(os.environ.get('HEALTHCHECK_PORT', '8081'))
RUN_HEALTHCHECK = os.environ.get('RUN_HEALTHCHECK', '0')
BUY_THRESHOLD = float(os.environ.get('BUY_THRESHOLD', '0.5'))
WEBHOOK_SECRET = os.environ.get('WEBHOOK_SECRET', '')
WEBHOOK_PORT = int(os.environ.get('WEBHOOK_PORT', '9000'))


def validate_alpaca_credentials() -> None:
    """Ensure required Alpaca credentials are present."""
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY or not ALPACA_BASE_URL:
        raise RuntimeError(
            "Missing Alpaca credentials. Please set ALPACA_API_KEY, "
            "ALPACA_SECRET_KEY and ALPACA_BASE_URL in your environment"
        )

