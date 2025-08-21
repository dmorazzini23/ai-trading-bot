import os
from functools import lru_cache

from ai_trading.settings import (
    Settings,
    _secret_to_str,
    get_settings as _base_get_settings,
)  # AI-AGENT-REF: canonical settings

TICKERS_FILE = os.getenv("AI_TRADER_TICKERS_FILE", "tickers.csv")
MODEL_PATH = os.getenv("AI_TRADER_MODEL_PATH")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance."""  # AI-AGENT-REF: simple wrapper
    return _base_get_settings()


def broker_keys(s: Settings | None = None) -> dict[str, str]:
    """Return broker credential mapping."""  # AI-AGENT-REF: key extractor
    s = s or get_settings()
    keys = {
        "ALPACA_API_KEY": getattr(s, "alpaca_api_key", ""),
        "ALPACA_SECRET_KEY": _secret_to_str(getattr(s, "alpaca_secret_key", None)) or "",
    }
    if getattr(s, "finnhub_api_key", None):
        keys["finnhub"] = s.finnhub_api_key
    return keys


__all__ = ["Settings", "get_settings", "broker_keys"]

