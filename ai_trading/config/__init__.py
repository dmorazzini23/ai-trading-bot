from .settings import Settings, get_settings, broker_keys  # noqa: F401
from .alpaca import get_alpaca_config, AlpacaConfig  # noqa: F401
from .management import TradingConfig  # AI-AGENT-REF: expose TradingConfig
import os
from typing import Any
import logging

logger = logging.getLogger(__name__)


def reload_env() -> None:
    """Reload .env if python-dotenv is present; ignore failures."""
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except Exception:
        pass


def get_env(name: str, default: Any = None, *, reload: bool = False, required: bool = False) -> Any:
    """Return env var; if reload=True, call reload_env() first."""
    if reload:
        reload_env()
    val = os.getenv(name, default)
    if required and val is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def _require_env_vars(*names: str) -> None:
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        msg = f"Missing required environment variables: {', '.join(missing)}"
        logger.critical(msg)
        raise RuntimeError(msg)

__all__ = [
    "Settings",
    "get_settings",
    "broker_keys",
    "get_alpaca_config",
    "AlpacaConfig",
    "TradingConfig",
    "get_env",
    "_require_env_vars",
    "reload_env",
]

