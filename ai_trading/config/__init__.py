from .settings import Settings, get_settings, broker_keys  # noqa: F401
from .alpaca import get_alpaca_config, AlpacaConfig  # noqa: F401
from .management import TradingConfig  # AI-AGENT-REF: expose TradingConfig
import logging
import os
from typing import Iterable


# AI-AGENT-REF: legacy environment accessor
def get_env(name: str, default: str | None = None, *, required: bool = False) -> str | None:
    val = os.environ.get(name, default)
    if required and val is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def _require_env_vars(*names: str) -> None:
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        logging.getLogger(__name__).critical(
            "Missing required environment variables: %s", ", ".join(missing)
        )
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}"
        )


def reload_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except Exception:
        pass

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

