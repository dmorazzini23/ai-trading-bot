from .settings import Settings, get_settings, broker_keys  # noqa: F401
from .alpaca import get_alpaca_config, AlpacaConfig  # noqa: F401
from .management import TradingConfig  # AI-AGENT-REF: expose TradingConfig
import os


# AI-AGENT-REF: legacy environment accessor
def get_env(name: str, default: str | None = None, *, required: bool = False) -> str | None:
    val = os.environ.get(name, default)
    if required and val is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return val

__all__ = [
    "Settings",
    "get_settings",
    "broker_keys",
    "get_alpaca_config",
    "AlpacaConfig",
    "TradingConfig",
    "get_env",
]

