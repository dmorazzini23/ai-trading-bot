"""Extended main utilities."""
from __future__ import annotations

import logging
import os

from ai_trading import config

logger = logging.getLogger(__name__)


def validate_environment() -> None:
    """Ensure required environment variables are present and dependencies are available."""
    from ai_trading.config.management import get_env, _resolve_alpaca_env

    missing: list[str] = []
    key, secret, base_url = _resolve_alpaca_env()
    if not key:
        missing.append("ALPACA_API_KEY")
    if not secret:
        missing.append("ALPACA_SECRET_KEY")
    if not base_url:
        missing.append("ALPACA_API_URL")
    for var in ("WEBHOOK_SECRET", "CAPITAL_CAP", "DOLLAR_RISK_LIMIT"):
        if not get_env(var):
            missing.append(var)
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )

    _ = config.get_settings()
    data_dir = "data"
    if not os.path.exists(data_dir):
        logger.info("Creating data directory: %s", data_dir)
        os.makedirs(data_dir, exist_ok=True)
