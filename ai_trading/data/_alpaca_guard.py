from __future__ import annotations

"""Helpers to determine whether Alpaca SDK access is required."""

from ai_trading.config import get_execution_settings
from ai_trading.config.settings import get_settings


def _execution_enabled() -> bool:
    """Return ``True`` when trade execution requires Alpaca SDK access."""

    try:
        mode = str(get_execution_settings().mode).lower()
    except Exception:
        return False
    return mode in {"paper", "live"}


def should_import_alpaca_sdk() -> bool:
    """Return ``True`` when Alpaca SDK resources are required."""

    try:
        provider = getattr(get_settings(), "data_provider", "")
    except Exception:
        provider = ""
    provider_normalized = str(provider or "").strip().lower()
    return _execution_enabled() and provider_normalized == "alpaca"


__all__ = ["should_import_alpaca_sdk"]
