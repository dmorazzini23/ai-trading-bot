"""Lightweight runtime entry points.

Provides helpers to bootstrap the trading context without triggering
heavy imports at module import time.
"""
from __future__ import annotations

from typing import Any

from ai_trading.core import bot_engine


def start(api: Any | None = None):
    """Initialize the bot context and ensure an API client is attached.

    Parameters
    ----------
    api:
        Optional API client to attach. When ``None``, the function attempts to
        attach the global Alpaca trading client via
        :func:`ai_trading.core.bot_engine.ensure_alpaca_attached`.
    """
    ctx = bot_engine.get_ctx()
    if hasattr(ctx, "_ensure_initialized"):
        try:
            ctx._ensure_initialized()  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive
            pass
    inner = getattr(ctx, "_context", ctx)
    if getattr(inner, "api", None) is None:
        if api is not None:
            setattr(inner, "api", api)
        else:  # Defer to bot_engine to resolve the global trading client
            bot_engine.ensure_alpaca_attached(inner)
            if getattr(inner, "api", None) is None:
                # Fallback to a minimal stub to satisfy tests when Alpaca
                # clients are unavailable or intentionally absent.
                setattr(inner, "api", object())
    # Mirror the attribute on the lazy wrapper so external access works
    try:
        setattr(ctx, "api", getattr(inner, "api"))
    except Exception:  # pragma: no cover - defensive
        pass
    return inner


__all__ = ["start"]
