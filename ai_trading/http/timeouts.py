from __future__ import annotations
from typing import Final

from ai_trading.config import management as config

_DEFAULT_TIMEOUT: Final[float] = 5.0


def _resolve_timeout() -> float:
    """Resolve default HTTP timeout from config or fallback."""
    try:
        t = float(
            config.get_env("AI_TRADING_HTTP_TIMEOUT", str(_DEFAULT_TIMEOUT), cast=float)
        )
        if t > 0:
            return t
    except Exception:
        pass
    return _DEFAULT_TIMEOUT


SESSION_TIMEOUT: Final[float] = _resolve_timeout()


def get_session_timeout() -> float:
    """Return the default timeout for HTTP sessions."""
    return SESSION_TIMEOUT


__all__ = ["SESSION_TIMEOUT", "get_session_timeout"]
