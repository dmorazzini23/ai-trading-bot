from __future__ import annotations

"""Environment variable validation helpers.

This module centralizes small helpers used across the codebase to ensure
required environment variables are present and to determine whether trading
should be halted based on environment flags.  Implementations are intentionally
lightweight to keep startup costs minimal.
"""

from collections.abc import Mapping
import os
from ai_trading.logging import get_logger

logger = get_logger(__name__)


def _require_env_vars(*names: str, env: Mapping[str, str] | None = None) -> None:
    """Raise ``RuntimeError`` if any of ``names`` are missing in ``env``.

    Parameters
    ----------
    names:
        Environment variable names that must be present.
    env:
        Optional mapping to check instead of ``os.environ``.  The tests use the
        default which reads from the real environment.
    """
    env_map = env or os.environ
    missing = [n for n in names if not env_map.get(n)]
    if missing:
        msg = f"Missing required environment variables: {', '.join(missing)}"
        logger.critical(msg)
        raise RuntimeError(msg)


def require_env_vars(*names: str, env: Mapping[str, str] | None = None) -> bool:
    """Return ``True`` if all ``names`` exist in ``env``.

    The function proxies to :func:`_require_env_vars` but returns ``False``
    instead of raising when variables are missing.  This small convenience is
    used in a few tests.
    """
    try:
        _require_env_vars(*names, env=env)
    except RuntimeError:
        return False
    return True


def _truthy(val: str | None) -> bool:
    return str(val).lower() in {"1", "true", "yes", "on"}


def should_halt_trading(env: Mapping[str, str]) -> bool:
    """Return ``True`` when environment indicates trading should halt.

    A simple guard used in tests; it checks common flag names such as
    ``HALT_TRADING`` or ``TRADING_HALTED`` for truthy values.  Callers supply
    the environment mapping explicitly to ease unit testing.
    """
    return _truthy(env.get("HALT_TRADING") or env.get("TRADING_HALTED"))


__all__ = ["_require_env_vars", "require_env_vars", "should_halt_trading"]
