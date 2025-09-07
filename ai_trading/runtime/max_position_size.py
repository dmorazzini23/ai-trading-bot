"""Runtime helper for resolving maximum position size."""

from __future__ import annotations

from ai_trading.config.management import TradingConfig, get_env
from ai_trading.position_sizing import get_max_position_size as _get_max_position_size


def get_max_position_size(cfg: TradingConfig) -> float:
    """Return max position size honoring ``TradingConfig`` overrides.

    Parameters
    ----------
    cfg:
        Trading configuration instance. If ``max_position_size`` is set on this
        config it takes precedence over environment variables or derived
        values.
    """
    val = getattr(cfg, "max_position_size", None)
    if val is not None:
        return float(val)

    try:
        env_val = get_env("MAX_POSITION_SIZE", cast=float)
    except (ImportError, RuntimeError):
        env_val = None
    if env_val is not None:
        return float(env_val)

    return float(_get_max_position_size(cfg))


__all__ = ["get_max_position_size"]

