"""Configuration auto-fix helpers."""

from __future__ import annotations

from ai_trading.logging import get_logger

_log = get_logger("ai_trading.startup.config")


def ensure_max_position_size(cfg, tcfg) -> float:
    """Ensure ``max_position_size`` is populated with a sensible default.

    Parameters
    ----------
    cfg:
        Primary configuration object used for fallback lookups.
    tcfg:
        Trading settings object expected to carry ``max_position_size``.

    Returns
    -------
    float
        The resolved ``max_position_size``.
    """
    raw = getattr(tcfg, "max_position_size", None)
    try:
        if raw is not None and float(raw) > 0:
            return float(raw)
    except (TypeError, ValueError):
        pass

    from ai_trading.position_sizing import _fallback_max_size

    fallback = float(_fallback_max_size(cfg, tcfg))
    _log.info("CONFIG_AUTOFIX", extra={"field": "max_position_size", "fallback": fallback})
    try:
        setattr(tcfg, "max_position_size", fallback)
    except (AttributeError, TypeError):
        try:
            object.__setattr__(tcfg, "max_position_size", fallback)
        except (AttributeError, TypeError):
            pass
    return fallback


__all__ = ["ensure_max_position_size"]
