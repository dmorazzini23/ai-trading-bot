"""Helpers for deciding whether the SIP feed may be used."""

from __future__ import annotations

from ai_trading.utils.environment import env


def sip_disallowed() -> bool:
    """Return ``True`` when the SIP feed should not be used."""

    if not env.ALPACA_ALLOW_SIP:
        return True

    has_creds = bool(env.ALPACA_KEY) and bool(env.ALPACA_SECRET)
    explicit_entitlement = getattr(env, "ALPACA_SIP_ENTITLED", None)
    if explicit_entitlement is not None:
        if not bool(explicit_entitlement):
            return True
        return not has_creds
    if hasattr(env, "ALPACA_HAS_SIP") and (env.ALPACA_HAS_SIP is not None):
        return not (has_creds and bool(env.ALPACA_HAS_SIP))
    return not has_creds


__all__ = ["sip_disallowed"]

