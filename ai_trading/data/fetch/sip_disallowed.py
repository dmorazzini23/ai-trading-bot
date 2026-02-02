"""Helpers for deciding whether the SIP feed may be used."""

from __future__ import annotations

from ai_trading.utils.environment import env


def _has_alpaca_credentials() -> bool:
    return bool(env.ALPACA_KEY) and bool(env.ALPACA_SECRET)


def sip_credentials_missing() -> bool:
    """Return ``True`` when Alpaca API credentials are absent."""

    return not _has_alpaca_credentials()


def sip_disallowed() -> bool:
    """Return ``True`` when the SIP feed should not be used."""

    def _coerce_flag(value):
        if isinstance(value, bool):
            return value
        if value is None:
            return None
        try:
            lowered = value.strip().lower()
        except AttributeError:
            return bool(value)
        if lowered in {"1", "true", "yes", "on", "enable", "enabled"}:
            return True
        if lowered in {"0", "false", "no", "off", "disable", "disabled"}:
            return False
        return None

    allow_flag = _coerce_flag(getattr(env, "ALPACA_ALLOW_SIP", None))
    explicit_entitlement = _coerce_flag(getattr(env, "ALPACA_SIP_ENTITLED", None))
    has_sip = _coerce_flag(getattr(env, "ALPACA_HAS_SIP", None))

    if any(flag is False for flag in (allow_flag, explicit_entitlement, has_sip)):
        return True

    if allow_flag is True and (has_sip is True or explicit_entitlement is True):
        return False

    return True


__all__ = ["sip_disallowed", "sip_credentials_missing"]
