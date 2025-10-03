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
    if allow_flag is False:
        return True

    explicit_entitlement = _coerce_flag(getattr(env, "ALPACA_SIP_ENTITLED", None))
    if explicit_entitlement is not None:
        return not explicit_entitlement

    has_sip = _coerce_flag(getattr(env, "ALPACA_HAS_SIP", None))
    if has_sip is not None:
        return not has_sip

    if not _has_alpaca_credentials():
        return True

    return False


__all__ = ["sip_disallowed", "sip_credentials_missing"]

