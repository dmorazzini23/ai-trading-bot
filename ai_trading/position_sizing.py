from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import floor
from typing import Any

from ai_trading.logging import get_logger  # AI-AGENT-REF: structured logging
from ai_trading.net.http import get_global_session

_log = get_logger(__name__)


@dataclass
class _Cache:
    value: float | None = None
    ts: datetime | None = None


_CACHE = _Cache()


def _now_utc() -> datetime:
    return datetime.now(tz=UTC)


def _should_refresh(ttl_seconds: float) -> bool:
    if _CACHE.ts is None:
        return True
    return _now_utc() - _CACHE.ts >= timedelta(seconds=ttl_seconds)


def _coerce_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _clamp(val: float, vmin: float | None, vmax: float | None) -> float:
    if vmin is not None:
        val = max(val, vmin)
    if vmax is not None:
        val = min(val, vmax)
    return val


def _resolve_max_position_size(
    provided: float,
    capital_cap: float,
    equity: float | None,
    *,
    default_equity: float = 200_000.0,
) -> float:
    """Autofix nonpositive max_position_size using equity caps."""  # AI-AGENT-REF
    if provided > 0:
        return float(provided)
    basis = equity if (equity is not None and equity > 0) else default_equity
    resolved = float(round(capital_cap * basis, 2))
    _log.info(
        "CONFIG_AUTOFIX",
        extra={
            "field": "max_position_size",
            "given": float(provided),
            "fallback": resolved,
            "reason": "derived_equity_cap" if equity in (None, 0) else "derived_from_equity",
            "equity": equity,
            "capital_cap": capital_cap,
        },
    )
    return resolved


def _fallback_max_size(cfg, tcfg) -> float:
    for name in ("max_position_size_fallback", "max_position_size_default"):
        v = getattr(tcfg, name, None)
        if v is not None:
            return _coerce_float(v, 8000.0)
    v = getattr(cfg, "default_max_position_size", None)
    if v is not None:
        return _coerce_float(v, 8000.0)
    return 8000.0


def _get_equity_from_alpaca(cfg) -> float:
    """Fetch account equity from Alpaca /v2/account.

    Returns 0.0 on any error (caller will fallback).
    """  # AI-AGENT-REF: external equity fetch
    try:
        base = str(getattr(cfg, "alpaca_base_url", "")).rstrip("/")
        url = f"{base}/v2/account"
        key = getattr(cfg, "alpaca_api_key", None)
        secret = getattr(cfg, "alpaca_secret_key_plain", None)
        if not key or not secret or not base:
            return 0.0
        s = get_global_session()
        resp = s.get(
            url,
            headers={
                "APCA-API-KEY-ID": key,
                "APCA-API-SECRET-KEY": secret,
            },
        )
        if getattr(resp, "status_code", 0) != 200:
            return 0.0
        data = resp.json()  # type: ignore[no-any-return]
        eq = _coerce_float(data.get("equity"), 0.0)
        return eq
    except (ValueError, TypeError):
        return 0.0


def resolve_max_position_size(cfg, tcfg, *, force_refresh: bool = False) -> tuple[float, dict[str, Any]]:
    """Resolve max_position_size according to mode and settings."""  # AI-AGENT-REF: AUTO sizing resolver
    mode = str(getattr(tcfg, "max_position_mode", getattr(cfg, "max_position_mode", "STATIC"))).upper()
    ttl = float(getattr(tcfg, "dynamic_size_refresh_secs", getattr(cfg, "dynamic_size_refresh_secs", 3600.0)))
    cap = _coerce_float(getattr(tcfg, "capital_cap", 0.0), 0.0)
    vmin = getattr(tcfg, "max_position_size_min", None)
    vmax = getattr(tcfg, "max_position_size_max", None)

    if mode != "AUTO":  # Static path
        cur = _coerce_float(getattr(tcfg, "max_position_size", 0.0), 0.0)
        if cur <= 0.0:
            cur = _fallback_max_size(cfg, tcfg)
        _CACHE.value, _CACHE.ts = cur, _now_utc()
        return cur, {
            "mode": mode,
            "source": "static",
            "capital_cap": cap,
            "refreshed_at": _CACHE.ts.isoformat(),
        }

    if not force_refresh and not _should_refresh(ttl) and _CACHE.value is not None:
        return _CACHE.value, {
            "mode": mode,
            "source": "cache",
            "capital_cap": cap,
            "refreshed_at": (_CACHE.ts or _now_utc()).isoformat(),
        }

    eq = _get_equity_from_alpaca(cfg)
    if eq <= 0.0 or cap <= 0.0:
        fb = _fallback_max_size(cfg, tcfg)
        val = _clamp(fb, vmin, vmax)
        _CACHE.value, _CACHE.ts = val, _now_utc()
        return val, {
            "mode": mode,
            "source": "fallback",
            "equity": eq,
            "capital_cap": cap,
            "clamp_min": vmin,
            "clamp_max": vmax,
            "refreshed_at": _CACHE.ts.isoformat(),
        }

    computed = float(floor(eq * cap))
    val = _clamp(computed, _coerce_float(vmin, None) if vmin is not None else None,
                 _coerce_float(vmax, None) if vmax is not None else None)
    if val <= 0.0:
        val = _fallback_max_size(cfg, tcfg)
    _CACHE.value, _CACHE.ts = val, _now_utc()
    return val, {
        "mode": mode,
        "source": "alpaca",
        "equity": eq,
        "capital_cap": cap,
        "computed": computed,
        "clamp_min": vmin,
        "clamp_max": vmax,
        "refreshed_at": _CACHE.ts.isoformat(),
    }

