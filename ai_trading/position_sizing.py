from __future__ import annotations
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import floor
from typing import Any
import os
from json import JSONDecodeError
from ai_trading.logging import get_logger, logger_once
from ai_trading.net.http import get_global_session
from ai_trading.settings import get_alpaca_secret_key_plain
from ai_trading.exc import HTTPError, RequestException
_log = get_logger(__name__)

@dataclass
class _Cache:
    value: float | None = None
    ts: datetime | None = None
    equity: float | None = None
_CACHE = _Cache()

def _now_utc() -> datetime:
    return datetime.now(tz=UTC)

def _should_refresh(ttl_seconds: float) -> bool:
    if _CACHE.ts is None:
        return True
    return _now_utc() - _CACHE.ts >= timedelta(seconds=ttl_seconds)

def _coerce_float(val: Any, default: float=0.0) -> float:
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
    default_equity: float = 200000.0,
) -> tuple[float, str]:
    """Resolve ``max_position_size`` with strict validation.

    When ``equity`` is ``None`` or non-positive, ``default_equity`` is used
    as the basis for the derived position size.
    """

    env_val = os.getenv("AI_TRADING_MAX_POSITION_SIZE")
    if env_val is not None:
        try:
            v = float(env_val)
        except ValueError as e:  # noqa: BLE001
            raise ValueError(
                f"AI_TRADING_MAX_POSITION_SIZE must be numeric, got {env_val!r}"
            ) from e
        if v <= 0:
            raise ValueError("AI_TRADING_MAX_POSITION_SIZE must be positive")
        return (v, "env_override")

    if provided < 0:
        raise ValueError("max_position_size must be positive")
    if provided > 0:
        return (float(provided), "provided")

    if equity is None:
        logger_once.warning(
            "EQUITY_MISSING",
            key="equity_missing",
            extra={"field": "equity", "default_equity": default_equity, "capital_cap": capital_cap},
        )
    basis = equity if equity is not None and equity > 0 else default_equity
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
    return (resolved, "autofix")

def _fallback_max_size(cfg, tcfg) -> float:
    env_val = os.getenv("AI_TRADING_MAX_POSITION_SIZE")
    if env_val is not None:
        try:
            v = float(env_val)
            if v > 0:
                return v
        except ValueError:
            _log.warning("INVALID_MAX_POSITION_SIZE", extra={"value": env_val})
    for name in ('max_position_size_fallback', 'max_position_size_default'):
        v = getattr(tcfg, name, None)
        if v is not None:
            return _coerce_float(v, 8000.0)
    v = getattr(cfg, 'default_max_position_size', None)
    if v is not None:
        return _coerce_float(v, 8000.0)
    return 8000.0


def _get_equity_from_alpaca(cfg, *, force_refresh: bool = False) -> float:
    """Fetch account equity using Alpaca SDK or HTTP fallback.

    The result is cached to avoid repeated network calls. When ``force_refresh``
    is ``True`` the cache is bypassed.

    Returns ``0.0`` on any error (caller will fallback).
    """
    if not force_refresh and _CACHE.equity is not None:
        return _CACHE.equity

    base = str(getattr(cfg, "alpaca_base_url", "")).rstrip("/")
    key = getattr(cfg, "alpaca_api_key", None)
    secret = getattr(cfg, "alpaca_secret_key_plain", None) or get_alpaca_secret_key_plain()
    if not key or not secret or not base:
        _CACHE.equity = 0.0
        return 0.0

    try:  # Prefer alpaca-py when available
        from alpaca.trading.client import TradingClient  # type: ignore

        client = TradingClient(api_key=key, secret_key=secret, url_override=base)
        acct = client.get_account()
        eq = _coerce_float(getattr(acct, "equity", None), 0.0)
        _CACHE.equity = eq
        return eq
    except ModuleNotFoundError:
        pass
    except Exception as e:  # noqa: BLE001 - log and fallback to HTTP
        _log.warning("ALPACA_SDK_ACCOUNT_FAILED", extra={"error": str(e)})

    url = f"{base}/v2/account"
    try:
        s = get_global_session()
        resp = s.get(url, headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
        if hasattr(resp, "raise_for_status"):
            resp.raise_for_status()
        else:
            status = getattr(resp, "status_code", None)
            if status != 200:
                err = HTTPError(f"HTTP {status}")
                setattr(err, "response", resp)
                raise err
        data = resp.json()
        eq = _coerce_float(data.get("equity"), 0.0)
        _CACHE.equity = eq
        return eq
    except HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status in {401, 403}:
            _log.warning("ALPACA_AUTH_FAILED", extra={"url": url, "status": status})
        else:
            _log.warning("ALPACA_HTTP_ERROR", extra={"url": url, "status": status})
    except RequestException as e:
        _log.warning("ALPACA_REQUEST_FAILED", extra={"url": url, "error": str(e)})
    except JSONDecodeError as e:
        _log.warning("ALPACA_INVALID_RESPONSE", extra={"url": url, "error": str(e)})
    except Exception:  # log and propagate unexpected errors
        _log.exception("ALPACA_UNEXPECTED_ERROR", extra={"url": url})
        raise
    _CACHE.equity = 0.0
    return 0.0

def resolve_max_position_size(cfg, tcfg, *, force_refresh: bool=False) -> tuple[float, dict[str, Any]]:
    """Resolve max_position_size according to mode and settings.

    Set ``force_refresh`` to ``True`` to bypass cached values.
    """
    mode = str(getattr(tcfg, 'max_position_mode', getattr(cfg, 'max_position_mode', 'STATIC'))).upper()
    ttl = float(getattr(tcfg, 'dynamic_size_refresh_secs', getattr(cfg, 'dynamic_size_refresh_secs', 3600.0)))
    cap = _coerce_float(getattr(tcfg, 'capital_cap', 0.0), 0.0)
    vmin = getattr(tcfg, 'max_position_size_min', None)
    vmax = getattr(tcfg, 'max_position_size_max', None)
    default_eq = _coerce_float(
        getattr(tcfg, 'max_position_equity_fallback', getattr(cfg, 'max_position_equity_fallback', 200000.0)),
        200000.0,
    )
    if mode != 'AUTO':
        if not force_refresh and _CACHE.value is not None:
            return (
                _CACHE.value,
                {
                    'mode': mode,
                    'source': 'cache',
                    'capital_cap': cap,
                    'refreshed_at': (_CACHE.ts or _now_utc()).isoformat(),
                },
            )
        raw_val = getattr(tcfg, 'max_position_size', None)
        cur = _coerce_float(raw_val, 0.0)
        source = 'static'
        if cur <= 0.0:
            if raw_val is not None:
                raise ValueError('max_position_size must be positive')
            eq = getattr(tcfg, 'equity', getattr(cfg, 'equity', None))
            if eq in (None, 0.0):
                fetched = _get_equity_from_alpaca(cfg, force_refresh=force_refresh)
                if fetched > 0:
                    eq = fetched
                    for obj in (cfg, tcfg):
                        try:
                            setattr(obj, 'equity', eq)
                        except Exception:
                            try:
                                object.__setattr__(obj, 'equity', eq)
                            except Exception:  # pragma: no cover - defensive
                                pass
                else:
                    eq = None
            cur, source = _resolve_max_position_size(cur, cap, eq, default_equity=default_eq)
            _CACHE.equity = eq
        _CACHE.value, _CACHE.ts = (cur, _now_utc())
        return (
            cur,
            {
                'mode': mode,
                'source': source,
                'capital_cap': cap,
                'refreshed_at': _CACHE.ts.isoformat(),
            },
        )
    if not force_refresh and (not _should_refresh(ttl)) and (_CACHE.value is not None):
        return (_CACHE.value, {'mode': mode, 'source': 'cache', 'capital_cap': cap, 'refreshed_at': (_CACHE.ts or _now_utc()).isoformat()})
    eq = _get_equity_from_alpaca(cfg, force_refresh=force_refresh)
    _CACHE.equity = eq
    if eq <= 0.0 or cap <= 0.0:
        fb = _fallback_max_size(cfg, tcfg)
        _log.info(
            "CONFIG_AUTOFIX",
            extra={
                'field': 'max_position_size',
                'given': 0.0,
                'fallback': fb,
                'reason': 'missing_equity_or_cap',
                'equity': eq,
                'capital_cap': cap,
            },
        )
        val = _clamp(fb, vmin, vmax)
        _CACHE.value, _CACHE.ts = (val, _now_utc())
        return (
            val,
            {
                'mode': mode,
                'source': 'fallback',
                'equity': eq,
                'capital_cap': cap,
                'clamp_min': vmin,
                'clamp_max': vmax,
                'refreshed_at': _CACHE.ts.isoformat(),
            },
        )
    computed = float(floor(eq * cap))
    val = _clamp(
        computed,
        _coerce_float(vmin, None) if vmin is not None else None,
        _coerce_float(vmax, None) if vmax is not None else None,
    )
    if val <= 0.0:
        val = _fallback_max_size(cfg, tcfg)
    _CACHE.value, _CACHE.ts = (val, _now_utc())
    return (val, {'mode': mode, 'source': 'alpaca', 'equity': eq, 'capital_cap': cap, 'computed': computed, 'clamp_min': vmin, 'clamp_max': vmax, 'refreshed_at': _CACHE.ts.isoformat()})


def get_max_position_size(cfg, tcfg, *, force_refresh: bool = False) -> float:
    """Return only the resolved ``max_position_size`` value.

    This is a thin convenience wrapper around :func:`resolve_max_position_size`
    used by modules that only care about the numeric size and not the
    accompanying metadata. Set ``force_refresh`` to ``True`` to bypass any
    cached values.
    """

    val, _ = resolve_max_position_size(cfg, tcfg, force_refresh=force_refresh)
    return val


__all__ = ["resolve_max_position_size", "get_max_position_size"]

