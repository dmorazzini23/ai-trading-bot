from __future__ import annotations
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import floor
from typing import Any
from types import SimpleNamespace
import os
from json import JSONDecodeError
from ai_trading.logging import EmitOnceLogger, get_logger
from ai_trading.net.http import get_global_session
from ai_trading.settings import get_alpaca_secret_key_plain
from ai_trading.exc import HTTPError, RequestException
from ai_trading.alpaca_api import ALPACA_AVAILABLE, get_trading_client_cls
_log = get_logger(__name__)
_once_logger = EmitOnceLogger(_log.logger)

@dataclass
class _Cache:
    value: float | None = None
    ts: datetime | None = None
    equity: float | None = None
    equity_error: str | None = None
    equity_missing_logged: bool = False
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
    force_refresh: bool = False,
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

    was_missing_logged = _CACHE.equity_missing_logged
    if force_refresh:
        _CACHE.equity_missing_logged = False

    numeric_equity: float | None
    try:
        numeric_equity = float(equity) if equity is not None else None
    except (TypeError, ValueError):
        numeric_equity = None

    if numeric_equity is None or numeric_equity <= 0.0:
        _once_logger.warning(
            "EQUITY_MISSING",
            key="position_sizing:equity_missing",
            extra={
                "field": "equity",
                "default_equity": default_equity,
                "capital_cap": capital_cap,
            },
        )
        _CACHE.equity_missing_logged = True
    elif was_missing_logged:
        _log.info(
            "EQUITY_RECOVERED",
            extra={"equity": numeric_equity, "capital_cap": capital_cap},
        )
        _CACHE.equity_missing_logged = False
    else:
        _CACHE.equity_missing_logged = False

    basis = numeric_equity if numeric_equity is not None and numeric_equity > 0 else default_equity
    resolved = float(round(capital_cap * basis, 2))
    _log.info(
        "CONFIG_AUTOFIX",
        extra={
            "field": "max_position_size",
            "given": float(provided),
            "fallback": resolved,
            "reason": "derived_from_equity" if numeric_equity is not None and numeric_equity > 0 else "derived_equity_cap",
            "equity": numeric_equity if numeric_equity is not None else equity,
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


def _fetch_equity(cfg, *, force_refresh: bool = False) -> float | None:
    """Fetch account equity using Alpaca SDK or HTTP fallback.

    Parameters
    ----------
    cfg:
        Configuration object containing Alpaca credentials and base URL.
    force_refresh:
        When ``True`` the cached equity value is ignored and a fresh value is
        fetched from the API.

    Returns
    -------
    float | None
        The current account equity. ``0.0`` is returned for recoverable
        errors, ``None`` is returned when credentials are missing, and the
        failure reason is recorded in the log payload.
    """
    if not force_refresh and _CACHE.equity is not None:
        return _CACHE.equity

    base = str(getattr(cfg, "alpaca_base_url", "")).rstrip("/")
    key = getattr(cfg, "alpaca_api_key", None)
    secret = getattr(cfg, "alpaca_secret_key_plain", None) or get_alpaca_secret_key_plain()
    if not key or not secret or not base:
        reason = "missing_credentials"
        _CACHE.equity_error = reason
        _log.warning(
            "ALPACA_EQUITY_UNAVAILABLE",
            extra={
                "reason": reason,
                "has_key": bool(key),
                "has_secret": bool(secret),
                "base_url": base,
            },
        )
        return None

    if ALPACA_AVAILABLE:
        try:
            TradingClient = get_trading_client_cls()
            is_paper = "paper" in base.lower()
            client = TradingClient(
                api_key=key,
                secret_key=secret,
                paper=is_paper,
                url_override=base,
            )
            if hasattr(client, "get_account"):
                acct = client.get_account()
                eq = _coerce_float(getattr(acct, "equity", None), 0.0)
                _CACHE.equity = eq
                _CACHE.equity_error = None
                return eq
            reason = "missing_get_account"
            _log.warning(
                "ALPACA_CLIENT_NO_GET_ACCOUNT",
                extra={"client": type(client).__name__, "reason": reason},
            )
        except Exception as e:  # noqa: BLE001 - log and fallback to HTTP
            reason = f"sdk_error:{type(e).__name__}"
            _CACHE.equity_error = reason
            _log.warning(
                "ALPACA_SDK_ACCOUNT_FAILED",
                extra={"error": str(e), "reason": reason},
            )

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
        _CACHE.equity_error = None
        return eq
    except HTTPError as e:
        resp_obj = getattr(e, "response", None)
        status = getattr(resp_obj, "status_code", None)
        reason = f"http_error:{status}" if status is not None else "http_error"
        _CACHE.equity = 0.0
        _CACHE.equity_error = reason
        if status in {401, 403}:
            _log.warning(
                "ALPACA_AUTH_FAILED",
                extra={"url": url, "status": status, "reason": reason},
            )
        else:
            _log.warning(
                "ALPACA_HTTP_ERROR",
                extra={"url": url, "status": status, "reason": reason},
            )
        return 0.0
    except RequestException as e:
        reason = f"request_error:{type(e).__name__}"
        _CACHE.equity = 0.0
        _CACHE.equity_error = reason
        _log.warning(
            "ALPACA_REQUEST_FAILED",
            extra={"url": url, "error": str(e), "reason": reason},
        )
        return 0.0
    except JSONDecodeError as e:
        reason = "invalid_json"
        _CACHE.equity = 0.0
        _CACHE.equity_error = reason
        _log.warning(
            "ALPACA_INVALID_RESPONSE",
            extra={"url": url, "error": str(e), "reason": reason},
        )
        return 0.0
    except Exception as exc:  # log and propagate unexpected errors
        _CACHE.equity = None
        _CACHE.equity_error = f"unexpected_error:{type(exc).__name__}"
        _log.exception("ALPACA_UNEXPECTED_ERROR", extra={"url": url})
        raise


# Backwards compatibility: older code imports `_get_equity_from_alpaca`
# directly. Keep it as an alias of the new `_fetch_equity` implementation.
_get_equity_from_alpaca = _fetch_equity

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
                # Allow tests to patch the public alias used by runtime.
                fetched = _get_equity_from_alpaca(cfg, force_refresh=force_refresh)
                if fetched is not None and fetched > 0:
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
            cur, source = _resolve_max_position_size(
                cur,
                cap,
                eq,
                default_equity=default_eq,
                force_refresh=force_refresh,
            )
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
    # Use public alias so callers/tests can patch equity retrieval.
    eq = _get_equity_from_alpaca(cfg, force_refresh=force_refresh)
    failure_reason = getattr(_CACHE, 'equity_error', None)
    source = 'alpaca'
    numeric_eq: float | None = None
    if eq is not None:
        try:
            numeric_eq = float(eq)
        except (TypeError, ValueError):
            numeric_eq = None
    if numeric_eq is None or numeric_eq <= 0.0:
        if failure_reason is None and numeric_eq is not None:
            failure_reason = 'non_positive_equity'
        candidates = (
            getattr(cfg, 'equity', None),
            getattr(tcfg, 'equity', None),
            _CACHE.equity,
        )
        for candidate in candidates:
            try:
                candidate_val = float(candidate)
            except (TypeError, ValueError):
                continue
            if candidate_val > 0.0:
                numeric_eq = candidate_val
                source = 'cached_equity'
                break
        if numeric_eq is None or numeric_eq <= 0.0:
            reason = failure_reason or 'equity_unavailable'
            _log.error(
                'AUTO_SIZING_ABORTED',
                extra={'reason': reason, 'capital_cap': cap},
            )
            raise RuntimeError(f"AUTO sizing aborted: {reason}")
        _log.info(
            'AUTO_SIZING_REUSED_EQUITY',
            extra={'reason': failure_reason, 'capital_cap': cap, 'equity': numeric_eq},
        )
        _CACHE.equity = numeric_eq
        _CACHE.equity_error = failure_reason
    else:
        numeric_eq = float(numeric_eq)
        _CACHE.equity = numeric_eq
        _CACHE.equity_error = None
    if cap <= 0.0:
        fb = _fallback_max_size(cfg, tcfg)
        _log.info(
            "CONFIG_AUTOFIX",
            extra={
                'field': 'max_position_size',
                'given': 0.0,
                'fallback': fb,
                'reason': 'missing_capital_cap',
                'equity': numeric_eq,
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
                'equity': numeric_eq,
                'capital_cap': cap,
                'clamp_min': vmin,
                'clamp_max': vmax,
                'refreshed_at': _CACHE.ts.isoformat(),
            },
        )
    computed = float(floor(numeric_eq * cap))
    val = _clamp(
        computed,
        _coerce_float(vmin, None) if vmin is not None else None,
        _coerce_float(vmax, None) if vmax is not None else None,
    )
    if val <= 0.0:
        fb = _fallback_max_size(cfg, tcfg)
        _log.info(
            "CONFIG_AUTOFIX",
            extra={
                'field': 'max_position_size',
                'given': computed,
                'fallback': fb,
                'reason': 'non_positive_computed',
                'equity': numeric_eq,
                'capital_cap': cap,
            },
        )
        val = _clamp(fb, vmin, vmax)
        source = 'fallback'
    _CACHE.value, _CACHE.ts = (val, _now_utc())
    return (
        val,
        {
            'mode': mode,
            'source': source,
            'equity': numeric_eq,
            'capital_cap': cap,
            'computed': computed,
            'clamp_min': vmin,
            'clamp_max': vmax,
            'refreshed_at': _CACHE.ts.isoformat(),
        },
    )


def get_max_position_size(
    cfg: Any | None = None,
    tcfg: Any | None = None,
    *,
    auto: bool = False,
    force_refresh: bool = False,
) -> float:
    """Return only the resolved ``max_position_size`` value.

    Parameters
    ----------
    cfg, tcfg:
        Optional configuration objects. When ``tcfg`` is ``None`` it defaults
        to ``cfg``. If both are ``None`` simple namespaces are used so the
        resolver can still apply fallback logic.
    auto:
        When ``True`` the function overrides any ``max_position_mode`` on the
        trading config and forces :func:`resolve_max_position_size` to operate
        in automatic mode where the current equity is multiplied by
        ``capital_cap``. If equity retrieval or ``capital_cap`` resolution
        fails, the resolver falls back to default sizing.
    force_refresh:
        When ``True`` cached values are ignored.
    """

    cfg = cfg or SimpleNamespace()
    tcfg = tcfg or cfg
    if auto:
        data = vars(tcfg).copy()
        data["max_position_mode"] = "AUTO"
        tcfg = SimpleNamespace(**data)
    val, _ = resolve_max_position_size(cfg, tcfg, force_refresh=force_refresh)
    return val


__all__ = ["resolve_max_position_size", "get_max_position_size"]
