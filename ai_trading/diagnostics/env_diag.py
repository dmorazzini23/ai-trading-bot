from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

from pathlib import Path
from typing import Any, Dict, Optional

from ai_trading.config.management import get_env, is_shadow_mode


_MASK = "***"


def _mask(value: str | None, keep: int = 4) -> str:
    """Return a masked representation of sensitive identifiers."""

    if not value:
        return ""
    prefix = str(value)[: max(keep, 0)]
    if not prefix:
        return _MASK
    return f"{prefix}{_MASK}"


def _get_env_str(name: str, default: str = "", *, resolve_aliases: bool = True) -> str:
    """Return an environment value via config management with safe fallback."""

    value = get_env(name, default, cast=str, resolve_aliases=resolve_aliases)
    if value is None:
        return default
    return str(value)


def _resolve_trading_base_url() -> str:
    return _get_env_str(
        "ALPACA_TRADING_BASE_URL",
        "",
        resolve_aliases=False,
    ).strip()


def _resolve_data_base_url() -> str:
    data_url = _get_env_str("ALPACA_DATA_BASE_URL", "")
    return data_url


def _resolve_configured_feed() -> str:
    feed = _get_env_str("ALPACA_DATA_FEED", "")
    if not feed:
        feed = _get_env_str("DATA_FEED", "")
    return feed or "(unset)"


def gather_alpaca_diag(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Collect Alpaca environment diagnostics with secrets redacted."""

    trading_base_url = _resolve_trading_base_url()
    data_base_url = _resolve_data_base_url()
    key_id = _get_env_str("ALPACA_API_KEY_ID", "")
    trading_key = _get_env_str("ALPACA_API_KEY", "")
    secret_key = _get_env_str("ALPACA_SECRET_KEY", "")

    missing_base_url = not bool(trading_base_url)
    environment = (
        "missing"
        if missing_base_url
        else "paper" if "paper" in trading_base_url.lower() else "live"
    )
    diag: Dict[str, Any] = {
        "trading_base_url": trading_base_url,
        "trading_base_url_missing": missing_base_url,
        "data_base_url": data_base_url,
        "environment": environment,
        "paper": environment == "paper",
        "key_id_masked": _mask(key_id or trading_key),
        "has_key": bool(trading_key or key_id),
        "has_secret": bool(secret_key),
        "shadow_mode": bool(is_shadow_mode()),
        "configured_feed": _resolve_configured_feed(),
        "cwd": str(Path.cwd()),
    }
    if extra:
        diag.update(extra)
    return diag


def gather_env_diag(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a full diagnostics payload suitable for HTTP responses."""

    payload: Dict[str, Any] = {"alpaca": gather_alpaca_diag()}
    if extra:
        payload.update(extra)
    return payload


def log_env_diag(
    logger: Any,
    *,
    once: bool = True,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Emit the ALPACA_DIAG log with structured, redacted details."""

    payload = gather_alpaca_diag(extra)
    try:
        if once:
            try:
                from ai_trading.logging import logger_once
            except AI_TRADING_FALLBACK_EXCEPTIONS:
                logger_once = None  # type: ignore[assignment]
            if logger_once is not None:
                logger_once.info("ALPACA_DIAG", extra=payload, key="alpaca_diag_once")
                return payload
        logger.info("ALPACA_DIAG", extra=payload)
    except AI_TRADING_FALLBACK_EXCEPTIONS:  # pragma: no cover - do not break startup on logging issues
        try:
            logger.debug("ALPACA_DIAG_LOG_FAILED", exc_info=True)
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            pass
    return payload
