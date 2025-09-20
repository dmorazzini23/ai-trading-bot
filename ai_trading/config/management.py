from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Mapping, Optional, TypeVar

from pathlib import Path

from dotenv import load_dotenv

from ai_trading.logging import logger
from .runtime import (
    CONFIG_SPECS,
    SPEC_BY_ENV,
    TradingConfig,
    generate_config_schema,
    get_trading_config,
    reload_trading_config,
)

T = TypeVar("T")


def reload_env(path: str | os.PathLike[str] | None = None, override: bool = True) -> str | None:
    """Reload environment variables from a dotenv file."""

    if path is None:
        candidate = Path.cwd() / ".env"
        path = candidate if candidate.exists() else None
    if path is None:
        reload_trading_config()
        return None
    load_dotenv(dotenv_path=path, override=override)
    reload_trading_config()
    return os.fspath(path)


def _coerce(value: Any, cast: Optional[Callable[[Any], T]]) -> T | Any:
    if cast is None:
        return value
    if cast is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    try:
        return cast(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Failed to cast value {value!r} using {cast}: {exc}") from exc


def get_env(
    key: str,
    default: Optional[str] = None,
    *,
    cast: Optional[Callable[[Any], T]] = None,
    required: bool = False,
) -> T | Any:
    """Compatibility shim returning values from :class:`TradingConfig`.

    Prefer using :func:`get_trading_config` directly; this helper exists to
    avoid touching legacy call-sites in a single patch.
    """

    spec = SPEC_BY_ENV.get(key.upper())
    if spec is None:
        raw = os.environ.get(key)
        if raw is None:
            if required:
                raise RuntimeError(f"Missing required environment variable: {key}")
            return default
        return _coerce(raw, cast)

    cfg = get_trading_config()
    value = getattr(cfg, spec.field)
    if value in (None, ""):
        if required:
            raise RuntimeError(f"Missing required environment variable: {spec.env[0]}")
        return default
    return _coerce(value, cast)


def is_shadow_mode() -> bool:
    return bool(get_trading_config().shadow_mode)


def validate_required_env(
    keys: Iterable[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Mapping[str, str]:
    """Ensure mandatory Alpaca credentials and risk limits are present."""

    cfg = TradingConfig.from_env(env) if env is not None else get_trading_config()

    required_fields = {
        "ALPACA_API_KEY": cfg.alpaca_api_key,
        "ALPACA_SECRET_KEY": cfg.alpaca_secret_key,
        "ALPACA_DATA_FEED": cfg.alpaca_data_feed,
        "ALPACA_API_URL": cfg.alpaca_base_url,
        "WEBHOOK_SECRET": cfg.webhook_secret,
        "CAPITAL_CAP": cfg.capital_cap,
        "DOLLAR_RISK_LIMIT": cfg.dollar_risk_limit,
    }
    if keys is not None:
        filtered: dict[str, str | None] = {}
        for key in keys:
            if key in required_fields:
                filtered[key] = required_fields[key]
            else:
                filtered[key] = getattr(cfg, key.lower(), None)
        required_fields = filtered

    env_lookup: dict[str, str] = {}
    if env is not None:
        env_lookup = {k.upper(): str(v) for k, v in env.items() if v not in (None, "")}
    if "ALPACA_API_URL" not in env_lookup and "ALPACA_BASE_URL" in env_lookup:
        env_lookup["ALPACA_API_URL"] = env_lookup["ALPACA_BASE_URL"]
    alias_sources: dict[str, tuple[str, ...]] = {
        "ALPACA_API_URL": ("ALPACA_BASE_URL",),
    }

    for key, value in list(required_fields.items()):
        if value in (None, ""):
            fallback = env_lookup.get(key)
            if fallback in (None, ""):
                fallback = os.environ.get(key)
            if fallback not in (None, ""):
                required_fields[key] = fallback
                continue
            for alias in alias_sources.get(key, ()):  # pragma: no branch - small tuple
                alias_value = env_lookup.get(alias)
                if alias_value in (None, ""):
                    alias_value = os.environ.get(alias)
                if alias_value not in (None, ""):
                    required_fields[key] = alias_value
                    break

    missing = [name for name, value in required_fields.items() if not value]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    masked: dict[str, str] = {}
    for name, value in required_fields.items():
        masked[name] = "***" if value else ""
    return masked


def _resolve_alpaca_env() -> tuple[str | None, str | None, str | None]:
    cfg = get_trading_config()
    return cfg.alpaca_api_key, cfg.alpaca_secret_key, cfg.alpaca_base_url


def validate_alpaca_credentials() -> None:
    try:
        validate_required_env(("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_API_URL"))
    except RuntimeError as exc:
        logger.error("ALPACA_CREDENTIALS_INVALID", extra={"error": str(exc)})
        raise


def get_config_schema() -> str:
    return generate_config_schema()


SEED = get_trading_config().seed
MAX_EMPTY_RETRIES = get_trading_config().max_empty_retries


__all__ = [
    "TradingConfig",
    "CONFIG_SPECS",
    "get_trading_config",
    "reload_trading_config",
    "reload_env",
    "get_env",
    "is_shadow_mode",
    "validate_required_env",
    "validate_alpaca_credentials",
    "_resolve_alpaca_env",
    "get_config_schema",
    "SEED",
    "MAX_EMPTY_RETRIES",
]
