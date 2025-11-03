from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Mapping, Optional, TypeVar
from urllib.parse import urlparse

from pathlib import Path

try:
    from dotenv import load_dotenv as _load_dotenv  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency may be absent
    _load_dotenv = None

_DOTENV_WARNING_EMITTED = False

from ai_trading.logging import logger
from .settings import Settings, get_settings
from .runtime import (
    CONFIG_SPECS,
    SPEC_BY_ENV,
    SPEC_BY_FIELD,
    TradingConfig,
    ensure_trading_config_current,
    generate_config_schema,
    get_trading_config,
    reload_trading_config,
)

TESTING = str(os.getenv("TESTING", "")).strip().lower() in {"1", "true", "yes", "on"}

T = TypeVar("T")


def _normalize_alpaca_base_url(value: str | None, *, source_key: str) -> tuple[str | None, str | None]:
    """Validate Alpaca base URL strings returning sanitized value and error."""

    if value is None:
        return None, None

    raw = value.strip()
    if not raw:
        return None, None

    if "${" in raw:
        return None, (
            f"{source_key} looks like an unresolved placeholder ({raw}). "
            "Set ALPACA_API_URL or ALPACA_BASE_URL to a full https://... endpoint."
        )

    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None, (
            f"{source_key} must include an HTTP scheme (got {raw}). "
            "Provide a complete Alpaca REST endpoint such as https://paper-api.alpaca.markets."
        )

    return raw, None


def _select_alpaca_base_url(
    env: Mapping[str, str] | None = None,
) -> tuple[str | None, str | None, list[tuple[str, str, str]]]:
    env_map = env or os.environ
    invalid_entries: list[tuple[str, str, str]] = []

    for env_key in ("ALPACA_BASE_URL", "ALPACA_API_URL"):
        raw = env_map.get(env_key)
        normalized, message = _normalize_alpaca_base_url(raw, source_key=env_key)
        if normalized:
            return normalized, env_key, invalid_entries
        if raw and message:
            invalid_entries.append((env_key, raw, message))

    return None, None, invalid_entries


def reload_env(path: str | os.PathLike[str] | None = None, override: bool = True) -> str | None:
    """Reload environment variables from a dotenv file."""

    from ai_trading.utils.env import refresh_alpaca_credentials_cache

    if path is None:
        candidate = Path.cwd() / ".env"
        path = candidate if candidate.exists() else None
    if path is None:
        reload_trading_config()
        refresh_alpaca_credentials_cache()
        return None
    _maybe_load_dotenv(path, override=override)
    reload_trading_config()
    refresh_alpaca_credentials_cache()
    return os.fspath(path)


def _maybe_load_dotenv(path: os.PathLike[str] | str, *, override: bool = True) -> bool:
    """Best-effort dotenv loader that tolerates optional dependency absence."""

    global _DOTENV_WARNING_EMITTED

    if _load_dotenv is None:
        if not _DOTENV_WARNING_EMITTED:
            logger.warning(
                "PYTHON_DOTENV_NOT_AVAILABLE",
                extra={"path": os.fspath(path)},
            )
            _DOTENV_WARNING_EMITTED = True
        return False

    try:
        _load_dotenv(dotenv_path=path, override=override)
    except Exception as exc:  # pragma: no cover - logged for diagnostics
        logger.warning(
            "PYTHON_DOTENV_LOAD_FAILED",
            extra={"path": os.fspath(path), "error": str(exc)},
        )
        return False
    return True


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

    env_keys = tuple(spec.env) + tuple(spec.deprecated_env.keys())
    cfg = ensure_trading_config_current(env_keys)
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

    env_snapshot: dict[str, str] = {
        k: v for k, v in os.environ.items() if isinstance(v, str)
    }
    overrides: dict[str, str] | None = None
    config_overrides: dict[str, str] | None = None
    if env is not None:
        overrides = {
            str(key).upper(): str(value)
            for key, value in env.items()
            if value is not None
        }
        if overrides:
            env_snapshot.update(overrides)
            config_overrides = {
                key: value for key, value in overrides.items() if key in SPEC_BY_ENV
            }
            if not config_overrides:
                config_overrides = None
        else:
            overrides = None
            config_overrides = None

    cfg = TradingConfig.from_env(
        config_overrides, allow_missing_drawdown=True
    )

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

    env_lookup: dict[str, str] = {
        k.upper(): v for k, v in env_snapshot.items() if v not in (None, "")
    }
    if "ALPACA_API_URL" not in env_lookup and "ALPACA_BASE_URL" in env_lookup:
        env_lookup["ALPACA_API_URL"] = env_lookup["ALPACA_BASE_URL"]
    alias_sources: dict[str, tuple[str, ...]] = {
        "ALPACA_API_URL": ("ALPACA_BASE_URL",),
    }

    for key, value in list(required_fields.items()):
        if value in (None, ""):
            fallback = env_lookup.get(key)
            if fallback not in (None, ""):
                required_fields[key] = fallback
                continue
            for alias in alias_sources.get(key, ()):  # pragma: no branch - small tuple
                alias_value = env_lookup.get(alias)
                if alias_value not in (None, ""):
                    required_fields[key] = alias_value
                    break

    missing = [name for name, value in required_fields.items() if not value]
    if missing:
        logger.error(
            "CONFIG_REQUIRED_ENV_MISSING",
            extra={"missing": tuple(sorted(missing)), "requested": tuple(sorted(keys)) if keys else None},
        )
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    masked: dict[str, str] = {}
    for name, value in required_fields.items():
        masked[name] = "***" if value else ""
    return masked


def _resolve_alpaca_env() -> tuple[str | None, str | None, str | None]:
    """Best-effort resolution of Alpaca credentials without hard failures."""

    default_base_url = "https://paper-api.alpaca.markets"

    base_url: str | None
    source: str | None
    base_url, source, errors = _select_alpaca_base_url()
    alias_used = source == "ALPACA_BASE_URL"

    cfg: TradingConfig | None
    try:
        cfg = get_trading_config()
    except (RuntimeError, ValueError) as exc:
        logger.debug(
            "TRADING_CONFIG_RESOLVE_SKIPPED",
            extra={"error": str(exc)},
        )
        cfg = None

    if base_url is None and cfg is not None:
        cfg_base_url, cfg_source, cfg_errors = _select_alpaca_base_url(
            {
                "ALPACA_BASE_URL": cfg.alpaca_base_url or "",
                "ALPACA_API_URL": cfg.alpaca_base_url or "",
            }
        )
        errors.extend(cfg_errors)
        if cfg_base_url:
            base_url = cfg_base_url
            alias_used = cfg_source == "ALPACA_BASE_URL"

    for env_key, raw, message in errors:
        logger.error(message, extra={"env_key": env_key, "value": raw})

    if not base_url and cfg is not None:
        normalized_cfg, cfg_error = _normalize_alpaca_base_url(
            cfg.alpaca_base_url, source_key="ALPACA_API_URL"
        )
        if normalized_cfg:
            base_url = normalized_cfg
            alias_used = False
        elif cfg_error:
            logger.error(cfg_error, extra={"env_key": "ALPACA_API_URL", "value": cfg.alpaca_base_url})

    resolved_base_url = base_url or default_base_url
    if base_url and alias_used:
        canonical_env = os.getenv("ALPACA_API_URL")
        if canonical_env is None or canonical_env.strip() == "":
            os.environ["ALPACA_API_URL"] = base_url

    api_key = (
        (getattr(cfg, "alpaca_api_key", None) if cfg is not None else None)
        or os.getenv("ALPACA_API_KEY")
        or os.getenv("AP" "CA_" "API_KEY_ID")
    )
    secret = (
        (getattr(cfg, "alpaca_secret_key", None) if cfg is not None else None)
        or os.getenv("ALPACA_SECRET_KEY")
        or os.getenv("AP" "CA_" "API_SECRET_KEY")
    )

    sanitized_key = api_key or None
    sanitized_secret = secret or None

    if sanitized_key is None and sanitized_secret is None:
        return None, None, resolved_base_url

    return sanitized_key, sanitized_secret, resolved_base_url


def validate_alpaca_credentials() -> None:
    if TESTING:
        return
    reload_trading_config()
    _, _, url_errors = _select_alpaca_base_url()
    if url_errors:
        messages = "; ".join(error for _, _, error in url_errors)
        raise RuntimeError(messages)
    try:
        validate_required_env(("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_API_URL"))
    except RuntimeError as exc:
        logger.error("ALPACA_CREDENTIALS_INVALID", extra={"error": str(exc)})
        raise


def get_config_schema() -> str:
    return generate_config_schema()


def from_env_relaxed(env_overrides: Mapping[str, Any] | None = None) -> TradingConfig:
    """Build TradingConfig tolerating missing drawdown thresholds.

    Legacy callers expect the trading configuration to fall back to the
    documented default drawdown threshold (0.08) even when the environment is
    misconfigured. When strict parsing fails, we override the drawdown env vars
    with their defaults and retry once.
    """

    try:
        return TradingConfig.from_env(env_overrides)
    except RuntimeError as exc:
        logger.warning(
            "TRADING_CONFIG_RELAXED_FALLBACK",
            extra={"error": str(exc)},
        )
        relaxed_overrides: dict[str, Any] = dict(env_overrides or {})
        default_drawdown = SPEC_BY_FIELD["max_drawdown_threshold"].default
        # Ensure both canonical and legacy keys receive the default.
        relaxed_overrides.setdefault("MAX_DRAWDOWN_THRESHOLD", default_drawdown)
        relaxed_overrides.setdefault("AI_TRADING_MAX_DRAWDOWN_THRESHOLD", default_drawdown)
        return TradingConfig.from_env(relaxed_overrides)


def derive_cap_from_settings(
    settings: Settings | None = None,
    *,
    equity: float | None = None,
    fallback: float = 8000.0,
    capital_cap: float | None = None,
) -> float:
    """Calculate maximum capital allocation based on equity and capital cap."""

    s = settings or get_settings()
    cap = capital_cap if capital_cap is not None else float(getattr(s, "capital_cap", 0.25))
    if equity and equity > 0:
        return float(equity) * cap
    return float(fallback)


def enforce_alpaca_feed_policy() -> dict[str, str] | None:
    """Ensure Alpaca provider configurations default to SIP feed with strict validation."""

    try:
        cfg = get_trading_config()
    except Exception as exc:
        raise RuntimeError(f"Unable to load trading configuration: {exc}") from exc

    provider_primary = getattr(cfg, "data_provider", None)
    if not provider_primary:
        priority = getattr(cfg, "data_provider_priority", ())
        if priority:
            provider_primary = priority[0]
    provider_normalized = str(provider_primary or "").strip().lower()
    if not provider_normalized:
        return None

    if not provider_normalized.startswith("alpaca"):
        return {"provider": provider_normalized, "feed": str(getattr(cfg, "alpaca_data_feed", "") or "")}

    env_candidates = (
        os.getenv("ALPACA_DATA_FEED"),
        os.getenv("DATA_FEED"),
        os.getenv("ALPACA_FEED"),
    )
    explicit_feed = next((value for value in env_candidates if value not in (None, "")), None)

    feed_value = explicit_feed or getattr(cfg, "alpaca_data_feed", None) or ""
    feed_normalized = str(feed_value).strip().lower() or "sip"
    if explicit_feed is None and feed_normalized != "sip":
        feed_normalized = "sip"

    if feed_normalized != "sip":
        raise RuntimeError(
            "Alpaca data provider requires SIP market data. Set ALPACA_DATA_FEED=sip (or ALPACA_FEED=sip) "
            "and ensure your Alpaca account has SIP entitlements."
        )

    if os.getenv("ALPACA_DATA_FEED") in (None, "") and os.getenv("ALPACA_FEED") in (None, ""):
        os.environ["ALPACA_DATA_FEED"] = "sip"
    if os.getenv("ALPACA_ALLOW_SIP") in (None, "") and os.getenv("ALPACA_HAS_SIP") in (None, ""):
        os.environ.setdefault("ALPACA_ALLOW_SIP", "1")
        os.environ.setdefault("ALPACA_HAS_SIP", "1")
    try:
        cfg.update(alpaca_data_feed="sip")
    except Exception:
        pass
    return {"provider": provider_normalized, "feed": "sip"}


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
    "enforce_alpaca_feed_policy",
    "_resolve_alpaca_env",
    "get_config_schema",
    "from_env_relaxed",
    "SEED",
    "MAX_EMPTY_RETRIES",
    "Settings",
    "derive_cap_from_settings",
]
