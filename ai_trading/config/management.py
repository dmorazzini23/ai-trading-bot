from __future__ import annotations

import os
from threading import RLock
from typing import Any, Callable, Iterable, Mapping, Optional, TypeVar
from urllib.parse import urlparse

from pathlib import Path

try:
    from dotenv import load_dotenv as _load_dotenv
except Exception:  # pragma: no cover - optional dependency may be absent
    _load_dotenv = None

_DOTENV_WARNING_EMITTED = False
_RUNTIME_ENV_LOCK = RLock()
_RUNTIME_ENV_OVERRIDES: dict[str, str] = {}

from ai_trading.logging import logger
from .settings import Settings, get_settings
from .runtime import (
    CONFIG_SPECS,
    SPEC_BY_ENV,
    SPEC_BY_FIELD,
    TradingConfig,
    config_snapshot_hash,
    ensure_trading_config_current,
    generate_config_schema,
    get_trading_config,
    reload_trading_config,
)

TESTING = str(os.getenv("TESTING", "")).strip().lower() in {"1", "true", "yes", "on"}

T = TypeVar("T")

ALPACA_URL_GUIDANCE = (
    "Set ALPACA_TRADING_BASE_URL for trading endpoints. "
    "Deprecated keys ALPACA_API_URL and ALPACA_BASE_URL are not supported."
)

_CANONICAL_ENV_MAP: dict[str, tuple[str, ...]] = {
    "ALPACA_TRADING_BASE_URL": ("ALPACA_API_URL", "ALPACA_BASE_URL"),
    "ALPACA_DATA_BASE_URL": ("ALPACA_DATA_URL",),
    "AI_TRADING_TRADING_MODE": ("TRADING_MODE",),
    "AI_TRADING_CAPITAL_CAP": ("CAPITAL_CAP",),
    "AI_TRADING_DAILY_LOSS_LIMIT": ("DAILY_LOSS_LIMIT",),
    "AI_TRADING_SIGNAL_MAX_POSITION_SIZE": ("MAX_POSITION_SIZE", "AI_TRADING_MAX_POSITION_SIZE"),
    "AI_TRADING_TAKE_PROFIT_FACTOR": ("TAKE_PROFIT_FACTOR",),
    "AI_TRADING_BUY_THRESHOLD": ("BUY_THRESHOLD",),
    "AI_TRADING_CONF_THRESHOLD": ("CONF_THRESHOLD",),
    "AI_TRADING_MIN_CONFIDENCE": ("MIN_CONFIDENCE",),
    "AI_TRADING_KELLY_FRACTION": ("KELLY_FRACTION",),
    "AI_TRADING_KELLY_FRACTION_MAX": ("KELLY_FRACTION_MAX",),
    "AI_TRADING_SIGNAL_CONFIRMATION_BARS": ("SIGNAL_CONFIRMATION_BARS",),
    "MAX_DRAWDOWN_THRESHOLD": ("AI_TRADING_MAX_DRAWDOWN_THRESHOLD",),
    "TRADING__ALLOW_SHORTS": ("AI_TRADING_ALLOW_SHORT",),
    "EXECUTION_ALLOW_FALLBACK_WITHOUT_NBBO": ("AI_TRADING_EXEC_ALLOW_FALLBACK_WITHOUT_NBBO",),
    "SENTIMENT_API_KEY": ("NEWS_API_KEY",),
}


def canonical_env_map() -> Mapping[str, tuple[str, ...]]:
    """Return canonical key -> deprecated keys mapping for diagnostics."""

    return dict(_CANONICAL_ENV_MAP)


def _deprecated_env_violations(
    env: Mapping[str, str] | None = None,
) -> list[tuple[str, str, str]]:
    """Return deprecated env key violations from *env* snapshot."""

    env_map = _merged_env_snapshot(env)
    violations: list[tuple[str, str, str]] = []
    for canonical, deprecated_keys in _CANONICAL_ENV_MAP.items():
        for deprecated in deprecated_keys:
            raw = env_map.get(deprecated)
            if raw in (None, ""):
                continue
            if canonical == "ALPACA_TRADING_BASE_URL":
                message = (
                    "Set ALPACA_TRADING_BASE_URL for trading endpoints; "
                    "remove ALPACA_API_URL/ALPACA_BASE_URL."
                )
            elif canonical == "ALPACA_DATA_BASE_URL":
                message = "Set ALPACA_DATA_BASE_URL and remove ALPACA_DATA_URL."
            else:
                message = f"Use {canonical} instead of {deprecated}."
            violations.append((deprecated, canonical, message))
    return violations


def validate_no_deprecated_env(env: Mapping[str, str] | None = None) -> None:
    """Fail fast when deprecated env aliases are present."""

    violations = _deprecated_env_violations(env=env)
    if not violations:
        return
    details = "; ".join(
        f"{deprecated} is deprecated. {message}"
        for deprecated, _canonical, message in violations
    )
    raise RuntimeError(
        "Deprecated environment keys are not supported. "
        f"{details}"
    )


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
            f"{ALPACA_URL_GUIDANCE}"
        )

    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None, (
            f"{source_key} must include an HTTP scheme (got {raw}). "
            "Provide a complete Alpaca REST endpoint such as https://paper-api.alpaca.markets."
        )

    return raw, None


def _normalise_runtime_env_key(key: str) -> str:
    return str(key or "").strip().upper()


def _runtime_env_lookup(key: str) -> str | None:
    env_key = _normalise_runtime_env_key(key)
    if not env_key:
        return None
    with _RUNTIME_ENV_LOCK:
        return _RUNTIME_ENV_OVERRIDES.get(env_key)


def _runtime_env_overrides_snapshot() -> dict[str, str]:
    with _RUNTIME_ENV_LOCK:
        return dict(_RUNTIME_ENV_OVERRIDES)


def _env_with_runtime_fallback(key: str) -> str | None:
    raw_env = os.environ.get(key)
    if raw_env not in (None, ""):
        return raw_env
    runtime_val = _runtime_env_lookup(key)
    if runtime_val in (None, ""):
        return raw_env
    return runtime_val


def _merged_env_snapshot(env: Mapping[str, str] | None = None) -> dict[str, str]:
    if env is None:
        snapshot = dict(os.environ)
    else:
        snapshot = {
            str(key): str(value)
            for key, value in env.items()
            if value is not None
        }
    for key, value in _runtime_env_overrides_snapshot().items():
        if snapshot.get(key) in (None, ""):
            snapshot[key] = value
    return snapshot


def set_runtime_env_override(key: str, value: Any) -> None:
    """Set an in-process environment override without mutating ``os.environ``."""

    env_key = _normalise_runtime_env_key(key)
    if not env_key:
        return
    with _RUNTIME_ENV_LOCK:
        _RUNTIME_ENV_OVERRIDES[env_key] = str(value)


def clear_runtime_env_override(key: str) -> None:
    """Remove a previously configured in-process environment override."""

    env_key = _normalise_runtime_env_key(key)
    if not env_key:
        return
    with _RUNTIME_ENV_LOCK:
        _RUNTIME_ENV_OVERRIDES.pop(env_key, None)


def clear_runtime_env_overrides(keys: Iterable[str] | None = None) -> None:
    """Clear one or all in-process environment overrides."""

    with _RUNTIME_ENV_LOCK:
        if keys is None:
            _RUNTIME_ENV_OVERRIDES.clear()
            return
        for key in keys:
            env_key = _normalise_runtime_env_key(key)
            if env_key:
                _RUNTIME_ENV_OVERRIDES.pop(env_key, None)


def _select_alpaca_base_url(
    env: Mapping[str, str] | None = None,
) -> tuple[str | None, str | None, list[tuple[str, str, str]]]:
    env_map = _merged_env_snapshot(env)
    invalid_entries: list[tuple[str, str, str]] = []

    raw = env_map.get("ALPACA_TRADING_BASE_URL")
    normalized, message = _normalize_alpaca_base_url(
        raw,
        source_key="ALPACA_TRADING_BASE_URL",
    )
    if normalized:
        return normalized, "ALPACA_TRADING_BASE_URL", invalid_entries
    if raw and message:
        invalid_entries.append(("ALPACA_TRADING_BASE_URL", raw, message))
    for deprecated_key in ("ALPACA_API_URL", "ALPACA_BASE_URL"):
        deprecated_value = env_map.get(deprecated_key)
        if deprecated_value not in (None, ""):
            invalid_entries.append(
                (
                    deprecated_key,
                    deprecated_value,
                    "Set ALPACA_TRADING_BASE_URL for trading endpoints; "
                    "remove ALPACA_API_URL/ALPACA_BASE_URL.",
                )
            )

    return None, None, invalid_entries


def reload_env(path: str | os.PathLike[str] | None = None, override: bool = True) -> str | None:
    """Reload environment variables from a dotenv file."""

    from ai_trading.utils.env import refresh_alpaca_credentials_cache

    def _invalidate_settings_caches() -> None:
        cache_clear = getattr(get_settings, "cache_clear", None)
        if callable(cache_clear):
            cache_clear()
        try:
            import ai_trading.config as config_pkg

            reset_cached = getattr(config_pkg, "_reset_cached_settings", None)
            if callable(reset_cached):
                reset_cached()
        except Exception:
            # Best-effort cache invalidation; runtime config reload still proceeds.
            pass

    skip_dotenv = bool(os.getenv("PYTEST_RUNNING") or os.getenv("TESTING"))

    if path is None:
        if skip_dotenv:
            path = None
        else:
            candidate = Path.cwd() / ".env"
            path = candidate if candidate.exists() else None
    if path is None:
        _invalidate_settings_caches()
        reload_trading_config()
        refresh_alpaca_credentials_cache()
        return None
    _maybe_load_dotenv(path, override=override)
    _invalidate_settings_caches()
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


def _get_env_exact(
    key: str,
    default: Optional[str] = None,
    *,
    cast: Optional[Callable[[Any], T]] = None,
    required: bool = False,
) -> T | Any:
    raw = _env_with_runtime_fallback(key)
    if raw is None:
        if required:
            raise RuntimeError(f"Missing required environment variable: {key}")
        return default
    return _coerce(raw, cast)


def get_env(
    key: str,
    default: Optional[str] = None,
    *,
    cast: Optional[Callable[[Any], T]] = None,
    required: bool = False,
    resolve_aliases: bool = True,
) -> T | Any:
    """Resolve config values from trading config, then runtime/env overrides."""

    if not resolve_aliases:
        return _get_env_exact(key, default, cast=cast, required=required)

    spec = SPEC_BY_ENV.get(key.upper())
    if spec is None:
        return _get_env_exact(key, default, cast=cast, required=required)

    for candidate in (*spec.env, *tuple(spec.deprecated_env.keys())):
        candidate_value = _env_with_runtime_fallback(candidate)
        if candidate_value not in (None, ""):
            return _coerce(candidate_value, cast)

    env_keys = tuple(
        dict.fromkeys(
            (
                *tuple(spec.env),
                *tuple(spec.deprecated_env.keys()),
                "AI_TRADING_TRADING_MODE",
                "AI_TRADING_TRADING_MODE_PRECEDENCE",
            )
        )
    )
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

    env_snapshot: dict[str, str] = _merged_env_snapshot(env)
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
    validate_no_deprecated_env(env_snapshot)

    cfg = TradingConfig.from_env(
        config_overrides, allow_missing_drawdown=True
    )

    required_fields = {
        "ALPACA_API_KEY": cfg.alpaca_api_key,
        "ALPACA_SECRET_KEY": cfg.alpaca_secret_key,
        "ALPACA_DATA_FEED": cfg.alpaca_data_feed,
        "ALPACA_TRADING_BASE_URL": cfg.alpaca_base_url,
        "WEBHOOK_SECRET": cfg.webhook_secret,
        "AI_TRADING_CAPITAL_CAP": cfg.capital_cap,
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

    for key, value in list(required_fields.items()):
        if value in (None, ""):
            fallback = env_lookup.get(key)
            if fallback not in (None, ""):
                required_fields[key] = fallback
                continue

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
    base_url, _source, errors = _select_alpaca_base_url()
    for env_key, _raw, _message in errors:
        if env_key in {"ALPACA_API_URL", "ALPACA_BASE_URL"}:
            raise RuntimeError(
                f"{env_key} is deprecated. Set ALPACA_TRADING_BASE_URL instead."
            )

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
                "ALPACA_TRADING_BASE_URL": cfg.alpaca_base_url or "",
            }
        )
        errors.extend(cfg_errors)
        if cfg_base_url:
            base_url = cfg_base_url
            _ = cfg_source

    for env_key, raw, message in errors:
        logger.error(message, extra={"env_key": env_key, "value": raw})

    if not base_url and cfg is not None:
        normalized_cfg, cfg_error = _normalize_alpaca_base_url(
            cfg.alpaca_base_url, source_key="ALPACA_TRADING_BASE_URL"
        )
        if normalized_cfg:
            base_url = normalized_cfg
        elif cfg_error:
            logger.error(
                cfg_error,
                extra={"env_key": "ALPACA_TRADING_BASE_URL", "value": cfg.alpaca_base_url},
            )

    resolved_base_url = base_url or default_base_url

    api_key = (
        (getattr(cfg, "alpaca_api_key", None) if cfg is not None else None)
        or _env_with_runtime_fallback("ALPACA_API_KEY")
    )
    secret = (
        (getattr(cfg, "alpaca_secret_key", None) if cfg is not None else None)
        or _env_with_runtime_fallback("ALPACA_SECRET_KEY")
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
    validate_no_deprecated_env()
    _, _, url_errors = _select_alpaca_base_url()
    if url_errors:
        messages = "; ".join(error for _, _, error in url_errors)
        raise RuntimeError(messages)
    try:
        validate_required_env(
            ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_TRADING_BASE_URL")
        )
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
        # Ensure the canonical key receives the default.
        relaxed_overrides.setdefault("MAX_DRAWDOWN_THRESHOLD", default_drawdown)
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
    """Honor explicit Alpaca feed selection; allow IEX without fallback.

    Behavior:
    - Non‑Alpaca provider => return a non_alpaca status (no changes).
    - Alpaca + sip => keep SIP and set convenience env defaults.
    - Alpaca + iex => accept IEX (no fallback away from Alpaca). Ensure env is
      aligned but do not alter provider priority.
    - No explicit feed => default remains SIP here (prefers SIP when using
      Alpaca), while Settings defaults ALPACA_DATA_FEED to "iex". Precedence
      order: ALPACA_DATA_FEED/DATA_FEED/ALPACA_FEED env, then cfg.alpaca_data_feed.
    """

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

    # If not using Alpaca at all, just report context for logging.
    if not provider_normalized.startswith("alpaca"):
        return {
            "provider": provider_normalized,
            "feed": str(getattr(cfg, "alpaca_data_feed", "") or ""),
            "status": "non_alpaca",
        }

    # Determine requested feed, honoring any explicit environment override.
    env_candidates = (
        os.getenv("ALPACA_DATA_FEED"),
        os.getenv("DATA_FEED"),
        os.getenv("ALPACA_FEED"),
    )
    explicit_feed = next((value for value in env_candidates if value not in (None, "")), None)
    feed_value = explicit_feed or getattr(cfg, "alpaca_data_feed", None) or ""
    feed_normalized = str(feed_value).strip().lower()
    if not feed_normalized:
        # Historically enforced SIP by default under Alpaca; preserve that here.
        feed_normalized = "sip"

    # Allow Alpaca + IEX without fallback.
    if feed_normalized == "iex":
        if os.getenv("ALPACA_DATA_FEED") in (None, "") and os.getenv("ALPACA_FEED") in (None, ""):
            set_runtime_env_override("ALPACA_DATA_FEED", "iex")
        # No provider switch; just report status so preflight logs at INFO.
        return {
            "provider": provider_normalized,
            "feed": "iex",
            "status": "alpaca_iex",
        }

    # SIP path: retain existing behavior and set helpful env defaults.
    if feed_normalized == "sip":
        if os.getenv("ALPACA_DATA_FEED") in (None, "") and os.getenv("ALPACA_FEED") in (None, ""):
            set_runtime_env_override("ALPACA_DATA_FEED", "sip")
        if os.getenv("ALPACA_ALLOW_SIP") in (None, "") and os.getenv("ALPACA_HAS_SIP") in (None, ""):
            set_runtime_env_override("ALPACA_ALLOW_SIP", "1")
        return {"provider": provider_normalized, "feed": "sip", "status": "sip"}

    # Unknown value: do nothing but surface context.
    return {
        "provider": provider_normalized,
        "feed": feed_normalized,
        "status": "alpaca_unknown_feed",
    }


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
    "config_snapshot_hash",
    "set_runtime_env_override",
    "clear_runtime_env_override",
    "clear_runtime_env_overrides",
]
