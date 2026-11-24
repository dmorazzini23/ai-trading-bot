from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

"""Model persistence utilities.

This module prefers :mod:`joblib` for serializing simple fallback models and
avoids ``pickle.load`` where possible.
"""

from ai_trading.logging import get_logger, logger_once
from ai_trading.paths import MODELS_DIR
from ai_trading.config.management import get_env
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ai_trading.models.artifacts import load_verified_joblib_artifact

logger = get_logger(__name__)
ML_MODELS: dict[str, object | None] = {}
ML_MODEL_CACHE_META: dict[str, dict[str, str | None]] = {}

DEFAULT_MODEL_MAX_AGE_DAYS = 14


def _active_model_timestamp(meta: dict[str, Any]) -> datetime:
    """Return the governance timestamp that proves an active model is fresh."""

    nested_meta = meta.get("meta")
    candidates: list[Any] = []
    for payload in (meta, nested_meta if isinstance(nested_meta, dict) else {}):
        candidates.extend(
            payload.get(key)
            for key in (
                "trained_at",
                "training_timestamp",
                "registered_at",
                "created_at",
            )
        )
    for raw_value in candidates:
        if raw_value in (None, ""):
            continue
        text = str(raw_value).strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    raise RuntimeError("Active model registry entry is missing freshness metadata")


def _active_model_freshness_snapshot(
    meta: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return the canonical freshness calculation used by loading and health."""

    trained_at = _active_model_timestamp(meta)
    max_age_days = int(
        get_env(
            "AI_TRADING_MODEL_MAX_AGE_DAYS",
            DEFAULT_MODEL_MAX_AGE_DAYS,
            cast=int,
        )
    )
    if max_age_days <= 0:
        raise RuntimeError("AI_TRADING_MODEL_MAX_AGE_DAYS must be positive")
    observed_at = (now or datetime.now(UTC)).astimezone(UTC)
    age = max(observed_at - trained_at, timedelta(0))
    age_days = age.total_seconds() / 86400.0
    return {
        "trained_at": trained_at.isoformat(),
        "age_days": age_days,
        "max_age_days": max_age_days,
        "fresh": age <= timedelta(days=max_age_days),
    }


def _validate_active_model_freshness(symbol: str, meta: dict[str, Any]) -> None:
    freshness = _active_model_freshness_snapshot(meta)
    if not bool(freshness["fresh"]):
        logger_once.error(
            "MODEL_REGISTRY_STALE",
            extra={
                "symbol": symbol,
                "trained_at": freshness["trained_at"],
                "age_days": freshness["age_days"],
                "max_age_days": freshness["max_age_days"],
            },
            key=f"model_registry_stale:{symbol}:{freshness['trained_at']}",
        )
        raise RuntimeError(
            f"Active model for '{symbol}' is stale: trained_at={freshness['trained_at']}"
        )


def _cache_meta_from_registry(meta: dict[str, Any]) -> dict[str, str | None]:
    trained_at = _active_model_timestamp(meta).isoformat()
    return {
        "path": str(meta.get("path") or ""),
        "manifest_path": str(meta.get("manifest_path") or "") or None,
        "trained_at": trained_at,
    }


def validate_cached_model(symbol: str) -> bool:
    """Return True when the cached model still matches active fresh registry metadata."""

    from ai_trading.model_registry import get_active_model_meta

    meta = get_active_model_meta(symbol)
    if not isinstance(meta, dict) or not meta.get("path"):
        logger.error("MODEL_CACHE_REGISTRY_ACTIVE_MISSING", extra={"symbol": symbol})
        return False
    try:
        _validate_active_model_freshness(symbol, meta)
        active_cache_meta = _cache_meta_from_registry(meta)
    except RuntimeError as exc:
        logger.error(
            "MODEL_CACHE_REGISTRY_INVALID",
            extra={"symbol": symbol, "error": str(exc)},
        )
        return False
    cached_meta = ML_MODEL_CACHE_META.get(symbol)
    if cached_meta != active_cache_meta:
        logger.warning(
            "MODEL_CACHE_REGISTRY_MISMATCH",
            extra={
                "symbol": symbol,
                "cached_path": (cached_meta or {}).get("path") if cached_meta else None,
                "active_path": active_cache_meta.get("path"),
            },
        )
        return False
    return True


def load_model(symbol: str) -> object:
    """Load the governed active registry model for ``symbol``.

    Runtime loading is intentionally registry-only. Missing active entries,
    missing freshness metadata, stale timestamps, or artifact verification
    failures all fail closed instead of searching default model paths.
    """

    from ai_trading.model_registry import get_active_model_meta

    meta = get_active_model_meta(symbol)
    if not isinstance(meta, dict) or not meta.get("path"):
        logger.error("MODEL_REGISTRY_ACTIVE_MISSING", extra={"symbol": symbol})
        raise RuntimeError(f"Active registry model required for '{symbol}'")

    _validate_active_model_freshness(symbol, meta)
    manifest_path = meta.get("manifest_path")
    try:
        model = load_verified_joblib_artifact(
            Path(str(meta["path"])),
            manifest_path=str(manifest_path) if manifest_path else None,
        )
    except RuntimeError as exc:
        msg = f"Failed to load registry model for '{symbol}' at '{meta.get('path')}': {exc}"
        logger.error(
            "MODEL_REGISTRY_LOAD_ERROR",
            extra={"symbol": symbol, "path": str(meta.get("path")), "error": str(exc)},
        )
        raise RuntimeError(msg) from exc
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
        msg = f"Failed to load registry model for '{symbol}' at '{meta.get('path')}': {exc}"
        logger.error(
            "MODEL_REGISTRY_LOAD_ERROR",
            extra={"symbol": symbol, "path": str(meta.get("path")), "error": str(exc)},
        )
        raise RuntimeError(msg) from exc
    ML_MODELS[symbol] = model
    ML_MODEL_CACHE_META[symbol] = _cache_meta_from_registry(meta)
    return model


# AI-AGENT-REF: avoid import-time model loading; expose explicit preload


def get_model(symbol: str | None = None) -> object:
    """Return a model instance via :func:`load_model`.

    This provides a ``get_model`` hook so the module can be referenced via
    ``AI_TRADING_MODEL_MODULE``. When ``symbol`` is ``None``, the first entry in
    ``config.SYMBOLS`` is used, defaulting to ``"SPY"`` if no symbols are
    configured.
    """
    from ai_trading.config import management as config

    if symbol is None:
        symbols = getattr(config, "SYMBOLS", ["SPY"])
        symbol = symbols[0] if symbols else "SPY"

    return load_model(symbol)
