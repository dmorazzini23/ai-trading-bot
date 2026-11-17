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
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
import json

import joblib
from ai_trading.models.artifacts import (
    load_artifact_manifest,
    load_verified_joblib_artifact,
    verify_artifact,
    write_artifact_manifest,
)
from ai_trading.models.contracts import DAY_SLEEVE_ML_BAR_TIMEFRAME, DAY_SLEEVE_ML_FEATURE_COLUMNS, DAY_SLEEVE_ML_FEATURE_CONTRACT_VERSION, normalize_bar_timeframe, model_feature_contract_hash

logger = get_logger(__name__)
ML_MODELS: dict[str, object | None] = {}
ML_MODEL_CACHE_META: dict[str, dict[str, str | None]] = {}

DEFAULT_MODEL_MAX_AGE_DAYS = 14


@dataclass(frozen=True, slots=True)
class DaySleeveProductionModel:
    """Verified production day model and immutable canonical lineage."""

    model: object
    lineage: Mapping[str, str]
    selected_threshold: float
    thresholds_by_regime: Mapping[str, float]
    market_regime_policy: Mapping[str, Any] | None
    governance_status: str
    serving_authority: str
    input_contract: Mapping[str, Any] | None = None


_DAY_SLEEVE_MODEL_CACHE: DaySleeveProductionModel | None = None
_DAY_SLEEVE_MODEL_CACHE_KEY: tuple[Any, ...] | None = None


def _path_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return int(stat.st_mtime_ns), int(stat.st_size)


def _clear_day_sleeve_model_cache() -> None:
    global _DAY_SLEEVE_MODEL_CACHE, _DAY_SLEEVE_MODEL_CACHE_KEY
    _DAY_SLEEVE_MODEL_CACHE = None
    _DAY_SLEEVE_MODEL_CACHE_KEY = None


def load_day_sleeve_production_model(
    *,
    allow_shadow: bool = False,
) -> DaySleeveProductionModel:
    """Load the governed rich-registry production model for the five-minute sleeve.

    This path is intentionally independent of the legacy per-symbol registry.
    Missing, stale, unverifiable, or contract-incompatible production entries
    fail closed. Verified shadow models are eligible only when the caller
    explicitly grants paper-only shadow authority.
    """

    global _DAY_SLEEVE_MODEL_CACHE, _DAY_SLEEVE_MODEL_CACHE_KEY

    from ai_trading.model_registry import ModelRegistry

    registry = ModelRegistry()
    production = registry.get_viable_production_model("ml_edge")
    if production is None and allow_shadow:
        production = registry.get_viable_shadow_model("ml_edge")
    if production is None:
        _clear_day_sleeve_model_cache()
        raise RuntimeError("Governed production day-sleeve model is unavailable")
    model_id, registry_meta = production
    governance = registry_meta.get("governance")
    governance_status = (
        str(governance.get("status") or "").strip().lower()
        if isinstance(governance, Mapping)
        else ""
    )
    if governance_status != "production" and not (
        allow_shadow and governance_status == "shadow"
    ):
        _clear_day_sleeve_model_cache()
        raise RuntimeError("Day-sleeve model governance authority is invalid")
    serving_authority = (
        "production" if governance_status == "production" else "paper_only"
    )
    _validate_active_model_freshness("day_sleeve", registry_meta)

    artifact_path = Path(str(registry_meta.get("production_path") or "")).expanduser()
    manifest_raw = str(
        registry_meta.get("production_manifest_path")
        or registry_meta.get("manifest_path")
        or ""
    ).strip()
    if not artifact_path.is_file() or not manifest_raw:
        _clear_day_sleeve_model_cache()
        raise RuntimeError("Day-sleeve production artifact or manifest is missing")
    manifest_path = Path(manifest_raw).expanduser()
    if not manifest_path.is_file():
        _clear_day_sleeve_model_cache()
        raise RuntimeError("Day-sleeve production manifest is missing")

    cache_key = (
        str(model_id),
        governance_status,
        serving_authority,
        str(artifact_path.resolve()),
        str(manifest_path.resolve()),
        _path_signature(artifact_path),
        _path_signature(manifest_path),
    )
    if _DAY_SLEEVE_MODEL_CACHE is not None and _DAY_SLEEVE_MODEL_CACHE_KEY == cache_key:
        return _DAY_SLEEVE_MODEL_CACHE

    verified, reason = verify_artifact(
        model_path=artifact_path,
        manifest_path=manifest_path,
    )
    if not verified:
        _clear_day_sleeve_model_cache()
        raise RuntimeError(f"Day-sleeve artifact verification failed: {reason}")
    manifest = load_artifact_manifest(manifest_path)
    metadata = dict(manifest.metadata or {})
    expected_columns = tuple(DAY_SLEEVE_ML_FEATURE_COLUMNS)
    declared_columns = tuple(str(value) for value in metadata.get("feature_columns", ()))
    expected_hash = model_feature_contract_hash(
        expected_columns,
        bar_timeframe=DAY_SLEEVE_ML_BAR_TIMEFRAME,
        contract_version=DAY_SLEEVE_ML_FEATURE_CONTRACT_VERSION,
    )
    compatibility_errors: list[str] = []
    if normalize_bar_timeframe(metadata.get("required_bar_timeframe")) != DAY_SLEEVE_ML_BAR_TIMEFRAME:
        compatibility_errors.append("required_bar_timeframe")
    if declared_columns != expected_columns:
        compatibility_errors.append("feature_columns")
    if str(metadata.get("feature_contract_version") or "") != DAY_SLEEVE_ML_FEATURE_CONTRACT_VERSION:
        compatibility_errors.append("feature_contract_version")
    if str(metadata.get("feature_contract_hash") or "") != expected_hash:
        compatibility_errors.append("feature_contract_hash")
    if compatibility_errors:
        _clear_day_sleeve_model_cache()
        raise RuntimeError(
            "Day-sleeve model contract is incompatible: "
            + ",".join(compatibility_errors)
        )

    selected_threshold_raw = metadata.get("selected_threshold")
    if selected_threshold_raw is None:
        selected_threshold_raw = metadata.get("default_threshold")
    try:
        selected_threshold = float(selected_threshold_raw)
    except (KeyError, TypeError, ValueError) as exc:
        _clear_day_sleeve_model_cache()
        raise RuntimeError(
            "Day-sleeve model selected threshold is missing or invalid"
        ) from exc
    if not 0.0 <= selected_threshold <= 1.0:
        _clear_day_sleeve_model_cache()
        raise RuntimeError("Day-sleeve model selected threshold is outside [0, 1]")
    regime_thresholds_raw = metadata.get("thresholds_by_regime")
    if not isinstance(regime_thresholds_raw, Mapping):
        _clear_day_sleeve_model_cache()
        raise RuntimeError("Day-sleeve regime thresholds are missing")
    thresholds_by_regime: dict[str, float] = {}
    try:
        for regime, raw_threshold in regime_thresholds_raw.items():
            normalized_regime = str(regime).strip().lower()
            threshold = float(raw_threshold)
            if normalized_regime and 0.0 <= threshold <= 1.0:
                thresholds_by_regime[normalized_regime] = threshold
    except (TypeError, ValueError) as exc:
        _clear_day_sleeve_model_cache()
        raise RuntimeError("Day-sleeve regime thresholds are invalid") from exc
    if not thresholds_by_regime:
        _clear_day_sleeve_model_cache()
        raise RuntimeError("Day-sleeve regime thresholds are empty")
    market_regime_policy_raw = metadata.get("market_regime_policy")
    if market_regime_policy_raw is None:
        market_regime_policy: Mapping[str, Any] | None = None
    elif isinstance(market_regime_policy_raw, Mapping):
        market_regime_policy = MappingProxyType(dict(market_regime_policy_raw))
    else:
        market_regime_policy = MappingProxyType(
            {"invalid_declared_policy": market_regime_policy_raw}
        )

    model = load_verified_joblib_artifact(
        artifact_path,
        manifest_path=manifest_path,
    )
    if not (hasattr(model, "predict") and hasattr(model, "predict_proba")):
        _clear_day_sleeve_model_cache()
        raise RuntimeError("Day-sleeve model prediction interface is incomplete")
    model_columns_raw = getattr(model, "feature_names_in_", None)
    if model_columns_raw is not None:
        model_columns = tuple(str(value) for value in model_columns_raw)
        if model_columns != expected_columns:
            _clear_day_sleeve_model_cache()
            raise RuntimeError("Day-sleeve model feature order is incompatible")

    dataset_hash = str(
        registry_meta.get("dataset_fingerprint")
        or metadata.get("dataset_fingerprint")
        or ""
    ).strip()
    lineage_values = {
        "model_id": str(model_id).strip(),
        "model_version": str(manifest.model_version or "").strip(),
        "dataset_hash": dataset_hash,
        "feature_version": DAY_SLEEVE_ML_FEATURE_CONTRACT_VERSION,
        "model_artifact_hash": str(manifest.checksum_sha256 or "").strip(),
    }
    missing_lineage = [key for key, value in lineage_values.items() if not value]
    if missing_lineage:
        _clear_day_sleeve_model_cache()
        raise RuntimeError(
            "Day-sleeve model lineage is incomplete: " + ",".join(missing_lineage)
        )
    loaded = DaySleeveProductionModel(
        model=model,
        lineage=MappingProxyType(lineage_values),
        selected_threshold=selected_threshold,
        thresholds_by_regime=MappingProxyType(thresholds_by_regime),
        market_regime_policy=market_regime_policy,
        governance_status=governance_status,
        serving_authority=serving_authority,
        input_contract=(MappingProxyType(dict(metadata['input_contract']))
                        if isinstance(metadata.get('input_contract'), Mapping) else None),
    )
    _DAY_SLEEVE_MODEL_CACHE = loaded
    _DAY_SLEEVE_MODEL_CACHE_KEY = cache_key
    logger.info(
        "DAY_SLEEVE_ML_MODEL_SELECTED",
        extra={
            "model_id": str(model_id),
            "governance_status": governance_status,
            "serving_authority": serving_authority,
            "selected_threshold": selected_threshold,
        },
    )
    return loaded


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


def day_sleeve_model_readiness_snapshot(
    *,
    allow_shadow: bool = False,
) -> dict[str, Any]:
    """Describe the exact governed artifact authority used by day-sleeve inference."""

    from ai_trading.model_registry import ModelRegistry

    registry = ModelRegistry()
    selected = registry.get_viable_production_model("ml_edge")
    if selected is None and allow_shadow:
        selected = registry.get_viable_shadow_model("ml_edge")
    if selected is None:
        return {
            "enabled": True,
            "available": False,
            "ok": False,
            "status": "unavailable",
            "reason": "required_model_unavailable",
            "allow_shadow": bool(allow_shadow),
            "serving_decision": "abstain",
            "model_new_exposure_allowed": False,
        }

    model_id, registry_meta = selected
    governance = registry_meta.get("governance")
    governance_status = (
        str(governance.get("status") or "").strip().lower()
        if isinstance(governance, Mapping)
        else ""
    )
    serving_authority = (
        "production" if governance_status == "production" else "paper_only"
    )
    base = {
        "enabled": True,
        "available": True,
        "model_id": str(model_id),
        "governance_status": governance_status,
        "serving_authority": serving_authority,
        "allow_shadow": bool(allow_shadow),
        "serving_decision": "abstain",
        "model_new_exposure_allowed": False,
        "fallback_policy": "no_automatic_substitution_for_invalid_or_stale_selected_model",
    }
    try:
        freshness = _active_model_freshness_snapshot(registry_meta)
    except RuntimeError as exc:
        return {
            **base,
            "ok": False,
            "status": "invalid",
            "reason": "required_model_invalid",
            "error": str(exc),
        }
    if not bool(freshness["fresh"]):
        return {
            **base,
            **freshness,
            "ok": False,
            "status": "stale",
            "reason": "required_model_stale",
        }
    try:
        load_day_sleeve_production_model(allow_shadow=allow_shadow)
    except RuntimeError as exc:
        return {
            **base,
            **freshness,
            "ok": False,
            "status": "invalid",
            "reason": "required_model_invalid",
            "error": str(exc),
        }
    return {
        **base,
        **freshness,
        "ok": True,
        "status": "ready",
        "reason": "required_model_ready",
        "serving_decision": "serve_governed_model",
        "model_new_exposure_allowed": True,
    }


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
