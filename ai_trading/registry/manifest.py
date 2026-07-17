"""Model manifest validation helpers for governance artifacts."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

from ai_trading.models.contracts import normalize_bar_timeframe


_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class ManifestValidationError(ValueError):
    """Raised when manifest metadata violates required schema constraints."""


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestValidationError(f"{field} must be a mapping")
    return value


def _require_non_empty_str(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ManifestValidationError(f"{field} must be non-empty")
    return text


def _require_positive_int(value: Any, *, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError(f"{field} must be an integer") from exc
    if parsed <= 0:
        raise ManifestValidationError(f"{field} must be > 0")
    return parsed


def _require_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ManifestValidationError(f"{field} must be boolean-like")


def _require_float(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ManifestValidationError(f"{field} must be finite")
    return parsed


def _require_float_range(
    value: Any,
    *,
    field: str,
    minimum: float,
    maximum: float,
) -> float:
    parsed = _require_float(value, field=field)
    if parsed < minimum or parsed > maximum:
        raise ManifestValidationError(f"{field} must be between {minimum} and {maximum}")
    return parsed


def _require_hash(value: Any, *, field: str) -> str:
    text = _require_non_empty_str(value, field=field).lower()
    if not _HEX64_RE.fullmatch(text):
        raise ManifestValidationError(f"{field} must be a 64-char lowercase hex digest")
    return text


def validate_manifest_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize model manifest metadata payloads."""

    source = _require_mapping(payload, field="manifest_metadata")
    strategy = _require_non_empty_str(source.get("strategy"), field="strategy")
    rows = _require_positive_int(source.get("rows"), field="rows")
    lookback_days = _require_positive_int(source.get("lookback_days"), field="lookback_days")
    horizon_days = _require_positive_int(source.get("horizon_days"), field="horizon_days")
    embargo_days = _require_positive_int(source.get("embargo_days"), field="embargo_days")
    default_threshold = _require_float_range(
        source.get("default_threshold"),
        field="default_threshold",
        minimum=0.0,
        maximum=1.0,
    )
    selected_threshold = _require_float_range(
        source.get("selected_threshold", default_threshold),
        field="selected_threshold",
        minimum=0.0,
        maximum=1.0,
    )
    cost_floor_bps = _require_float_range(
        source.get("cost_floor_bps"),
        field="cost_floor_bps",
        minimum=0.0,
        maximum=10_000.0,
    )
    cost_model_version = _require_non_empty_str(
        source.get("cost_model_version"), field="cost_model_version"
    )
    feature_hash = _require_hash(source.get("feature_hash"), field="feature_hash")
    feature_contract_hash = str(source.get("feature_contract_hash") or "").strip()
    if feature_contract_hash:
        feature_contract_hash = _require_hash(
            feature_contract_hash,
            field="feature_contract_hash",
        )
    feature_contract_version = str(source.get("feature_contract_version") or "").strip()
    training_bar_timeframe = normalize_bar_timeframe(source.get("training_bar_timeframe"))
    required_bar_timeframe = normalize_bar_timeframe(source.get("required_bar_timeframe"))
    if not training_bar_timeframe:
        raise ManifestValidationError("training_bar_timeframe must be non-empty")
    if not required_bar_timeframe:
        raise ManifestValidationError("required_bar_timeframe must be non-empty")
    dataset_fingerprint = _require_hash(
        source.get("dataset_fingerprint"), field="dataset_fingerprint"
    )

    symbols_raw = source.get("symbols")
    if not isinstance(symbols_raw, (list, tuple)) or not symbols_raw:
        raise ManifestValidationError("symbols must be a non-empty sequence")
    symbols = [_require_non_empty_str(item, field="symbols[]") for item in symbols_raw]

    features_raw = source.get("feature_columns")
    if not isinstance(features_raw, (list, tuple)) or not features_raw:
        raise ManifestValidationError("feature_columns must be a non-empty sequence")
    feature_columns = [
        _require_non_empty_str(item, field="feature_columns[]") for item in features_raw
    ]

    thresholds_by_regime = _require_mapping(
        source.get("thresholds_by_regime"), field="thresholds_by_regime"
    )
    normalized_thresholds = {
        _require_non_empty_str(regime, field="thresholds_by_regime.key"): _require_float_range(
            threshold,
            field=f"thresholds_by_regime.{regime}",
            minimum=0.0,
            maximum=1.0,
        )
        for regime, threshold in thresholds_by_regime.items()
    }

    data_sources = _require_mapping(source.get("data_sources"), field="data_sources")
    normalized_sources = {
        "daily_source": _require_non_empty_str(
            data_sources.get("daily_source"), field="data_sources.daily_source"
        ),
        "minute_source": _require_non_empty_str(
            data_sources.get("minute_source"), field="data_sources.minute_source"
        ),
        "data_provenance": _require_non_empty_str(
            data_sources.get("data_provenance"), field="data_sources.data_provenance"
        ),
        "alpaca_data_feed": _require_non_empty_str(
            data_sources.get("alpaca_data_feed"), field="data_sources.alpaca_data_feed"
        ),
    }

    sensitivity = _require_mapping(source.get("sensitivity_sweep"), field="sensitivity_sweep")
    normalized_sensitivity = {
        "enabled": _require_bool(sensitivity.get("enabled"), field="sensitivity_sweep.enabled"),
        "gate": _require_bool(sensitivity.get("gate"), field="sensitivity_sweep.gate"),
        "summary": dict(
            _require_mapping(
                sensitivity.get("summary", {}),
                field="sensitivity_sweep.summary",
            )
        ),
    }

    return {
        "strategy": strategy,
        "symbols": symbols,
        "rows": rows,
        "lookback_days": lookback_days,
        "horizon_days": horizon_days,
        "embargo_days": embargo_days,
        "feature_columns": feature_columns,
        "feature_hash": feature_hash,
        "feature_contract_version": feature_contract_version,
        "feature_contract_hash": feature_contract_hash,
        "training_bar_timeframe": training_bar_timeframe,
        "required_bar_timeframe": required_bar_timeframe,
        "default_threshold": default_threshold,
        "selected_threshold": selected_threshold,
        "thresholds_by_regime": normalized_thresholds,
        "cost_floor_bps": cost_floor_bps,
        "cost_model_version": cost_model_version,
        "data_sources": normalized_sources,
        "dataset_fingerprint": dataset_fingerprint,
        "sensitivity_sweep": normalized_sensitivity,
    }
