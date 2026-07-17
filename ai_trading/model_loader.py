from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

"""Model persistence utilities.

This module prefers :mod:`joblib` for serializing simple fallback models and
avoids ``pickle.load`` where possible.
"""

from ai_trading.logging import get_logger
from ai_trading.paths import MODELS_DIR
from ai_trading.config.management import get_env
from datetime import UTC, datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
import json

import joblib
from ai_trading.utils.lazy_imports import load_pandas
from ai_trading.models.artifacts import (
    load_artifact_manifest,
    load_verified_joblib_artifact,
    verify_artifact,
    write_artifact_manifest,
)
from ai_trading.models.contracts import (
    AFTER_HOURS_ML_BAR_TIMEFRAME,
    DAY_SLEEVE_ML_BAR_TIMEFRAME,
    DAY_SLEEVE_ML_FEATURE_COLUMNS,
    DAY_SLEEVE_ML_FEATURE_CONTRACT_VERSION,
    LIVE_ML_FEATURE_COLUMNS,
    MODEL_FEATURE_CONTRACT_VERSION,
    normalize_bar_timeframe,
    model_feature_contract_hash,
)

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
    governance_status: str
    serving_authority: str


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
        governance_status=governance_status,
        serving_authority=serving_authority,
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


def _build_training_frame(df: Any) -> Any:
    """Build past-only model features and labels from OHLCV history."""
    import numpy as np

    from ai_trading.features.indicators import compute_atr, compute_macd, compute_sma, compute_vwap
    from ai_trading.indicators import rsi as rsi_indicator

    df = df.copy()
    df.columns = [str(column).lower() for column in df.columns]
    if "close" not in df:
        return df.iloc[0:0]
    close = df["close"].astype(float)
    if "open" not in df:
        df["open"] = close.shift(1).fillna(close)
    if "high" not in df:
        df["high"] = close.rolling(3, min_periods=1).max()
    if "low" not in df:
        df["low"] = close.rolling(3, min_periods=1).min()
    if "volume" not in df:
        df["volume"] = 1.0

    df = compute_macd(df)
    df = compute_atr(df)
    df = compute_vwap(df)
    df = compute_sma(df, windows=(50, 200))
    rsi_values = rsi_indicator(tuple(close), 14)
    if len(rsi_values) == len(df):
        df["rsi"] = np.asarray(rsi_values, dtype=float)
    else:
        df["rsi"] = np.nan
        try:
            df.loc[df.index[-len(rsi_values):], "rsi"] = np.asarray(rsi_values, dtype=float)
        except (ValueError, TypeError):
            df["rsi"] = np.nan

    future_close = close.shift(-1)
    df = df.loc[future_close.notna()].copy()
    df["y"] = (future_close.loc[df.index] > df["close"]).astype(int)
    return df.dropna(subset=list(LIVE_ML_FEATURE_COLUMNS) + ["y"])


def _drop_next_bar_boundary_overlap(train_idx: Any, test_idx: Any) -> Any:
    import numpy as np

    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    if len(train_idx) == 0 or len(test_idx) == 0:
        return train_idx
    first_test = int(test_idx[0])
    return train_idx[train_idx + 1 < first_test]


def _next_bar_safe_time_series_splits(X: Any, *, n_splits: int = 5) -> Any:
    from sklearn.model_selection import TimeSeriesSplit

    try:
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=1)
    except TypeError:
        tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tscv.split(X):
        yield (_drop_next_bar_boundary_overlap(train_idx, test_idx), test_idx)


def _synthetic_training_allowed() -> bool:
    return (
        bool(get_env("PYTEST_CURRENT_TEST", "", cast=str))
        or bool(get_env("PYTEST_RUNNING", False, cast=bool))
        or bool(get_env("TESTING", False, cast=bool))
        or bool(get_env("AI_TRADING_MODEL_TRAINING_SMOKE", False, cast=bool))
    )


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


def _validate_active_model_freshness(symbol: str, meta: dict[str, Any]) -> None:
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
    age = datetime.now(UTC) - trained_at
    if age > timedelta(days=max_age_days):
        logger.error(
            "MODEL_REGISTRY_STALE",
            extra={
                "symbol": symbol,
                "trained_at": trained_at.isoformat(),
                "max_age_days": max_age_days,
            },
        )
        raise RuntimeError(
            f"Active model for '{symbol}' is stale: trained_at={trained_at.isoformat()}"
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


def train_and_save_model(symbol: str, models_dir: Path) -> object:
    """Train a simple feature/label pipeline with rolling OOS validation and persist it.

    - Engineers basic features (momentum, volatility, skew/kurtosis, liquidity).
    - Labels next-period direction.
    - Uses ``TimeSeriesSplit`` for OOS scoring and fits LogisticRegression.
    - Persists model and lightweight metadata for governance.
    """

    from datetime import datetime, timedelta

    import numpy as np
    pd = load_pandas()
    if pd is None or not hasattr(pd, "DataFrame"):
        raise ImportError("pandas is required for train_and_save_model")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score
    from sklearn.dummy import DummyClassifier

    from ai_trading.data.fetch import get_daily_df

    end = datetime.now(UTC)
    start = end - timedelta(days=420)
    try:
        df = get_daily_df(symbol, start, end)
    except (ValueError, TypeError) as exc:
        logger.warning("Data fetch failed for %s: %s", symbol, exc)
        df = None
    synthetic_data_used = False
    if df is None or df.empty or "close" not in df:
        if not _synthetic_training_allowed():
            raise RuntimeError(f"Real training bars unavailable for {symbol}")
        # Fallback synthetic series with both up and down labels for tests/smoke only.
        steps = np.arange(420, dtype=float)
        close = 100.0 + 0.02 * steps + np.sin(steps / 3.0)
        df = pd.DataFrame({"close": close, "volume": np.linspace(1e5, 2e5, 420)})
        synthetic_data_used = True

    df = _build_training_frame(df)
    if df.empty:
        raise RuntimeError(f"No labeled training rows available for {symbol}")

    def _classifier_for(target: np.ndarray) -> Any:
        if len(np.unique(target)) < 2:
            return DummyClassifier(strategy="most_frequent")
        return LogisticRegression(max_iter=500)

    feature_cols = [c for c in LIVE_ML_FEATURE_COLUMNS if c in df.columns]
    X = df[feature_cols].astype(float)
    y = df["y"].astype(int).values
    if len(X) < 60:
        raise RuntimeError(
            f"Insufficient labeled training rows for {symbol}: {len(X)} < 60"
        )

    # Walk-forward OOS validation
    scores: list[float] = []
    for train_idx, test_idx in _next_bar_safe_time_series_splits(X, n_splits=5):
        if len(test_idx) == 0 or len(train_idx) < 20:
            continue
        pipe = make_pipeline(StandardScaler(with_mean=True), _classifier_for(y[train_idx]))
        pipe.fit(X.iloc[train_idx], y[train_idx])
        yhat = pipe.predict(X.iloc[test_idx])
        try:
            scores.append(accuracy_score(y[test_idx], yhat))
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            continue

    # Final fit on all but last 5 samples to reduce leakage
    cutoff = max(0, len(X) - 5)
    if cutoff <= 0:
        cutoff = len(X)
    final_pipe = make_pipeline(StandardScaler(with_mean=True), _classifier_for(y[:cutoff]))
    final_pipe.fit(X.iloc[:cutoff], y[:cutoff])
    contract_hash = model_feature_contract_hash(
        feature_cols,
        bar_timeframe=AFTER_HOURS_ML_BAR_TIMEFRAME,
    )
    setattr(final_pipe, "required_bar_timeframe_", AFTER_HOURS_ML_BAR_TIMEFRAME)
    setattr(final_pipe, "training_bar_timeframe_", AFTER_HOURS_ML_BAR_TIMEFRAME)
    setattr(final_pipe, "feature_contract_version_", MODEL_FEATURE_CONTRACT_VERSION)
    setattr(final_pipe, "feature_contract_hash_", contract_hash)

    # Persist model and metadata
    if synthetic_data_used:
        return final_pipe
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{symbol}.pkl"
        joblib.dump(final_pipe, model_path)
        meta = {
            "version": "1.0",
            "model": "logreg",
            "features": feature_cols,
            "feature_columns": feature_cols,
            "feature_contract_version": MODEL_FEATURE_CONTRACT_VERSION,
            "feature_contract_hash": contract_hash,
            "training_bar_timeframe": AFTER_HOURS_ML_BAR_TIMEFRAME,
            "required_bar_timeframe": AFTER_HOURS_ML_BAR_TIMEFRAME,
            "oos_accuracy_mean": float(sum(scores) / max(1, len(scores))),
            "n_samples": int(len(df)),
            "trained_at": datetime.now(UTC).isoformat(),
        }
        write_artifact_manifest(
            model_path=model_path,
            model_version=f"{symbol}-1.0",
            training_data_range={"start": start.isoformat(), "end": end.isoformat()},
            metadata=meta,
        )
        with (models_dir / f"{symbol}.meta.json").open("w") as f:
            json.dump(meta, f)
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Failed saving model for %s: %s", symbol, exc)

    return final_pipe


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
def preload_models(symbols: list[str] | None = None) -> None:
    """Eagerly load models for ``symbols``.

    If ``symbols`` is ``None``, falls back to ``config.SYMBOLS``. This function
    imports configuration lazily to keep startup lean.
    """
    from ai_trading.config import management as config

    for sym in symbols or getattr(config, "SYMBOLS", []):
        ML_MODELS[sym] = load_model(sym)


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
