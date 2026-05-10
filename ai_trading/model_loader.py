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
from pathlib import Path
from typing import Any
import json

import joblib
from ai_trading.utils.lazy_imports import load_pandas
from ai_trading.models.artifacts import load_verified_joblib_artifact, write_artifact_manifest
from ai_trading.models.contracts import (
    AFTER_HOURS_ML_BAR_TIMEFRAME,
    LIVE_ML_FEATURE_COLUMNS,
    MODEL_FEATURE_CONTRACT_VERSION,
    model_feature_contract_hash,
)

logger = get_logger(__name__)
ML_MODELS: dict[str, object | None] = {}
ML_MODEL_CACHE_META: dict[str, dict[str, str | None]] = {}

DEFAULT_MODEL_MAX_AGE_DAYS = 14


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
