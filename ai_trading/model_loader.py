from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

"""Model persistence utilities.

This module prefers :mod:`joblib` for serializing simple fallback models and
avoids ``pickle.load`` where possible.
"""

from ai_trading.logging import get_logger
from ai_trading.paths import MODELS_DIR
from ai_trading.config.management import get_env
from datetime import UTC
from pathlib import Path
from typing import Any
import json

import joblib
from ai_trading.utils.lazy_imports import load_pandas
from ai_trading.models.artifacts import load_verified_joblib_artifact, write_artifact_manifest

logger = get_logger(__name__)
ML_MODELS: dict[str, object | None] = {}

# Built-in models bundled with the package live alongside this module.
INTERNAL_MODELS_DIR = Path(__file__).resolve().parent / "models"


def _rolling_slope(values: Any) -> float:
    import numpy as np

    clean = np.asarray(values, dtype=float)
    if clean.size < 2:
        return 0.0
    x = np.arange(clean.size, dtype=float)
    try:
        return float(np.polyfit(x, clean, 1)[0])
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        return 0.0


def _build_training_frame(df: Any) -> Any:
    """Build past-only model features and labels from OHLCV history."""
    import numpy as np

    df = df.copy()
    df["ret1"] = df["close"].pct_change()
    df["mom5"] = df["close"].pct_change(5)
    df["mom10"] = df["close"].pct_change(10)
    df["vol20"] = df["ret1"].rolling(20).std()
    with np.errstate(invalid="ignore"):
        try:
            df["skew20"] = df["ret1"].rolling(20).skew()
            df["kurt20"] = df["ret1"].rolling(20).kurt()
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            df["skew20"] = 0.0
            df["kurt20"] = 0.0
    if "volume" in df:
        df["liq20"] = df["volume"].rolling(20).mean()
        df["volchg5"] = df["volume"].pct_change(5)
    else:
        df["liq20"] = 0.0
        df["volchg5"] = 0.0
    try:
        trend_source = df["ret1"].fillna(0.0).cumsum()
        df["trend"] = trend_source.rolling(20, min_periods=2).apply(_rolling_slope, raw=True).fillna(0.0)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        df["trend"] = 0.0

    future_close = df["close"].shift(-1)
    df = df.loc[future_close.notna()].copy()
    df["y"] = (future_close.loc[df.index] > df["close"]).astype(int)
    return df.dropna()


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
    start = end - timedelta(days=240)
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
        steps = np.arange(240, dtype=float)
        close = 100.0 + 0.02 * steps + np.sin(steps / 3.0)
        df = pd.DataFrame({"close": close, "volume": np.linspace(1e5, 2e5, 240)})
        synthetic_data_used = True

    df = _build_training_frame(df)
    if df.empty:
        raise RuntimeError(f"No labeled training rows available for {symbol}")

    def _classifier_for(target: np.ndarray) -> Any:
        if len(np.unique(target)) < 2:
            return DummyClassifier(strategy="most_frequent")
        return LogisticRegression(max_iter=500)

    feature_cols = [c for c in ("ret1", "mom5", "mom10", "vol20", "skew20", "kurt20", "liq20", "volchg5", "trend") if c in df.columns]
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
    """Load an ML model for ``symbol``.

    Two locations are searched in order:

    * The external models directory resolved from
      :data:`ai_trading.paths.MODELS_DIR` (``AI_TRADING_MODELS_DIR``).
    * Built-in models shipped with the package under ``INTERNAL_MODELS_DIR``.

    The first matching ````.pkl```` file is deserialized with :mod:`joblib`. If no
    model file exists in either location, a ``RuntimeError`` is raised. Paths are
    validated to remain within their respective base directories.
    """

    # Prefer active model from registry when available
    try:
        from ai_trading.model_registry import get_active_model_meta

        meta = get_active_model_meta(symbol)
        if meta and meta.get("path"):
            try:
                manifest_path = meta.get("manifest_path")
                model = load_verified_joblib_artifact(
                    Path(meta["path"]),
                    manifest_path=manifest_path if manifest_path else None,
                )
                ML_MODELS[symbol] = model
                return model
            except RuntimeError as exc:
                msg = f"Failed to load registry model for '{symbol}' at '{meta.get('path')}': {exc}"
                logger.error(
                    "MODEL_REGISTRY_LOAD_ERROR",
                    extra={"symbol": symbol, "path": str(meta.get("path")), "error": str(exc)},
                )
                raise RuntimeError(msg) from exc
            except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
                logger.warning("MODEL_REGISTRY_LOAD_FAILED for %s: %s", symbol, exc)
    except (AttributeError, ImportError, LookupError, OSError, TypeError, ValueError):
        pass

    dirs = (MODELS_DIR, INTERNAL_MODELS_DIR)
    for base in dirs:
        path = (base / f"{symbol}.pkl").resolve()
        if not path.is_relative_to(base):
            raise RuntimeError(f"Model path escapes models directory: {path}")
        if path.exists():
            try:
                model = load_verified_joblib_artifact(path)
            except AI_TRADING_FALLBACK_EXCEPTIONS as exc:  # noqa: BLE001 - joblib may raise various errors
                msg = f"Failed to load model for '{symbol}' at '{path}': {exc}"
                logger.error(
                    "MODEL_LOAD_ERROR",
                    extra={"symbol": symbol, "path": str(path), "error": str(exc)},
                )
                raise RuntimeError(msg) from exc
            ML_MODELS[symbol] = model
            return model

    logger.error(
        "MODEL_FILE_MISSING",
        extra={"symbol": symbol, "paths": [str(p) for p in dirs]},
    )
    test_mode = (
        bool(get_env("PYTEST_CURRENT_TEST", "", cast=str))
        or bool(get_env("PYTEST_RUNNING", False, cast=bool))
        or bool(get_env("TESTING", False, cast=bool))
    )
    allow_placeholder = "PLACEHOLDER" in str(symbol).upper()
    if test_mode and allow_placeholder:
        from ai_trading.simple_models import get_model

        model = get_model()
        ML_MODELS[symbol] = model
        logger.warning(
            "MODEL_PLACEHOLDER_USED",
            extra={"symbol": symbol, "reason": "missing_model"},
        )
        return model
    raise RuntimeError("Model required but not configured")


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
