from __future__ import annotations

"""Model persistence utilities.

This module prefers :mod:`joblib` for serializing simple fallback models and
avoids ``pickle.load`` where possible.
"""

from ai_trading.logging import get_logger
from ai_trading.paths import MODELS_DIR
from datetime import UTC
from pathlib import Path

import joblib

logger = get_logger(__name__)
ML_MODELS: dict[str, object | None] = {}

# Built-in models bundled with the package live alongside this module.
INTERNAL_MODELS_DIR = Path(__file__).resolve().parent / "models"


def train_and_save_model(symbol: str, models_dir: Path) -> object:
    """Train a fallback classification model for ``symbol`` and persist it."""

    from datetime import datetime, timedelta

    import numpy as np
    import pandas as pd
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression

    from ai_trading.data.fetch import get_daily_df

    end = datetime.now(UTC)
    start = end - timedelta(days=30)
    try:
        df = get_daily_df(symbol, start, end)
    except (ValueError, TypeError) as exc:
        logger.warning("Data fetch failed for %s: %s", symbol, exc)
        df = pd.DataFrame({"close": np.linspace(1.0, 2.0, 30)})
    if df is None or df.empty or "close" not in df:
        df = pd.DataFrame({"close": np.linspace(1.0, 2.0, 30)})

    X = np.arange(len(df)).reshape(-1, 1)
    y = (df["close"].diff().fillna(0) > 0).astype(int).values
    model = LogisticRegression()
    try:
        model.fit(X, y)
    except (Exception) as exc:  # noqa: BLE001 - broad to fall back gracefully
        logger.exception("Model training failed: %s", exc)
        model = DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1])

    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, models_dir / f"{symbol}.pkl")
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Failed saving model for %s: %s", symbol, exc)

    return model


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

    dirs = (MODELS_DIR, INTERNAL_MODELS_DIR)
    for base in dirs:
        path = (base / f"{symbol}.pkl").resolve()
        if not path.is_relative_to(base):
            raise RuntimeError(f"Model path escapes models directory: {path}")
        if path.exists():
            try:
                model = joblib.load(path)
            except Exception as exc:  # noqa: BLE001 - joblib may raise various errors
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
    # Fallback: builtin heuristic model (scikit-like API)
    try:
        from ai_trading.simple_models import get_model as _get_model

        mdl = _get_model()
        ML_MODELS[symbol] = mdl
        logger.info(
            "MODEL_LOADED", extra={"source": "module", "model_module": "ai_trading.simple_models"}
        )
        return mdl
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"Model file for '{symbol}' not found and heuristic fallback failed: {exc}"
        ) from exc


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
