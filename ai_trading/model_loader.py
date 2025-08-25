from __future__ import annotations

import logging
import pickle
from datetime import UTC
from pathlib import Path
from pickle import UnpicklingError

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[1]
ML_MODELS: dict[str, object | None] = {}


def train_and_save_model(symbol: str, models_dir: Path) -> object:
    """Train a fallback linear model for ``symbol`` and persist it.

    Parameters
    ----------
    symbol:
        Ticker symbol to train a model for.
    models_dir:
        Directory where the trained model should be stored.

    Returns
    -------
    object
        The trained model instance.
    """
    from datetime import datetime, timedelta

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    from ai_trading.data_fetcher import get_daily_df

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
    y = df["close"].astype(float).values
    model = LinearRegression()
    try:
        model.fit(X, y)
    except (ValueError, TypeError) as exc:
        logger.exception("Model training failed: %s", exc)
        model = LinearRegression().fit([[0], [1]], [0, 1])

    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        with open(models_dir / f"{symbol}.pkl", "wb") as f:
            pickle.dump(model, f)
    except (ValueError, TypeError) as exc:
        logger.warning("Failed saving model for %s: %s", symbol, exc)

    return model


def load_model(symbol: str) -> object:
    """Load or train an ML model for ``symbol``.

    The models directory is resolved via ``config.get_env('MODELS_DIR')`` if set,
    otherwise defaults to ``<project>/models``. If loading fails due to
    ``ValueError``, ``TypeError``, or ``UnpicklingError``, a ``RuntimeError`` is
    raised that includes the absolute path and ``symbol`` for context.
    """
    from ai_trading.config import management as config

    models_dir = Path(config.get_env("MODELS_DIR") or BASE_DIR / "models")
    path = models_dir / f"{symbol}.pkl"
    if path.exists():
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
        except (ValueError, TypeError, UnpicklingError) as exc:
            abs_path = path.resolve()
            msg = f"Failed to load model for '{symbol}' at '{abs_path}': {exc}"
            raise RuntimeError(msg) from exc
    else:
        model = train_and_save_model(symbol, models_dir)

    ML_MODELS[symbol] = model
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

