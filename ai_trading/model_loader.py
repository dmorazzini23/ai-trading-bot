from __future__ import annotations

from pathlib import Path
import logging
import pickle

from ai_trading.config import management as config

logger = logging.getLogger(__name__)

# AI-AGENT-REF: resolve repo root reliably for model loading
BASE_DIR = Path(__file__).resolve().parents[1]
_CFG_MODELS_DIR = config.get_env("MODELS_DIR")
MODELS_DIR = Path(_CFG_MODELS_DIR) if _CFG_MODELS_DIR else BASE_DIR / "models"

ML_MODELS: dict[str, object | None] = {}


def train_and_save_model(symbol: str):
    """Train a fallback linear model and persist it."""
    from sklearn.linear_model import LinearRegression
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta, timezone
    from ai_trading.data_fetcher import get_daily_df

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30)
    try:
        df = get_daily_df(symbol, start, end)
    except Exception as exc:  # pragma: no cover - network may fail
        logger.warning("Data fetch failed for %s: %s", symbol, exc)
        df = pd.DataFrame({"close": np.linspace(1.0, 2.0, 30)})

    if df is None or df.empty or "close" not in df:
        df = pd.DataFrame({"close": np.linspace(1.0, 2.0, 30)})

    X = np.arange(len(df)).reshape(-1, 1)
    y = df["close"].astype(float).values
    model = LinearRegression()
    try:
        model.fit(X, y)
    except Exception as exc:  # pragma: no cover - unexpected sklearn failure
        logger.exception("Model training failed: %s", exc)
        model = LinearRegression().fit([[0], [1]], [0, 1])

    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODELS_DIR / f"{symbol}.pkl", "wb") as f:
            pickle.dump(model, f)
    except Exception as exc:  # pragma: no cover - disk issues
        logger.warning("Failed saving model for %s: %s", symbol, exc)

    return model


def load_model(symbol: str):
    """Load or train an ML model for ``symbol``."""
    path = MODELS_DIR / f"{symbol}.pkl"
    model = None
    if path.exists():
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
        except Exception as exc:
            logger.warning("Model load failed for %s: %s", symbol, exc)
            model = None
    if model is None:
        model = train_and_save_model(symbol)
    ML_MODELS[symbol] = model
    return model


# AI-AGENT-REF: Defer model loading in testing environments to prevent import blocking
import os
import sys
# AI-AGENT-REF: More aggressive testing mode detection to prevent import hangs
_is_testing = (
    os.getenv("TESTING") 
    or os.getenv("PYTEST_RUNNING")
    or getattr(config, "TESTING", False) 
    or "pytest" in sys.modules 
    or "test_" in os.path.basename(sys.argv[0] if sys.argv else "")
)
if not _is_testing:
    for sym in getattr(config, "SYMBOLS", []):
        ML_MODELS[sym] = load_model(sym)
