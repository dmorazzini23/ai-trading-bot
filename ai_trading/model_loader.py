from __future__ import annotations

from pathlib import Path
import logging
import pickle

import config

logger = logging.getLogger(__name__)

# AI-AGENT-REF: resolve repo root reliably for model loading
BASE_DIR = Path(__file__).resolve().parents[1]
_CFG_MODELS_DIR = config.get_env("MODELS_DIR")
MODELS_DIR = Path(_CFG_MODELS_DIR) if _CFG_MODELS_DIR else BASE_DIR / "models"

ML_MODELS: dict[str, object | None] = {}
for sym in getattr(config, "SYMBOLS", []):
    path = MODELS_DIR / f"{sym}.pkl"
    if not path.exists():
        logger.warning(f"No ML model for {sym} at {path}")
        ML_MODELS[sym] = None
    else:
        with path.open("rb") as f:
            ML_MODELS[sym] = pickle.load(f)


def load_model(symbol: str):
    """Return the preloaded ML model for ``symbol``."""
    return ML_MODELS.get(symbol)
