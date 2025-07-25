from __future__ import annotations

from pathlib import Path
import logging
import pickle

import config

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
_CFG_MODELS_DIR = config.get_env("MODELS_DIR")
MODELS_DIR = Path(_CFG_MODELS_DIR) if _CFG_MODELS_DIR else BASE_DIR / "models"

# AI-AGENT-REF: helper to load per-symbol ML models safely

def load_model(symbol: str):
    """Return the ML model for ``symbol`` or ``None`` when missing."""
    model_path = MODELS_DIR / f"{symbol}.pkl"
    if not model_path.exists():
        logger.warning(f"No ML model for {symbol} at {model_path}")
        return None
    with model_path.open("rb") as f:
        return pickle.load(f)
