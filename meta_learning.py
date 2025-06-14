import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


def load_weights(path: str, default: np.ndarray | None = None) -> np.ndarray:
    """Load signal weights array from ``path`` or return ``default``."""
    p = Path(path)
    if default is None:
        default = np.zeros(0)
    if not p.exists():
        logger.error("Signal weight file missing: %s", path)
        if default.size > 0:
            try:
                np.savetxt(p, default, delimiter=",")
            except Exception as exc:
                logger.exception("Failed initializing default weights: %s", exc)
        return default
    try:
        weights = np.loadtxt(p, delimiter=",")
        return weights
    except Exception as exc:
        logger.exception("Failed to load signal weights: %s", exc)
        return default


def update_weights(weight_path: str, new_weights: np.ndarray, metrics: dict, history_file: str = "metrics.json", n_history: int = 5) -> bool:
    """Update signal weights if changed and persist metric history."""
    p = Path(weight_path)
    prev = None
    try:
        if p.exists():
            prev = np.loadtxt(p, delimiter=",")
            if np.allclose(prev, new_weights):
                logger.info("META_WEIGHTS_UNCHANGED")
                return False
        np.savetxt(p, new_weights, delimiter=",")
        logger.info(
            "META_WEIGHTS_UPDATED",
            extra={"previous": prev, "current": new_weights.tolist()},
        )
    except Exception as exc:
        logger.exception(f"META_WEIGHT_UPDATE_FAILED: {exc}")
        return False
    try:
        if Path(history_file).exists():
            with open(history_file) as f:
                hist = json.load(f)
        else:
            hist = []
    except Exception as e:
        logger.error("Failed to read metric history: %s", e)
        hist = []
    hist.append({"ts": datetime.now(timezone.utc).isoformat(), **metrics})
    hist = hist[-n_history:]
    with open(history_file, "w") as f:
        json.dump(hist, f)
    logger.info("META_METRICS", extra={"recent": hist})
    return True
