import json
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_weights(weight_path: str) -> dict[str, float]:
    """Load signal weights from ``weight_path`` returning a mapping."""
    p = Path(weight_path)
    if not p.exists():
        logger.error("Weight file %s missing; creating default", weight_path)
        try:
            p.write_text("signal,weight\n")
        except Exception as exc:
            logger.exception("Failed creating default weight file: %s", exc)
        return {}
    try:
        df = pd.read_csv(p)
        return {row["signal"]: row["weight"] for _, row in df.iterrows()}
    except Exception as exc:
        logger.exception("Failed loading signal weights: %s", exc)
        return {}

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
    except Exception:
        hist = []
    hist.append({"ts": datetime.utcnow().isoformat(), **metrics})
    hist = hist[-n_history:]
    with open(history_file, "w") as f:
        json.dump(hist, f)
    logger.info("META_METRICS", extra={"recent": hist})
    return True
