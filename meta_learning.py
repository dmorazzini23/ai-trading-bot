"""Utility helpers for meta-learning weight management."""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

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
                logger.info("Initialized default weights at %s", path)
            except Exception as exc:
                logger.exception("Failed initializing default weights: %s", exc)
        return default
    try:
        weights = np.loadtxt(p, delimiter=",")
        return weights
    except Exception as exc:
        logger.exception("Failed to load signal weights: %s", exc)
        return default


def update_weights(
    weight_path: str,
    new_weights: np.ndarray,
    metrics: dict,
    history_file: str = "metrics.json",
    n_history: int = 5,
) -> bool:
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


def update_signal_weights(weights: Dict[str, float], performance: Dict[str, float]) -> Optional[Dict[str, float]]:
    if not weights or not performance:
        logger.error("Empty weights or performance dict passed to update_signal_weights")
        return None
    try:
        total_perf = sum(performance.values())
        if total_perf == 0:
            logger.warning("Total performance sum is zero, skipping weight update")
            return weights
        updated_weights = {}
        for key in weights.keys():
            perf = performance.get(key, 0)
            updated_weights[key] = weights[key] * (perf / total_perf)
        norm_factor = sum(updated_weights.values())
        if norm_factor == 0:
            logger.warning("Normalization factor zero in weight update")
            return weights
        for key in updated_weights:
            updated_weights[key] /= norm_factor
        return updated_weights
    except Exception as e:
        logger.error(f"Exception in update_signal_weights: {e}", exc_info=True)
        return weights


def save_model_checkpoint(model: Any, filepath: str) -> None:
    """Serialize ``model`` to ``filepath`` using :mod:`pickle`."""
    try:
        with open(filepath, "wb") as f:
            pickle.dump(model, f)
        logger.info("MODEL_CHECKPOINT_SAVED", extra={"path": filepath})
    except Exception as exc:  # pragma: no cover - unexpected I/O
        logger.error("Failed to save model checkpoint: %s", exc, exc_info=True)


def load_model_checkpoint(filepath: str) -> Optional[Any]:
    """Load a model from ``filepath`` previously saved with ``save_model_checkpoint``."""
    if not Path(filepath).exists():
        logger.warning("Checkpoint file missing: %s", filepath)
        return None
    try:
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        logger.info("MODEL_CHECKPOINT_LOADED", extra={"path": filepath})
        return model
    except Exception as exc:  # pragma: no cover - unexpected I/O
        logger.error("Failed to load model checkpoint: %s", exc, exc_info=True)
        return None


def retrain_meta_learner(
    trade_log_path: str = "trades.csv",
    model_path: str = "meta_model.pkl",
    history_path: str = "meta_retrain_history.pkl",
    min_samples: int = 20,
) -> bool:
    """Retrain the meta-learner model from trade logs.

    Parameters
    ----------
    trade_log_path : str
        CSV file containing historical trades.
    model_path : str
        Destination to write the trained model pickle.
    history_path : str
        Path to a pickle file storing retrain metrics history.
    min_samples : int
        Minimum number of samples required to train.

    Returns
    -------
    bool
        ``True`` if retraining succeeded and the checkpoint was written.
    """

    logger.info(
        "META_RETRAIN_START",
        extra={"trade_log": trade_log_path, "model_path": model_path},
    )

    if not Path(trade_log_path).exists():
        logger.error("Training data not found: %s", trade_log_path)
        return False
    try:
        df = pd.read_csv(trade_log_path)
    except Exception as exc:  # pragma: no cover - I/O failures
        logger.error("Failed reading trade log: %s", exc, exc_info=True)
        return False

    df = df.dropna(subset=["entry_price", "exit_price", "signal_tags", "side"])
    if len(df) < min_samples:
        logger.warning(
            "META_RETRAIN_INSUFFICIENT_DATA", extra={"rows": len(df)}
        )
        return False

    df["pnl"] = (df["exit_price"] - df["entry_price"]) * df["side"].map(
        {"buy": 1, "sell": -1}
    )
    df["outcome"] = (df["pnl"] > 0).astype(int)

    tags = sorted({t for row in df["signal_tags"] for t in str(row).split("+")})
    X = np.array(
        [[int(t in str(row).split("+")) for t in tags] for row in df["signal_tags"]]
    )
    y = df["outcome"].values
    sample_w = df["pnl"].abs() + 1e-3

    from sklearn.linear_model import Ridge

    model = Ridge(alpha=1.0, fit_intercept=True)
    try:
        model.fit(X, y, sample_weight=sample_w)
    except Exception as exc:  # pragma: no cover - sklearn failure
        logger.error("Meta-learner training failed: %s", exc, exc_info=True)
        return False

    save_model_checkpoint(model, model_path)

    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "samples": len(y),
        "model_path": model_path,
    }
    hist: list[dict[str, Any]] = []
    if Path(history_path).exists():
        loaded = load_model_checkpoint(history_path)
        if isinstance(loaded, list):
            hist = loaded
    hist.append(metrics)
    hist = hist[-5:]
    try:
        with open(history_path, "wb") as f:
            pickle.dump(hist, f)
    except Exception as exc:  # pragma: no cover - unexpected I/O
        logger.error("Failed to update retrain history: %s", exc, exc_info=True)

    logger.info(
        "META_RETRAIN_SUCCESS",
        extra={"samples": len(y), "model": model_path},
    )
    return True
