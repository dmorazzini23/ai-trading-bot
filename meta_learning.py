"""Utility helpers for meta-learning weight management."""

import json
import logging
import pickle
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import metrics_logger
import pandas as pd

open = open  # allow monkeypatching built-in open

logger = logging.getLogger(__name__)


class MetaLearning:
    """Simple meta-learning stub for dynamic strategy tuning."""

    def __init__(self, model: Optional[Any] = None) -> None:
        self.model = model or {}

    def update(self, data: Any) -> None:
        """Placeholder update routine with basic logging."""
        logger.info(
            "Updating MetaLearning with data shape: %s",
            getattr(data, "shape", None),
        )

    def predict(self, features: Any) -> float:
        """Return a dummy prediction score."""
        return random.uniform(0, 1)


def normalize_score(score: float, cap: float = 1.2) -> float:
    """Clip ``score`` to ``cap`` preserving sign."""
    try:
        score = float(score)
    except Exception:
        return 0.0
    return max(-cap, min(cap, score))


def adjust_confidence(confidence: float, volatility: float, threshold: float = 1.0) -> float:
    """Scale confidence by inverse volatility to reduce spam at high levels."""
    try:
        conf = float(confidence)
        vol = float(volatility)
    except Exception:
        return 0.0
    factor = 1.0 if vol <= threshold else 1.0 / max(vol, 1e-3)
    return max(0.0, min(1.0, conf * factor))


def volatility_regime_filter(atr: float, sma100: float) -> str:
    """Return volatility regime string based on ATR and SMA."""
    if sma100 == 0:
        return "unknown"
    ratio = atr / sma100
    regime = "high_vol" if ratio > 0.05 else "low_vol"
    metrics_logger.log_volatility(ratio)
    metrics_logger.log_regime_toggle("generic", regime)
    return regime


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
            except OSError as exc:
                logger.exception("Failed initializing default weights: %s", exc)
        return default
    try:
        weights = np.loadtxt(p, delimiter=",")
        return weights
    except (OSError, ValueError) as exc:
        logger.exception("Failed to load signal weights: %s", exc)
        return default


def update_weights(
    weight_path: str,
    new_weights: np.ndarray,
    metrics: dict,
    history_file: str = "metrics.json",
    n_history: int = 5,
) -> bool:
    """Update signal weights and append metric history."""
    if new_weights.size == 0:
        logger.error("update_weights called with empty weight array")
        return False
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
    except (OSError, ValueError) as exc:
        logger.exception("META_WEIGHT_UPDATE_FAILED: %s", exc)
        return False
    try:
        if Path(history_file).exists():
            with open(history_file, encoding="utf-8") as f:
                hist = json.load(f)
        else:
            hist = []
    except (OSError, json.JSONDecodeError) as e:
        logger.error("Failed to read metric history: %s", e)
        hist = []
    hist.append({"ts": datetime.now(timezone.utc).isoformat(), **metrics})
    hist = hist[-n_history:]
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(hist, f)
    logger.info("META_METRICS", extra={"recent": hist})
    return True


def update_signal_weights(
    weights: Dict[str, float], performance: Dict[str, float]
) -> Optional[Dict[str, float]]:
    if not weights or not performance:
        logger.error(
            "Empty weights or performance dict passed to update_signal_weights"
        )
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
    except (ZeroDivisionError, TypeError) as exc:
        logger.exception("Exception in update_signal_weights: %s", exc)
        return weights


def save_model_checkpoint(model: Any, filepath: str) -> None:
    """Serialize ``model`` to ``filepath`` using :mod:`pickle`."""
    try:
        with open(filepath, "wb") as f:
            pickle.dump(model, f)
        logger.info("MODEL_CHECKPOINT_SAVED", extra={"path": filepath})
    except (OSError, pickle.PickleError) as exc:  # pragma: no cover - unexpected I/O
        logger.error("Failed to save model checkpoint: %s", exc, exc_info=True)


def load_model_checkpoint(filepath: str) -> Optional[Any]:
    """Load a model from ``filepath`` previously saved with ``save_model_checkpoint``."""
    p = Path(filepath)
    if not p.exists():
        logger.warning("Checkpoint file missing: %s", filepath)
    try:
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        logger.info("MODEL_CHECKPOINT_LOADED", extra={"path": filepath})
        return model
    except (OSError, pickle.PickleError) as exc:  # pragma: no cover - unexpected I/O
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
    except (OSError, pd.errors.ParserError) as exc:  # pragma: no cover - I/O failures
        logger.error("Failed reading trade log: %s", exc, exc_info=True)
        return False

    df = df.dropna(subset=["entry_price", "exit_price", "signal_tags", "side"])
    if len(df) < min_samples:
        logger.warning("META_RETRAIN_INSUFFICIENT_DATA", extra={"rows": len(df)})
        return False

    direction = np.where(df["side"] == "buy", 1, -1)
    df["pnl"] = (df["exit_price"] - df["entry_price"]) * direction
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
    except (ValueError, RuntimeError) as exc:  # pragma: no cover - sklearn failure
        logger.exception("Meta-learner training failed: %s", exc)
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
    except (OSError, pickle.PickleError) as exc:  # pragma: no cover - unexpected I/O
        logger.error("Failed to update retrain history: %s", exc, exc_info=True)

    logger.info(
        "META_RETRAIN_SUCCESS",
        extra={"samples": len(y), "model": model_path},
    )
    return True


def optimize_signals(signal_data: Any, cfg: Any, model: Any | None = None, *, volatility: float = 1.0) -> Any:
    """Optimize trading signals using ``model`` if provided."""
    if model is None:
        model = load_model_checkpoint(cfg.MODEL_PATH)
    if model is None:
        return signal_data
    try:
        preds = model.predict(signal_data)
        preds = np.clip(preds, -1.2, 1.2)
        factor = 1.0 if volatility <= 1.0 else 1.0 / max(volatility, 1e-3)
        preds = preds * factor
        return preds
    except (ValueError, RuntimeError) as exc:  # pragma: no cover - model may fail
        logger.exception("optimize_signals failed: %s", exc)
        return signal_data


from portfolio_rl import PortfolioReinforcementLearner


def trigger_rebalance_on_regime(df: pd.DataFrame) -> None:
    """Invoke the RL rebalancer when the market regime changes."""
    rl = PortfolioReinforcementLearner()
    if "Regime" in df.columns and len(df) > 2:
        if df["Regime"].iloc[-1] != df["Regime"].iloc[-2]:
            state_data = df.tail(10).dropna().values.flatten()
            rl.rebalance_portfolio(state_data)
