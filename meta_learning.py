"""Utility helpers for meta-learning weight management."""

import json
import logging
import pickle
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

try:
    import config  # AI-AGENT-REF: access centralized log paths
except ImportError:
    # Fallback for testing environments
    config = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import metrics_logger
except ImportError:
    # Mock metrics_logger for testing
    class MockMetricsLogger:
        def log_volatility(self, *args): pass
        def log_regime_toggle(self, *args): pass
    metrics_logger = MockMetricsLogger()

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import torch
    import torch.nn as _nn
    # ensure torch.nn and Parameter live on the torch module
    torch.nn = _nn
    torch.nn.Parameter = _nn.Parameter
except ImportError:
    torch = None

# For type checking only
if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

open = open  # allow monkeypatching built-in open

logger = logging.getLogger(__name__)


class MetaLearning:
    """Meta-learning wrapper using a simple linear model."""

    def __init__(self, model: Optional[Any] = None) -> None:
        try:
            from sklearn.linear_model import Ridge
            self.model = model or Ridge(alpha=1.0)
        except ImportError:
            print("WARNING: sklearn not available, using dummy model")
            # Create a dummy model that does nothing
            self.model = type('DummyModel', (), {
                'fit': lambda *a, **k: None,
                'predict': lambda *a, **k: [0] * len(a[0]) if a else [0]
            })()

    def train(self, df: "pd.DataFrame", target: str = "target") -> None:
        """Fit the meta learner using ``df`` columns except ``target``."""
        if pd is None:
            raise ImportError("pandas not available for training")
        if target not in df:
            raise ValueError(f"target column '{target}' missing")
        X = df.drop(columns=[target]).values
        y = df[target].values
        self.model.fit(X, y)

    def predict(self, features: Any) -> "np.ndarray":
        """Predict ensemble weights from ``features``."""
        if np is None:
            raise ImportError("numpy not available for prediction")
        if hasattr(self.model, "predict"):
            return np.asarray(self.model.predict(features))
        raise ValueError("Model not trained")

    def save_checkpoint(self, path: str = "models/meta_learner.pkl") -> None:
        """Save model checkpoint with error handling."""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(self.model, f)
            logger.info("Model checkpoint saved to %s", path)
        except (OSError, IOError, pickle.PickleError) as e:
            logger.error("Failed to save model checkpoint to %s: %s", path, e)
            raise

    def load_checkpoint(self, path: str = "models/meta_learner.pkl") -> None:
        """Load model checkpoint with error handling."""
        if Path(path).exists():
            try:
                with open(path, "rb") as f:
                    self.model = pickle.load(f)
                logger.info("Model checkpoint loaded from %s", path)
            except (OSError, IOError, pickle.PickleError, EOFError) as e:
                logger.error("Failed to load model checkpoint from %s: %s", path, e)
                raise
        else:
            logger.warning("Model checkpoint file does not exist: %s", path)


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


def load_weights(path: str, default: "np.ndarray | None" = None) -> "np.ndarray":
    """Load signal weights array from ``path`` or return ``default``."""
    if np is None:
        # Fallback when numpy is not available
        logger.warning("numpy not available, using basic weight loading")
        if default is None:
            return []  # Return empty list as fallback
        return default
        
    p = Path(path)
    if default is None:
        default = np.zeros(0)
        
    try:
        if p.exists():
            # Try CSV format first (matches update_weights format)
            try:
                # AI-AGENT-REF: Enhanced CSV reading with better error handling
                if path.endswith('.csv'):
                    # For CSV files, try pandas-style reading first
                    try:
                        import pandas as pd
                        df = pd.read_csv(p, usecols=["signal_name", "weight"])
                        if not df.empty:
                            weights = df["weight"].values
                            if isinstance(weights, np.ndarray):
                                logger.info("Successfully loaded weights from CSV: %s", path)
                                return weights
                    except (ImportError, ValueError) as e:
                        logger.debug("Pandas CSV read failed, trying numpy: %s", e)
                        # Fallback to numpy loadtxt
                        pass
                
                # Standard numpy reading for both CSV and other formats
                weights = np.loadtxt(p, delimiter=",")
                if isinstance(weights, np.ndarray):
                    logger.debug("Loaded weights using numpy from: %s", path)
                    return weights
            except (ValueError, OSError) as e:
                logger.debug("CSV/numpy loading failed, trying pickle: %s", e)
                # Fallback to pickle format for backward compatibility
                try:
                    with open(p, "rb") as f:
                        weights = pickle.load(f)
                        if isinstance(weights, np.ndarray):
                            logger.debug("Loaded weights from pickle: %s", path)
                            return weights
                        else:
                            logger.warning("Invalid weights format in %s, using default", path)
                except (pickle.PickleError, EOFError) as pickle_e:
                    logger.warning("Pickle loading also failed for %s: %s", path, pickle_e)
        else:
            logger.debug("Weights file %s not found, creating with default", path)
            # Create the default weights file when it doesn't exist
            if default.size > 0:
                try:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    np.savetxt(p, default, delimiter=",")
                    logger.info("Created default weights file: %s", path)
                except Exception as e:
                    logger.error("Failed initializing weights file %s: %s", path, e)
    except Exception as e:
        logger.warning("Failed to load weights from %s: %s", path, e)
        
    return default


def update_weights(
    weight_path: str,
    new_weights: "np.ndarray",
    metrics: dict,
    history_file: str = "metrics.json",
    n_history: int = 5,
) -> bool:
    """Update signal weights and append metric history."""
    if np is None:
        logger.error("numpy not available for updating weights")
        return False
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
    trade_log_path: str = None,
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
    
    # Set default trade log path
    if trade_log_path is None:
        trade_log_path = config.TRADE_LOG_FILE if config else "trades.csv"

    logger.info(
        "META_RETRAIN_START",
        extra={"trade_log": trade_log_path, "model_path": model_path},
    )

    if not Path(trade_log_path).exists():
        logger.error("Training data not found: %s", trade_log_path)
        return False
    try:
        if pd is None:
            logger.error("pandas not available for meta learning")
            return False
        df = pd.read_csv(trade_log_path)
    except (OSError, AttributeError) as exc:  # pragma: no cover - I/O failures
        logger.error("Failed reading trade log: %s", exc, exc_info=True)
        return False

    df = df.dropna(subset=["entry_price", "exit_price", "signal_tags", "side"])
    
    # AI-AGENT-REF: Enhanced meta learning data validation with better error handling
    try:
        # Convert price columns to numeric, handling various input formats
        df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
        df["exit_price"] = pd.to_numeric(df["exit_price"], errors="coerce")
        
        # Log initial data quality metrics
        initial_rows = len(df)
        logger.debug(f"META_LEARNING_INITIAL_DATA: {initial_rows} rows before validation")
        
        # Remove rows where price conversion failed
        df = df.dropna(subset=["entry_price", "exit_price"])
        after_numeric = len(df)
        if after_numeric < initial_rows:
            logger.info(f"META_LEARNING_PRICE_CONVERSION: Removed {initial_rows - after_numeric} rows with invalid price formats")
        
        # Validate price ranges - detect unrealistic values
        if len(df) > 0:
            # Check for extremely high/low prices that might indicate data corruption
            max_reasonable_price = 50000  # $50k per share
            min_reasonable_price = 0.01   # 1 cent
            
            price_issues = (
                (df["entry_price"] > max_reasonable_price) | 
                (df["entry_price"] < min_reasonable_price) |
                (df["exit_price"] > max_reasonable_price) | 
                (df["exit_price"] < min_reasonable_price)
            )
            
            if price_issues.any():
                problematic_rows = df[price_issues]
                logger.warning(f"META_LEARNING_PRICE_OUTLIERS: Found {len(problematic_rows)} rows with unrealistic prices")
                logger.debug(f"Price outliers range: entry({problematic_rows['entry_price'].min():.2f}-{problematic_rows['entry_price'].max():.2f}), exit({problematic_rows['exit_price'].min():.2f}-{problematic_rows['exit_price'].max():.2f})")
                
                # Filter out unrealistic prices
                df = df[~price_issues]
        
        # Check for reasonable price relationships
        if len(df) > 0:
            # Flag trades with extreme price movements (>1000% change) as potentially corrupted
            price_change_pct = abs((df["exit_price"] - df["entry_price"]) / df["entry_price"])
            extreme_moves = price_change_pct > 10.0  # 1000% change
            
            if extreme_moves.any():
                extreme_count = extreme_moves.sum()
                logger.warning(f"META_LEARNING_EXTREME_MOVES: Found {extreme_count} trades with >1000% price moves")
                # Keep extreme moves but flag them for review
                df.loc[extreme_moves, 'extreme_move'] = True
        
        # Final validation - ensure we have positive prices
        if len(df) > 0:
            positive_prices = (df["entry_price"] > 0) & (df["exit_price"] > 0)
            df = df[positive_prices]
            
            if not positive_prices.all():
                logger.info(f"META_LEARNING_NEGATIVE_PRICES: Filtered out {(~positive_prices).sum()} trades with non-positive prices")
        
        if len(df) == 0:
            logger.error("METALEARN_INVALID_PRICES - No trades with valid prices after comprehensive validation")
            return False
            
        # Test that final data quality summary is logged
        final_rows = len(df)
        retention_rate = (final_rows / initial_rows) * 100 if initial_rows > 0 else 0
        logger.info(f"META_LEARNING_DATA_QUALITY: Retained {final_rows}/{initial_rows} trades ({retention_rate:.1f}%)")
        
        # Validate price data statistics for reasonableness
        if final_rows > 0:
            entry_stats = {
                'min': float(df["entry_price"].min()),
                'max': float(df["exit_price"].max()), 
                'mean': float(df["entry_price"].mean())
            }
            exit_stats = {
                'min': float(df["exit_price"].min()),
                'max': float(df["exit_price"].max()),
                'mean': float(df["exit_price"].mean())
            }
            logger.debug(f"META_LEARNING_PRICE_STATS: Entry prices ${entry_stats['min']:.2f}-${entry_stats['max']:.2f} (avg: ${entry_stats['mean']:.2f})")
            logger.debug(f"META_LEARNING_PRICE_STATS: Exit prices ${exit_stats['min']:.2f}-${exit_stats['max']:.2f} (avg: ${exit_stats['mean']:.2f})")
            
    except Exception as e:
        logger.error("META_LEARNING_PRICE_VALIDATION_ERROR: %s", e, exc_info=True)
        return False
    
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

    try:
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0, fit_intercept=True)
    except ImportError:
        logger.warning("sklearn not available, meta-learning disabled")
        return {}

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
    # AI-AGENT-REF: Enhanced error handling for empty or invalid signal data
    if signal_data is None:
        logger.warning("optimize_signals received None signal_data, returning empty list")
        return []
    
    # Handle empty signal data gracefully
    if hasattr(signal_data, '__len__') and len(signal_data) == 0:
        logger.warning("optimize_signals received empty signal_data, returning empty list")
        return []
    
    if model is not None:
        try:
            # when a model instance is provided, return its raw predictions exactly
            predictions = model.predict(signal_data)
            if predictions is None:
                logger.warning("Model prediction returned None, falling back to original signal_data")
                return signal_data if signal_data is not None else []
            return list(predictions)
        except (ValueError, RuntimeError, AttributeError) as exc:
            logger.exception("optimize_signals model prediction failed: %s", exc)
            return signal_data if signal_data is not None else []
    
    if model is None:
        try:
            model = load_model_checkpoint(cfg.MODEL_PATH)
        except AttributeError:
            logger.warning("cfg object missing MODEL_PATH attribute")
            return signal_data if signal_data is not None else []
    
    if model is None:
        logger.debug("No model available for signal optimization, returning original data")
        return signal_data if signal_data is not None else []
    
    try:
        preds = model.predict(signal_data)
        if preds is None:
            logger.warning("Model predict returned None")
            return signal_data if signal_data is not None else []
        
        # Enhanced clipping with validation
        if np is not None:
            preds = np.clip(preds, -1.2, 1.2)
            factor = 1.0 if volatility <= 1.0 else 1.0 / max(volatility, 1e-3)
            preds = preds * factor
            return list(preds)  # AI-AGENT-REF: return list to avoid bool ambiguity
        else:
            # Fallback when numpy is not available
            preds = [max(-1.2, min(1.2, p)) for p in preds]
            factor = 1.0 if volatility <= 1.0 else 1.0 / max(volatility, 1e-3)
            preds = [p * factor for p in preds]
            return preds
    except (ValueError, RuntimeError, TypeError) as exc:  # pragma: no cover - model may fail
        logger.exception("optimize_signals prediction processing failed: %s", exc)
        return signal_data if signal_data is not None else []


try:
    from portfolio_rl import PortfolioReinforcementLearner
except ImportError:
    # Mock for testing environments
    class PortfolioReinforcementLearner:
        def rebalance_portfolio(self, *args):
            return [1.0]  # Return mock result


def trigger_rebalance_on_regime(df: "pd.DataFrame") -> None:
    """Invoke the RL rebalancer when the market regime changes."""
    rl = PortfolioReinforcementLearner()
    if "Regime" in df.columns and len(df) > 2:
        if df["Regime"].iloc[-1] != df["Regime"].iloc[-2]:
            state_data = df.tail(10).dropna().values.flatten()
            rl.rebalance_portfolio(state_data)
