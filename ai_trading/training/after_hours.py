"""After-hours ML/RL training orchestration.

This module builds cost-aware datasets, runs purged walk-forward evaluation,
trains calibrated baseline ML models, and optionally trains an RL overlay.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time as dt_time, timedelta
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

import numpy as np

from ai_trading import paths
from ai_trading.config.management import get_env
from ai_trading.data.fetch import get_daily_df
from ai_trading.data.splits import PurgedGroupTimeSeriesSplit
from ai_trading.features.indicators import (
    compute_atr,
    compute_macd,
    compute_macds,
    compute_sma,
    compute_vwap,
)
from ai_trading.indicators import rsi as rsi_indicator
from ai_trading.logging import get_logger
from ai_trading.model_registry import ModelRegistry
from ai_trading.models.artifacts import write_artifact_manifest
from ai_trading.monitoring.model_liveness import note_after_hours_training_complete
from ai_trading.registry.manifest import validate_manifest_metadata
from ai_trading.research.leakage_tests import run_leakage_guards
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path

logger = get_logger(__name__)

_BASE_FEATURE_COLUMNS: tuple[str, ...] = (
    "rsi",
    "macd",
    "atr",
    "vwap",
    "sma_50",
    "sma_200",
    "signal",
)
_DERIVED_FEATURE_COLUMNS: tuple[str, ...] = (
    "atr_pct",
    "vwap_distance",
    "sma_spread",
    "macd_signal_gap",
    "rsi_centered",
)
FEATURE_COLUMNS: tuple[str, ...] = _BASE_FEATURE_COLUMNS + _DERIVED_FEATURE_COLUMNS
_THRESHOLD_GRID: tuple[float, ...] = (0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)
_TCA_TIMESTAMP_KEYS: tuple[str, ...] = (
    "ts",
    "timestamp",
    "created_at",
    "updated_at",
    "submitted_at",
    "decision_ts",
    "fill_ts",
)
_MODEL_SELECTION_WEIGHT_SPECS: tuple[tuple[str, str, float, float, float], ...] = (
    (
        "drawdown_penalty",
        "AI_TRADING_AFTER_HOURS_MODEL_SELECTION_DRAWDOWN_PENALTY",
        0.003,
        0.0,
        0.05,
    ),
    (
        "turnover_penalty",
        "AI_TRADING_AFTER_HOURS_MODEL_SELECTION_TURNOVER_PENALTY",
        0.75,
        0.0,
        5.0,
    ),
    (
        "brier_penalty",
        "AI_TRADING_AFTER_HOURS_MODEL_SELECTION_BRIER_PENALTY",
        4.0,
        0.0,
        60.0,
    ),
    (
        "stability_weight",
        "AI_TRADING_AFTER_HOURS_MODEL_SELECTION_STABILITY_WEIGHT",
        0.5,
        0.0,
        5.0,
    ),
    (
        "support_log_weight",
        "AI_TRADING_AFTER_HOURS_MODEL_SELECTION_SUPPORT_LOG_WEIGHT",
        0.05,
        0.0,
        1.0,
    ),
    (
        "profitable_fold_weight",
        "AI_TRADING_AFTER_HOURS_MODEL_SELECTION_PROFITABLE_FOLD_WEIGHT",
        0.5,
        0.0,
        5.0,
    ),
)
def _resolve_after_hours_output_path(path_value: str, *, default_relative: str) -> Path:
    """Resolve output paths relative to writable runtime roots when possible."""
    return resolve_runtime_artifact_path(
        path_value,
        default_relative=default_relative,
    )


def _resolve_writable_output_dir(
    *,
    requested: Path,
    fallback: Path | None = None,
    event_name: str,
) -> Path:
    candidates: list[Path] = [requested]
    if fallback is not None and fallback != requested:
        candidates.append(fallback)

    last_error: Exception | None = None
    for idx, candidate in enumerate(candidates):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            last_error = exc
            if idx + 1 < len(candidates):
                logger.warning(
                    event_name,
                    extra={
                        "requested": str(requested),
                        "fallback": str(candidates[idx + 1]),
                        "reason": "mkdir_failed",
                        "error": str(exc),
                    },
                )
                continue
            break
        if os.access(candidate, os.W_OK):
            if candidate != requested:
                logger.warning(
                    event_name,
                    extra={
                        "requested": str(requested),
                        "fallback": str(candidate),
                        "reason": "requested_not_writable",
                    },
                )
            return candidate
        last_error = PermissionError(f"Directory is not writable: {candidate}")
        if idx + 1 < len(candidates):
            logger.warning(
                event_name,
                extra={
                    "requested": str(requested),
                    "fallback": str(candidates[idx + 1]),
                    "reason": "requested_not_writable",
                },
            )
            continue
        break

    detail = str(last_error) if last_error is not None else "unknown_error"
    raise RuntimeError(f"No writable directory available for {requested}: {detail}")


def _dump_model_with_fallback(
    model: Any,
    model_path: Path,
    *,
    fallback_dir: Path | None,
) -> Path:
    import joblib

    try:
        joblib.dump(model, model_path)
        return model_path
    except OSError as exc:
        is_perm_error = exc.errno in {errno.EROFS, errno.EACCES, errno.EPERM}
        if not is_perm_error or fallback_dir is None:
            raise
        fallback_path = fallback_dir / model_path.name
        fallback_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(fallback_dir, os.W_OK):
            raise RuntimeError(f"Fallback model directory is not writable: {fallback_dir}") from exc
        logger.warning(
            "AFTER_HOURS_MODEL_WRITE_FALLBACK",
            extra={
                "requested_path": str(model_path),
                "fallback_path": str(fallback_path),
                "error": str(exc),
            },
        )
        joblib.dump(model, fallback_path)
        return fallback_path


@dataclass(slots=True)
class EdgeTargets:
    """Minimum edge criteria required for promotion gates."""

    min_expectancy_bps: float
    max_drawdown_bps: float
    max_turnover_ratio: float
    min_hit_rate_stability: float


@dataclass(slots=True)
class CandidateMetrics:
    """Aggregated out-of-sample metrics for a model candidate."""

    name: str
    fold_count: int
    profitable_fold_count: int
    profitable_fold_ratio: float
    support: int
    mean_expectancy_bps: float
    max_drawdown_bps: float
    turnover_ratio: float
    mean_hit_rate: float
    hit_rate_stability: float
    regime_metrics: dict[str, dict[str, float]]
    oof_probabilities: np.ndarray
    brier_score: float = 1.0
    regime_calibration: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(slots=True)
class _CandidateSnapshot:
    name: str
    mean_expectancy_bps: float
    max_drawdown_bps: float
    turnover_ratio: float
    hit_rate_stability: float
    brier_score: float
    support: int
    profitable_fold_ratio: float


@dataclass(slots=True)
class _ReportSnapshot:
    ts: datetime
    source: str
    model_name: str
    mean_expectancy_bps: float
    hit_rate_stability: float
    brier_score: float
    candidates: list[_CandidateSnapshot]


class _FallbackProbabilityModel:
    """Lightweight probabilistic classifier used when sklearn is stubbed."""

    def __init__(self) -> None:
        self._weights: np.ndarray | None = None
        self.feature_names_in_: np.ndarray | None = None

    @staticmethod
    def _as_array(X: Any) -> np.ndarray:
        if hasattr(X, "to_numpy"):
            arr = np.asarray(X.to_numpy(), dtype=float)
        else:
            arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return np.asarray(
            np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0),
            dtype=float,
        )

    def fit(self, X: Any, y: Any) -> "_FallbackProbabilityModel":
        X_arr = self._as_array(X)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        else:
            self.feature_names_in_ = np.asarray(
                [f"f_{idx}" for idx in range(X_arr.shape[1])], dtype=object
            )
        y_centered = np.clip(y_arr, 0.0, 1.0) * 2.0 - 1.0
        Xb = np.hstack([np.ones((X_arr.shape[0], 1), dtype=float), X_arr])
        ridge = np.eye(Xb.shape[1], dtype=float) * 1e-3
        self._weights = np.linalg.pinv(Xb.T @ Xb + ridge) @ Xb.T @ y_centered
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("Model not fitted")
        X_arr = self._as_array(X)
        Xb = np.hstack([np.ones((X_arr.shape[0], 1), dtype=float), X_arr])
        logits = Xb @ self._weights
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X: Any) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


def _fetch_daily_bars(symbol: str, start_dt: datetime, end_dt: datetime):
    return get_daily_df(symbol, start_dt, end_dt)


def _load_symbols() -> list[str]:
    path_raw = str(get_env("AI_TRADING_AFTER_HOURS_TICKERS_CSV", "") or "").strip()
    if not path_raw:
        path_raw = str(get_env("AI_TRADING_TICKERS_FILE", "") or "").strip()
    path = Path(path_raw) if path_raw else None
    symbols: list[str] = []
    if path is not None and path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        for raw in lines:
            token = raw.strip().upper()
            if not token or token == "SYMBOL":
                continue
            if token not in symbols:
                symbols.append(token)
    if symbols:
        return symbols
    fallback = str(get_env("AI_TRADING_RESEARCH_SYMBOLS", "AAPL,MSFT,SPY") or "")
    for raw in fallback.split(","):
        token = raw.strip().upper()
        if token and token not in symbols:
            symbols.append(token)
    return symbols


def _read_jsonl_records(path: str, *, max_records: int = 20000) -> list[dict[str, Any]]:
    src = Path(path)
    if not src.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        payload_raw = line.strip()
        if not payload_raw:
            continue
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    if len(rows) > max_records:
        return rows[-max_records:]
    return rows


def _parse_ts(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _tca_row_timestamp(row: Mapping[str, Any]) -> datetime | None:
    for key in _TCA_TIMESTAMP_KEYS:
        parsed = _parse_ts(row.get(key))
        if parsed is not None:
            return parsed
    return None


def _clamp(value: float, *, low: float, high: float) -> float:
    lo = min(low, high)
    hi = max(low, high)
    return float(min(max(float(value), lo), hi))


def _build_fill_quality_metrics(tca_records: list[dict[str, Any]]) -> dict[str, float]:
    if not tca_records:
        return {
            "mean_is_bps": 0.0,
            "p95_is_bps": 0.0,
            "fill_rate": 0.0,
            "partial_fill_rate": 0.0,
            "mean_fill_latency_ms": 0.0,
        }
    is_bps_vals: list[float] = []
    latency_vals: list[float] = []
    filled = 0
    partial = 0
    for row in tca_records:
        try:
            is_bps_vals.append(float(row.get("is_bps", 0.0) or 0.0))
        except (TypeError, ValueError):
            continue
        latency_raw = row.get("fill_latency_ms")
        if latency_raw is not None:
            try:
                latency_vals.append(float(latency_raw))
            except (TypeError, ValueError):
                pass
        status = str(row.get("status", "")).lower()
        if status in {"filled", "partially_filled"}:
            filled += 1
        if bool(row.get("partial_fill")):
            partial += 1
    p95 = float(np.percentile(is_bps_vals, 95)) if is_bps_vals else 0.0
    return {
        "mean_is_bps": float(mean(is_bps_vals)) if is_bps_vals else 0.0,
        "p95_is_bps": p95,
        "fill_rate": float(filled) / float(max(1, len(tca_records))),
        "partial_fill_rate": float(partial) / float(max(1, len(tca_records))),
        "mean_fill_latency_ms": float(mean(latency_vals)) if latency_vals else 0.0,
    }


def _estimate_cost_floor_bps(tca_records: list[dict[str, Any]]) -> float:
    baseline = float(get_env("AI_TRADING_COST_FLOOR_BPS", 12.0, cast=float))
    min_bps = float(get_env("AI_TRADING_AFTER_HOURS_COST_FLOOR_MIN_BPS", 4.0, cast=float))
    max_bps = float(get_env("AI_TRADING_AFTER_HOURS_COST_FLOOR_MAX_BPS", 25.0, cast=float))
    if not tca_records:
        return _clamp(baseline, low=min_bps, high=max_bps)

    lookback_days = int(get_env("AI_TRADING_AFTER_HOURS_COST_FLOOR_LOOKBACK_DAYS", 45, cast=int))
    min_samples = int(get_env("AI_TRADING_AFTER_HOURS_COST_FLOOR_MIN_SAMPLES", 40, cast=int))
    require_filled = bool(
        get_env("AI_TRADING_AFTER_HOURS_COST_FLOOR_REQUIRE_FILLED", True, cast=bool)
    )
    outlier_cap_bps = float(
        get_env("AI_TRADING_AFTER_HOURS_COST_FLOOR_OUTLIER_BPS", 120.0, cast=float)
    )
    quantile = _clamp(
        float(get_env("AI_TRADING_AFTER_HOURS_COST_FLOOR_QUANTILE", 0.6, cast=float)),
        low=0.05,
        high=0.95,
    )
    tca_weight = _clamp(
        float(get_env("AI_TRADING_AFTER_HOURS_COST_FLOOR_TCA_WEIGHT", 0.65, cast=float)),
        low=0.0,
        high=1.0,
    )

    cutoff = datetime.now(UTC) - timedelta(days=max(1, lookback_days))
    values: list[float] = []
    for row in tca_records:
        status = str(row.get("status", "")).lower()
        if require_filled and status not in {"filled", "partially_filled"}:
            continue
        row_ts = _tca_row_timestamp(row)
        if row_ts is not None and row_ts < cutoff:
            continue
        try:
            value = abs(float(row.get("is_bps", 0.0) or 0.0))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        if outlier_cap_bps > 0.0 and value > outlier_cap_bps:
            continue
        values.append(value)

    if len(values) < max(1, min_samples):
        return _clamp(baseline, low=min_bps, high=max_bps)

    arr = np.asarray(values, dtype=float)
    if arr.size >= 20:
        lo = float(np.percentile(arr, 5))
        hi = float(np.percentile(arr, 95))
        arr = np.clip(arr, lo, hi)
    tca_estimate = float(np.percentile(arr, quantile * 100.0))
    blended = (tca_weight * tca_estimate) + ((1.0 - tca_weight) * baseline)
    stabilized = _clamp(blended, low=min_bps, high=max_bps)
    logger.info(
        "AFTER_HOURS_COST_FLOOR_ESTIMATE",
        extra={
            "baseline_bps": baseline,
            "tca_samples": int(len(values)),
            "tca_estimate_bps": tca_estimate,
            "stabilized_bps": stabilized,
            "lookback_days": int(max(1, lookback_days)),
            "quantile": quantile,
            "tca_weight": tca_weight,
        },
    )
    return stabilized


def _infer_regime(close: np.ndarray) -> np.ndarray:
    if close.size == 0:
        return np.array([], dtype=object)
    returns = np.zeros_like(close, dtype=float)
    returns[1:] = np.diff(close) / np.maximum(close[:-1], 1e-9)
    vol = np.full_like(returns, np.nan, dtype=float)
    trend = np.full_like(returns, np.nan, dtype=float)
    window = 20
    for idx in range(window, len(returns)):
        seg = returns[idx - window : idx]
        vol[idx] = float(np.std(seg))
        trend[idx] = float((close[idx] / close[idx - window]) - 1.0)
    vol_threshold = np.nanpercentile(vol, 70) if np.isfinite(vol).any() else 0.02
    labels: list[str] = []
    for idx in range(len(close)):
        vol_val = vol[idx]
        trend_val = trend[idx]
        if np.isfinite(vol_val) and vol_val >= vol_threshold:
            labels.append("volatile")
        elif np.isfinite(trend_val) and trend_val >= 0.02:
            labels.append("uptrend")
        elif np.isfinite(trend_val) and trend_val <= -0.02:
            labels.append("downtrend")
        else:
            labels.append("sideways")
    return np.array(labels, dtype=object)


def _safe_rsi(close_values: np.ndarray) -> np.ndarray:
    if close_values.size == 0:
        return np.array([], dtype=float)
    try:
        out = rsi_indicator(tuple(close_values.tolist()), 14)
    except Exception:
        out = None
    if out is None:
        return np.zeros_like(close_values, dtype=float)
    try:
        arr = np.asarray(out, dtype=float)
    except Exception:
        return np.zeros_like(close_values, dtype=float)
    if arr.size != close_values.size:
        return np.zeros_like(close_values, dtype=float)
    return arr


def _augment_training_features(frame: Any) -> Any:
    import pandas as pd

    close = pd.to_numeric(frame.get("close"), errors="coerce")
    close_abs = close.abs().replace(0.0, np.nan)
    atr = pd.to_numeric(frame.get("atr"), errors="coerce")
    vwap = pd.to_numeric(frame.get("vwap"), errors="coerce").replace(0.0, np.nan)
    sma_50 = pd.to_numeric(frame.get("sma_50"), errors="coerce")
    sma_200 = pd.to_numeric(frame.get("sma_200"), errors="coerce")
    macd = pd.to_numeric(frame.get("macd"), errors="coerce")
    rsi = pd.to_numeric(frame.get("rsi"), errors="coerce")
    signal = pd.to_numeric(
        frame.get("signal", frame.get("macds", frame.get("macd"))),
        errors="coerce",
    )

    frame["signal"] = signal
    frame["atr_pct"] = (atr / close_abs) * 100.0
    frame["vwap_distance"] = (close / vwap) - 1.0
    frame["sma_spread"] = (sma_50 - sma_200) / close_abs
    frame["macd_signal_gap"] = macd - signal
    frame["rsi_centered"] = (rsi - 50.0) / 50.0

    for column in _DERIVED_FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    return frame


def _build_symbol_dataset(
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    *,
    cost_floor_bps: float,
):
    import pandas as pd

    bars = _fetch_daily_bars(symbol, start_dt, end_dt)
    if bars is None or bars.empty:
        return pd.DataFrame()
    frame = bars.copy()
    columns_lower = {str(col): str(col).lower() for col in frame.columns}
    frame = frame.rename(columns=columns_lower)
    for required in ("open", "high", "low", "close", "volume"):
        if required not in frame.columns:
            return pd.DataFrame()
    frame = frame.sort_index()
    frame = compute_macd(frame)
    frame = compute_macds(frame)
    frame = compute_atr(frame)
    frame = compute_vwap(frame)
    frame = compute_sma(frame, windows=(50, 200))
    close_arr = frame["close"].astype(float).to_numpy()
    frame["rsi"] = _safe_rsi(close_arr)
    frame = _augment_training_features(frame)
    future_ret_bps = (
        frame["close"].astype(float).shift(-1) / frame["close"].astype(float) - 1.0
    ) * 10_000.0
    frame["realized_edge_bps"] = future_ret_bps - float(cost_floor_bps)
    frame["label"] = (frame["realized_edge_bps"] > 0).astype(int)
    frame["regime"] = _infer_regime(close_arr)
    timestamps = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame["timestamp"] = timestamps
    frame["label_ts"] = timestamps + pd.Timedelta(days=1)
    frame["symbol"] = symbol
    frame = frame.dropna(
        subset=list(FEATURE_COLUMNS)
        + ["timestamp", "label_ts", "realized_edge_bps", "label", "regime"]
    )
    return frame


def _build_training_dataset(
    symbols: list[str],
    *,
    lookback_days: int,
    cost_floor_bps: float,
    now_utc: datetime,
):
    import pandas as pd

    start_dt = now_utc - timedelta(days=int(max(180, lookback_days)))
    rows: list[pd.DataFrame] = []
    for symbol in symbols:
        frame = _build_symbol_dataset(
            symbol,
            start_dt,
            now_utc,
            cost_floor_bps=cost_floor_bps,
        )
        if frame is not None and not frame.empty:
            rows.append(frame)
    if not rows:
        return pd.DataFrame()
    dataset = pd.concat(rows, axis=0, ignore_index=True)
    dataset = dataset.sort_values("timestamp").reset_index(drop=True)
    return dataset


def _fit_candidate_model(name: str, seed: int):
    if name == "logreg":
        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            base = make_pipeline(
                StandardScaler(with_mean=True),
                LogisticRegression(
                    max_iter=1000,
                    random_state=seed,
                    class_weight="balanced",
                ),
            )
            return CalibratedClassifierCV(base, method="sigmoid", cv=3)
        except Exception:
            return _FallbackProbabilityModel()
    if name == "histgb":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import HistGradientBoostingClassifier

        base = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=350,
            min_samples_leaf=24,
            l2_regularization=0.02,
            random_state=seed,
        )
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)
    if name == "lightgbm":
        from sklearn.calibration import CalibratedClassifierCV

        import lightgbm as lgb

        base = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=seed,
            objective="binary",
            subsample=0.8,
            colsample_bytree=0.8,
            verbosity=int(get_env("AI_TRADING_AFTER_HOURS_LIGHTGBM_VERBOSITY", -1, cast=int)),
        )
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)
    if name == "xgboost":
        from sklearn.calibration import CalibratedClassifierCV

        import xgboost as xgb

        base = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
        )
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)
    raise ValueError(f"Unsupported candidate model: {name}")


def _predict_probabilities(model: Any, X: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        out = model.predict_proba(X)
        arr = np.asarray(out, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, 1]
        if arr.ndim == 1:
            return arr
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        return np.asarray(1.0 / (1.0 + np.exp(-scores)), dtype=float)
    preds = np.asarray(model.predict(X), dtype=float)
    return np.clip(preds, 0.0, 1.0)


def _max_drawdown_bps(edge_bps: np.ndarray) -> float:
    if edge_bps.size == 0:
        return 0.0
    equity = np.cumsum(edge_bps)
    peaks = np.maximum.accumulate(equity)
    drawdown = peaks - equity
    return float(np.max(drawdown)) if drawdown.size else 0.0


def _regime_summary(
    regimes: np.ndarray,
    selected: np.ndarray,
    realized_edge_bps: np.ndarray,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if regimes.size == 0:
        return out
    for regime in sorted({str(val) for val in regimes}):
        mask = regimes.astype(str) == regime
        if not np.any(mask):
            continue
        chosen = selected & mask
        support = int(np.sum(chosen))
        if support <= 0:
            out[regime] = {"support": 0.0, "expectancy_bps": 0.0, "hit_rate": 0.0}
            continue
        selected_edges = realized_edge_bps[chosen]
        out[regime] = {
            "support": float(support),
            "expectancy_bps": float(np.mean(selected_edges)),
            "hit_rate": float(np.mean(selected_edges > 0)),
        }
    return out


def _expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    labels_arr = np.asarray(labels, dtype=float)
    probs_arr = np.asarray(probabilities, dtype=float)
    if labels_arr.size == 0 or probs_arr.size == 0 or labels_arr.size != probs_arr.size:
        return 1.0
    clipped = np.clip(probs_arr, 1e-6, 1.0 - 1e-6)
    bucket_ids = np.minimum((clipped * bins).astype(int), bins - 1)
    total = float(clipped.size)
    ece = 0.0
    for bucket in range(max(1, int(bins))):
        mask = bucket_ids == bucket
        if not np.any(mask):
            continue
        conf = float(np.mean(clipped[mask]))
        acc = float(np.mean(labels_arr[mask]))
        weight = float(np.sum(mask)) / total
        ece += weight * abs(acc - conf)
    return float(max(0.0, min(1.0, ece)))


def _regime_calibration_summary(
    regimes: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    min_support: int,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    regimes_arr = np.asarray(regimes).astype(str)
    labels_arr = np.asarray(labels, dtype=float)
    probs_arr = np.asarray(probabilities, dtype=float)
    valid_mask = np.isfinite(probs_arr)
    if regimes_arr.size == 0 or labels_arr.size == 0 or probs_arr.size == 0:
        return out
    for regime in sorted({str(value) for value in regimes_arr}):
        regime_mask = regimes_arr == regime
        chosen = regime_mask & valid_mask
        support = int(np.sum(chosen))
        if support < max(1, int(min_support)):
            continue
        regime_labels = labels_arr[chosen]
        regime_probs = np.clip(probs_arr[chosen], 1e-6, 1.0 - 1e-6)
        brier = float(np.mean((regime_probs - regime_labels) ** 2))
        ece = _expected_calibration_error(regime_labels, regime_probs)
        out[regime] = {
            "support": float(support),
            "brier_score": brier,
            "ece": ece,
        }
    return out


def _evaluate_candidate(
    name: str,
    dataset: Any,
    *,
    seed: int,
    threshold: float,
) -> CandidateMetrics | None:
    import pandas as pd

    if dataset is None or dataset.empty:
        return None
    X = dataset.loc[:, FEATURE_COLUMNS]
    y = dataset["label"].astype(int)
    edge = dataset["realized_edge_bps"].astype(float).to_numpy()
    timestamps = pd.to_datetime(dataset["timestamp"], utc=True)
    label_ts = pd.to_datetime(dataset["label_ts"], utc=True)
    regimes = dataset["regime"].astype(str).to_numpy()
    horizon_days = int(get_env("AI_TRADING_AFTER_HOURS_HORIZON_DAYS", 1, cast=int))
    embargo_days = int(get_env("AI_TRADING_AFTER_HOURS_EMBARGO_DAYS", 1, cast=int))
    fold_gap_days = max(1, horizon_days + embargo_days)
    split_count = int(get_env("AI_TRADING_AFTER_HOURS_CV_SPLITS", 5, cast=int))
    split_count = max(2, min(split_count, max(2, len(dataset) // 80)))
    splitter = PurgedGroupTimeSeriesSplit(
        n_splits=split_count,
        embargo_pct=float(get_env("AI_TRADING_AFTER_HOURS_EMBARGO_PCT", 0.01, cast=float)),
        purge_pct=float(get_env("AI_TRADING_AFTER_HOURS_PURGE_PCT", 0.02, cast=float)),
    )
    split_frame = X.copy()
    split_frame.index = pd.DatetimeIndex(timestamps)
    folds = list(splitter.split(split_frame, y, t1=label_ts))
    if not folds:
        test_size = max(20, len(dataset) // (split_count + 1))
        fallback_folds: list[tuple[np.ndarray, np.ndarray]] = []
        for split_idx in range(split_count):
            test_start = int(len(dataset) * (split_idx + 1) / (split_count + 1))
            test_end = min(len(dataset), test_start + test_size)
            train_end = max(1, test_start - 3)
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            if len(train_idx) >= 60 and len(test_idx) >= 20:
                fallback_folds.append((train_idx, test_idx))
        folds = fallback_folds
    fold_edges: list[float] = []
    fold_turnover: list[float] = []
    fold_hits: list[float] = []
    fold_dd: list[float] = []
    total_support = 0
    oof_probs = np.full(len(dataset), np.nan, dtype=float)
    valid_folds = 0
    for train_idx, test_idx in folds:
        if len(train_idx) < 60 or len(test_idx) < 20:
            continue
        test_labels = label_ts.iloc[test_idx]
        if test_labels.empty:
            continue
        test_start = test_labels.min()
        cutoff = test_start - pd.Timedelta(days=fold_gap_days)
        train_mask = label_ts.iloc[train_idx] <= cutoff
        train_idx_safe = np.asarray(train_idx)[np.asarray(train_mask, dtype=bool)]
        if len(train_idx_safe) < 60:
            continue
        y_train = y.iloc[train_idx_safe]
        y_test = y.iloc[test_idx]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue
        try:
            run_leakage_guards(
                feature_timestamps=timestamps.iloc[test_idx],
                label_timestamps=label_ts.iloc[test_idx],
                train_label_times=label_ts.iloc[train_idx_safe],
                test_label_times=label_ts.iloc[test_idx],
                horizon_days=horizon_days,
                embargo_days=embargo_days,
            )
        except AssertionError as exc:
            logger.warning(
                "AFTER_HOURS_FOLD_SKIPPED_LEAKAGE",
                extra={"model": name, "error": str(exc)},
            )
            continue
        model = _fit_candidate_model(name, seed=seed)
        try:
            model.fit(X.iloc[train_idx_safe], y_train)
        except Exception:
            continue
        probs = _predict_probabilities(model, X.iloc[test_idx])
        if probs.shape[0] != len(test_idx):
            continue
        oof_probs[test_idx] = probs
        selected = probs >= threshold
        support = int(np.sum(selected))
        total_support += support
        if support <= 0:
            fold_edges.append(0.0)
            fold_turnover.append(0.0)
            fold_hits.append(0.0)
            fold_dd.append(0.0)
            valid_folds += 1
            continue
        selected_edge = edge[test_idx][selected]
        fold_edges.append(float(np.mean(selected_edge)))
        fold_turnover.append(float(np.mean(selected)))
        fold_hits.append(float(np.mean(selected_edge > 0)))
        strategy_path = np.where(selected, edge[test_idx], 0.0)
        fold_dd.append(_max_drawdown_bps(strategy_path))
        valid_folds += 1
    if valid_folds <= 0:
        return None
    valid_probs_mask = np.isfinite(oof_probs)
    selected_all = valid_probs_mask & (oof_probs >= threshold)
    regime_metrics = _regime_summary(regimes, selected_all, edge)
    brier_score = 1.0
    y_values_all = y.to_numpy(dtype=float)
    regime_calibration: dict[str, dict[str, float]] = {}
    if np.any(valid_probs_mask):
        y_values = y_values_all[valid_probs_mask]
        probs_values = np.clip(oof_probs[valid_probs_mask], 1e-6, 1.0 - 1e-6)
        if y_values.size and probs_values.size == y_values.size:
            brier_score = float(np.mean((probs_values - y_values) ** 2))
        regime_calibration = _regime_calibration_summary(
            regimes,
            y_values_all,
            oof_probs,
            min_support=int(
                get_env("AI_TRADING_AFTER_HOURS_REGIME_CALIBRATION_MIN_SUPPORT", 25, cast=int)
            ),
        )
    hit_stability = 1.0
    if len(fold_hits) > 1:
        hit_stability = max(0.0, 1.0 - float(pstdev(fold_hits)))
    profitable_fold_count = int(sum(1 for edge_value in fold_edges if edge_value > 0.0))
    profitable_fold_ratio = float(profitable_fold_count) / float(max(1, valid_folds))
    return CandidateMetrics(
        name=name,
        fold_count=valid_folds,
        profitable_fold_count=profitable_fold_count,
        profitable_fold_ratio=profitable_fold_ratio,
        support=total_support,
        mean_expectancy_bps=float(mean(fold_edges)) if fold_edges else 0.0,
        max_drawdown_bps=float(max(fold_dd)) if fold_dd else 0.0,
        turnover_ratio=float(mean(fold_turnover)) if fold_turnover else 0.0,
        mean_hit_rate=float(mean(fold_hits)) if fold_hits else 0.0,
        hit_rate_stability=float(hit_stability),
        regime_metrics=regime_metrics,
        oof_probabilities=oof_probs,
        brier_score=float(brier_score),
        regime_calibration=regime_calibration,
    )


def _score_expectancy_with_drawdown_penalty(
    *,
    expectancy_bps: float,
    max_drawdown_bps: float,
    penalty_per_bps: float,
) -> float:
    return float(expectancy_bps) - (float(max_drawdown_bps) * float(penalty_per_bps))


def _model_selection_weight_bounds() -> dict[str, tuple[float, float]]:
    return {name: (low, high) for name, _env, _default, low, high in _MODEL_SELECTION_WEIGHT_SPECS}


def _model_selection_default_weights() -> dict[str, float]:
    out: dict[str, float] = {}
    for name, env_name, default_value, low, high in _MODEL_SELECTION_WEIGHT_SPECS:
        value = float(get_env(env_name, default_value, cast=float))
        if not math.isfinite(value):
            value = float(default_value)
        out[name] = _clamp(value, low=low, high=high)
    return out


def _normalize_model_selection_weights(
    candidate: Mapping[str, Any] | None,
    *,
    defaults: Mapping[str, float] | None = None,
) -> dict[str, float]:
    base = dict(defaults or _model_selection_default_weights())
    if not isinstance(candidate, Mapping):
        return base
    bounds = _model_selection_weight_bounds()
    for key, (low, high) in bounds.items():
        raw_value = candidate.get(key)
        if raw_value is None:
            continue
        try:
            parsed = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(parsed):
            continue
        base[key] = _clamp(parsed, low=low, high=high)
    return base


def _model_selection_overrides_path() -> Path:
    raw = str(
        get_env(
            "AI_TRADING_AFTER_HOURS_MODEL_SELECTION_OVERRIDES_PATH",
            "runtime/model_selection_overrides.json",
            cast=str,
        )
        or ""
    ).strip()
    if not raw:
        raw = "runtime/model_selection_overrides.json"
    return _resolve_after_hours_output_path(
        raw,
        default_relative="runtime/model_selection_overrides.json",
    )


def _load_model_selection_overrides() -> dict[str, float] | None:
    enabled = bool(
        get_env("AI_TRADING_AFTER_HOURS_MODEL_SELECTION_OVERRIDES_ENABLED", True, cast=bool)
    )
    if not enabled:
        return None
    path = _model_selection_overrides_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        logger.warning(
            "AFTER_HOURS_SELECTION_OVERRIDE_READ_FAILED",
            extra={"path": str(path)},
        )
        return None
    if isinstance(payload, Mapping):
        nested = payload.get("weights")
        if isinstance(nested, Mapping):
            return _normalize_model_selection_weights(nested)
        return _normalize_model_selection_weights(payload)
    return None


def _resolved_model_selection_weights(
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    defaults = _model_selection_default_weights()
    runtime_overrides = _load_model_selection_overrides() if overrides is None else dict(overrides)
    return _normalize_model_selection_weights(runtime_overrides, defaults=defaults)


def _candidate_selection_score_from_snapshot(
    metrics: _CandidateSnapshot,
    *,
    weights: Mapping[str, float],
) -> float:
    drawdown_penalty = float(weights["drawdown_penalty"])
    turnover_penalty = float(weights["turnover_penalty"])
    brier_penalty = float(weights["brier_penalty"])
    stability_weight = float(weights["stability_weight"])
    support_log_weight = float(weights["support_log_weight"])
    profitable_fold_weight = float(weights["profitable_fold_weight"])
    return (
        float(metrics.mean_expectancy_bps)
        - (float(metrics.max_drawdown_bps) * drawdown_penalty)
        - (float(metrics.turnover_ratio) * turnover_penalty)
        - (float(metrics.brier_score) * brier_penalty)
        + (float(metrics.hit_rate_stability) * stability_weight)
        + (float(math.log1p(max(0, int(metrics.support)))) * support_log_weight)
        + (float(metrics.profitable_fold_ratio) * profitable_fold_weight)
    )


def _candidate_selection_score(
    metrics: CandidateMetrics,
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    resolved_weights = (
        _resolved_model_selection_weights()
        if weights is None
        else _normalize_model_selection_weights(weights)
    )
    snapshot = _CandidateSnapshot(
        name=metrics.name,
        mean_expectancy_bps=float(metrics.mean_expectancy_bps),
        max_drawdown_bps=float(metrics.max_drawdown_bps),
        turnover_ratio=float(metrics.turnover_ratio),
        hit_rate_stability=float(metrics.hit_rate_stability),
        brier_score=float(metrics.brier_score),
        support=int(metrics.support),
        profitable_fold_ratio=float(metrics.profitable_fold_ratio),
    )
    return _candidate_selection_score_from_snapshot(snapshot, weights=resolved_weights)


def _coerce_finite_float(raw: Any, *, default: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return float(value)


def _selection_eval_weights() -> dict[str, float]:
    defaults = {
        "drawdown_penalty": float(
            get_env("AI_TRADING_AFTER_HOURS_RETUNE_EVAL_DRAWDOWN_PENALTY", 0.003, cast=float)
        ),
        "turnover_penalty": float(
            get_env("AI_TRADING_AFTER_HOURS_RETUNE_EVAL_TURNOVER_PENALTY", 0.75, cast=float)
        ),
        "brier_penalty": float(
            get_env("AI_TRADING_AFTER_HOURS_RETUNE_EVAL_BRIER_PENALTY", 4.0, cast=float)
        ),
        "stability_weight": float(
            get_env("AI_TRADING_AFTER_HOURS_RETUNE_EVAL_STABILITY_WEIGHT", 0.5, cast=float)
        ),
        "support_log_weight": float(
            get_env("AI_TRADING_AFTER_HOURS_RETUNE_EVAL_SUPPORT_LOG_WEIGHT", 0.05, cast=float)
        ),
        "profitable_fold_weight": float(
            get_env("AI_TRADING_AFTER_HOURS_RETUNE_EVAL_PROFITABLE_FOLD_WEIGHT", 0.5, cast=float)
        ),
    }
    return _normalize_model_selection_weights(defaults, defaults=defaults)


def _selection_utility(
    snapshot: _CandidateSnapshot,
    *,
    eval_weights: Mapping[str, float],
) -> float:
    return _candidate_selection_score_from_snapshot(snapshot, weights=eval_weights)


def _report_candidate_from_payload(payload: Mapping[str, Any]) -> _CandidateSnapshot | None:
    name = str(payload.get("name", "") or "").strip()
    if not name:
        return None
    return _CandidateSnapshot(
        name=name,
        mean_expectancy_bps=_coerce_finite_float(
            payload.get("mean_expectancy_bps"),
            default=0.0,
        ),
        max_drawdown_bps=_coerce_finite_float(
            payload.get("max_drawdown_bps"),
            default=0.0,
        ),
        turnover_ratio=_coerce_finite_float(
            payload.get("turnover_ratio"),
            default=0.0,
        ),
        hit_rate_stability=_coerce_finite_float(
            payload.get("hit_rate_stability"),
            default=0.0,
        ),
        brier_score=_coerce_finite_float(
            payload.get("brier_score"),
            default=1.0,
        ),
        support=max(0, int(_coerce_finite_float(payload.get("support"), default=0.0))),
        profitable_fold_ratio=_coerce_finite_float(
            payload.get("profitable_fold_ratio"),
            default=0.0,
        ),
    )


def _report_snapshot_from_payload(
    payload: Mapping[str, Any],
    *,
    source: str,
) -> _ReportSnapshot | None:
    ts = _parse_ts(payload.get("ts"))
    if ts is None:
        return None
    raw_candidates = payload.get("candidate_metrics")
    candidates: list[_CandidateSnapshot] = []
    if isinstance(raw_candidates, list):
        for item in raw_candidates:
            if not isinstance(item, Mapping):
                continue
            parsed = _report_candidate_from_payload(item)
            if parsed is not None:
                candidates.append(parsed)
    if not candidates:
        return None
    model_name = ""
    model_payload = payload.get("model")
    if isinstance(model_payload, Mapping):
        model_name = str(model_payload.get("name", "") or "").strip()
    if not model_name:
        for item in raw_candidates if isinstance(raw_candidates, list) else []:
            if isinstance(item, Mapping) and bool(item.get("selected")):
                model_name = str(item.get("name", "") or "").strip()
                if model_name:
                    break
    if not model_name:
        model_name = candidates[0].name
    selected = next((item for item in candidates if item.name == model_name), candidates[0])
    metrics_payload = payload.get("metrics")
    metrics = metrics_payload if isinstance(metrics_payload, Mapping) else {}
    return _ReportSnapshot(
        ts=ts,
        source=source,
        model_name=model_name,
        mean_expectancy_bps=_coerce_finite_float(
            metrics.get("mean_expectancy_bps"),
            default=selected.mean_expectancy_bps,
        ),
        hit_rate_stability=_coerce_finite_float(
            metrics.get("hit_rate_stability"),
            default=selected.hit_rate_stability,
        ),
        brier_score=_coerce_finite_float(
            metrics.get("brier_score"),
            default=selected.brier_score,
        ),
        candidates=candidates,
    )


def _recent_report_snapshots(
    *,
    report_dir: Path,
    max_reports: int,
    pending_report: Mapping[str, Any] | None = None,
) -> list[_ReportSnapshot]:
    snapshots: list[_ReportSnapshot] = []
    seen_keys: set[tuple[str, str]] = set()
    if report_dir.exists():
        report_paths = _after_hours_report_paths(report_dir)
        for report_path in report_paths[-max_reports:]:
            try:
                payload = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(payload, Mapping):
                continue
            parsed = _report_snapshot_from_payload(payload, source=str(report_path))
            if parsed is not None:
                dedupe_key = (parsed.ts.isoformat(), str(parsed.model_name))
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                snapshots.append(parsed)
    if isinstance(pending_report, Mapping):
        parsed_pending = _report_snapshot_from_payload(pending_report, source="<pending>")
        if parsed_pending is not None:
            dedupe_key = (parsed_pending.ts.isoformat(), str(parsed_pending.model_name))
            if dedupe_key not in seen_keys:
                snapshots.append(parsed_pending)
    snapshots.sort(key=lambda item: item.ts)
    if len(snapshots) > max_reports:
        return snapshots[-max_reports:]
    return snapshots


def _drift_breach_summary(reports: list[_ReportSnapshot]) -> dict[str, Any]:
    baseline_window = max(
        1,
        int(get_env("AI_TRADING_AFTER_HOURS_RETUNE_BASELINE_WINDOW", 5, cast=int)),
    )
    recent_window = max(
        1,
        int(get_env("AI_TRADING_AFTER_HOURS_RETUNE_RECENT_WINDOW", 2, cast=int)),
    )
    min_reports = max(
        3,
        int(get_env("AI_TRADING_AFTER_HOURS_RETUNE_MIN_REPORTS", 8, cast=int)),
    )
    summary: dict[str, Any] = {
        "ready": False,
        "breached": False,
        "report_count": int(len(reports)),
        "baseline_window": int(baseline_window),
        "recent_window": int(recent_window),
        "min_reports": int(min_reports),
        "gates": {},
    }
    if len(reports) < min_reports or len(reports) <= recent_window:
        summary["reason"] = "insufficient_reports"
        return summary
    baseline_count = min(baseline_window, len(reports) - recent_window)
    if baseline_count <= 0:
        summary["reason"] = "insufficient_baseline_window"
        return summary
    baseline_reports = reports[-(recent_window + baseline_count) : -recent_window]
    recent_reports = reports[-recent_window:]
    baseline_brier = float(mean(item.brier_score for item in baseline_reports))
    recent_brier = float(mean(item.brier_score for item in recent_reports))
    baseline_expectancy = float(mean(item.mean_expectancy_bps for item in baseline_reports))
    recent_expectancy = float(mean(item.mean_expectancy_bps for item in recent_reports))
    baseline_stability = float(mean(item.hit_rate_stability for item in baseline_reports))
    recent_stability = float(mean(item.hit_rate_stability for item in recent_reports))
    brier_drift_pct = _clamp(
        float(get_env("AI_TRADING_AFTER_HOURS_RETUNE_BRIER_DRIFT_PCT", 0.15, cast=float)),
        low=0.0,
        high=10.0,
    )
    brier_abs_threshold = _clamp(
        float(
            get_env("AI_TRADING_AFTER_HOURS_RETUNE_BRIER_ABS_THRESHOLD", 0.25, cast=float)
        ),
        low=0.0,
        high=1.0,
    )
    expectancy_drop_bps = _clamp(
        float(get_env("AI_TRADING_AFTER_HOURS_RETUNE_EXPECTANCY_DROP_BPS", 5.0, cast=float)),
        low=0.0,
        high=500.0,
    )
    stability_drop = _clamp(
        float(get_env("AI_TRADING_AFTER_HOURS_RETUNE_STABILITY_DROP", 0.08, cast=float)),
        low=0.0,
        high=1.0,
    )
    brier_gate = (
        recent_brier >= brier_abs_threshold
        if baseline_brier <= 0.0
        else recent_brier > (baseline_brier * (1.0 + brier_drift_pct))
    )
    expectancy_gate = recent_expectancy < (baseline_expectancy - expectancy_drop_bps)
    stability_gate = recent_stability < (baseline_stability - stability_drop)
    gates = {
        "brier_drift": bool(brier_gate),
        "expectancy_drop": bool(expectancy_gate),
        "stability_drop": bool(stability_gate),
    }
    summary.update(
        {
            "ready": True,
            "breached": any(gates.values()),
            "gates": gates,
            "baseline": {
                "brier_score": baseline_brier,
                "mean_expectancy_bps": baseline_expectancy,
                "hit_rate_stability": baseline_stability,
                "sample": len(baseline_reports),
            },
            "recent": {
                "brier_score": recent_brier,
                "mean_expectancy_bps": recent_expectancy,
                "hit_rate_stability": recent_stability,
                "sample": len(recent_reports),
            },
            "thresholds": {
                "brier_drift_pct": brier_drift_pct,
                "brier_abs_threshold": brier_abs_threshold,
                "expectancy_drop_bps": expectancy_drop_bps,
                "stability_drop": stability_drop,
            },
        }
    )
    return summary


def _evaluate_weight_configuration(
    reports: list[_ReportSnapshot],
    *,
    selection_weights: Mapping[str, float],
    eval_weights: Mapping[str, float],
    reference_weights: Mapping[str, float],
    regularization: float,
) -> float:
    utilities: list[float] = []
    for report in reports:
        if not report.candidates:
            continue
        selected = max(
            report.candidates,
            key=lambda item: (
                _candidate_selection_score_from_snapshot(item, weights=selection_weights),
                item.mean_expectancy_bps,
                -item.max_drawdown_bps,
                item.support,
            ),
        )
        utilities.append(_selection_utility(selected, eval_weights=eval_weights))
    if not utilities:
        return -float("inf")
    deviation = sum(
        abs(float(selection_weights[key]) - float(reference_weights[key]))
        for key in reference_weights
    )
    return float(mean(utilities)) - (float(regularization) * float(deviation))


def _retune_selection_weights(
    reports: list[_ReportSnapshot],
    *,
    current_weights: Mapping[str, float],
) -> dict[str, Any]:
    normalized_current = _normalize_model_selection_weights(current_weights)
    eval_weights = _selection_eval_weights()
    regularization = max(
        0.0,
        float(get_env("AI_TRADING_AFTER_HOURS_RETUNE_WEIGHT_REGULARIZATION", 0.2, cast=float)),
    )
    search_rounds = max(
        1,
        int(get_env("AI_TRADING_AFTER_HOURS_RETUNE_SEARCH_ROUNDS", 4, cast=int)),
    )
    factor_values = [
        value
        for value in _parse_float_grid(
            str(
                get_env(
                    "AI_TRADING_AFTER_HOURS_RETUNE_SEARCH_FACTORS",
                    "0.5,0.75,1.0,1.25,1.5,2.0",
                    cast=str,
                )
                or ""
            ),
            fallback=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
        )
        if value > 0.0 and math.isfinite(value)
    ]
    if not factor_values:
        factor_values = [1.0]
    bounds = _model_selection_weight_bounds()
    baseline_score = _evaluate_weight_configuration(
        reports,
        selection_weights=normalized_current,
        eval_weights=eval_weights,
        reference_weights=normalized_current,
        regularization=regularization,
    )
    best_weights = dict(normalized_current)
    best_score = float(baseline_score)
    tolerance = float(get_env("AI_TRADING_AFTER_HOURS_RETUNE_SEARCH_TOLERANCE", 1e-9, cast=float))
    for _ in range(search_rounds):
        improved = False
        for key, (low, high) in bounds.items():
            base_value = float(best_weights[key])
            candidate_values = {
                _clamp(base_value * factor, low=low, high=high) for factor in factor_values
            }
            candidate_values.add(base_value)
            local_best_score = best_score
            local_best_weights = dict(best_weights)
            for candidate_value in sorted(candidate_values):
                trial = dict(best_weights)
                trial[key] = float(candidate_value)
                trial_score = _evaluate_weight_configuration(
                    reports,
                    selection_weights=trial,
                    eval_weights=eval_weights,
                    reference_weights=normalized_current,
                    regularization=regularization,
                )
                if trial_score > (local_best_score + tolerance):
                    local_best_score = float(trial_score)
                    local_best_weights = trial
            if local_best_score > (best_score + tolerance):
                best_score = local_best_score
                best_weights = local_best_weights
                improved = True
        if not improved:
            break
    min_improvement = max(
        0.0,
        float(get_env("AI_TRADING_AFTER_HOURS_RETUNE_MIN_UTILITY_IMPROVEMENT", 0.02, cast=float)),
    )
    max_relative_change = max(
        (
            abs(best_weights[key] - normalized_current[key]) / max(abs(normalized_current[key]), 1e-9)
            for key in normalized_current
        ),
        default=0.0,
    )
    min_weight_change_pct = _clamp(
        float(get_env("AI_TRADING_AFTER_HOURS_RETUNE_MIN_WEIGHT_CHANGE_PCT", 0.05, cast=float)),
        low=0.0,
        high=10.0,
    )
    improved_enough = best_score >= (baseline_score + min_improvement)
    moved_enough = max_relative_change >= min_weight_change_pct
    retuned = bool(improved_enough and moved_enough)
    return {
        "retuned": retuned,
        "baseline_score": float(baseline_score),
        "best_score": float(best_score),
        "min_improvement": float(min_improvement),
        "max_relative_change": float(max_relative_change),
        "min_weight_change_pct": float(min_weight_change_pct),
        "weights": dict(best_weights if retuned else normalized_current),
        "search_rounds": int(search_rounds),
        "search_factors": list(factor_values),
        "eval_weights": dict(eval_weights),
    }


def _write_model_selection_overrides(
    *,
    now_utc: datetime,
    weights: Mapping[str, float],
    drift_summary: Mapping[str, Any],
) -> Path:
    requested_path = _model_selection_overrides_path()
    fallback_path = (paths.DATA_DIR / "runtime/model_selection_overrides.json").resolve()
    writable_dir = _resolve_writable_output_dir(
        requested=requested_path.parent,
        fallback=fallback_path.parent,
        event_name="AFTER_HOURS_SELECTION_OVERRIDE_PATH_FALLBACK",
    )
    if writable_dir == requested_path.parent:
        target_path = requested_path
    elif writable_dir == fallback_path.parent:
        target_path = fallback_path
    else:
        target_path = writable_dir / requested_path.name
    payload = {
        "updated_at": now_utc.isoformat(),
        "weights": _normalize_model_selection_weights(weights),
        "drift_summary": dict(drift_summary),
    }
    return _write_json(target_path, payload)


def _load_existing_selection_overrides_payload() -> Mapping[str, Any] | None:
    path = _model_selection_overrides_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if isinstance(payload, Mapping):
        return payload
    return None


def _retune_cooldown_allows(*, now_utc: datetime) -> dict[str, Any]:
    min_hours = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_AFTER_HOURS_RETUNE_MIN_HOURS_BETWEEN_UPDATES",
                24.0,
                cast=float,
            )
        ),
    )
    payload = _load_existing_selection_overrides_payload()
    if payload is None:
        return {"allowed": True, "min_hours": float(min_hours), "hours_since_last": None}
    updated_raw = payload.get("updated_at")
    updated_at = _parse_ts(updated_raw)
    if updated_at is None:
        return {"allowed": True, "min_hours": float(min_hours), "hours_since_last": None}
    elapsed_hours = max(0.0, (now_utc - updated_at).total_seconds() / 3600.0)
    return {
        "allowed": bool(elapsed_hours >= min_hours),
        "min_hours": float(min_hours),
        "hours_since_last": float(elapsed_hours),
        "updated_at": updated_at.isoformat(),
    }


def _apply_selection_weight_safety_envelope(
    *,
    current_weights: Mapping[str, float],
    candidate_weights: Mapping[str, float],
) -> dict[str, Any]:
    normalized_current = _normalize_model_selection_weights(current_weights)
    normalized_candidate = _normalize_model_selection_weights(candidate_weights)
    max_abs_delta = _clamp(
        float(get_env("AI_TRADING_AFTER_HOURS_RETUNE_MAX_ABS_WEIGHT_DELTA", 0.5, cast=float)),
        low=0.0,
        high=5.0,
    )
    max_rel_delta = _clamp(
        float(get_env("AI_TRADING_AFTER_HOURS_RETUNE_MAX_RELATIVE_WEIGHT_DELTA", 0.6, cast=float)),
        low=0.0,
        high=10.0,
    )
    bounded: dict[str, float] = {}
    clamps: dict[str, dict[str, float]] = {}
    for key, current_value in normalized_current.items():
        candidate_value = float(normalized_candidate.get(key, current_value))
        abs_low = float(current_value) - max_abs_delta
        abs_high = float(current_value) + max_abs_delta
        rel_span = max(abs(float(current_value)) * max_rel_delta, 1e-9)
        rel_low = float(current_value) - rel_span
        rel_high = float(current_value) + rel_span
        bounded_low = max(abs_low, rel_low)
        bounded_high = min(abs_high, rel_high)
        safe_value = _clamp(candidate_value, low=bounded_low, high=bounded_high)
        bounded[key] = float(safe_value)
        if abs(safe_value - candidate_value) > 1e-12:
            clamps[key] = {
                "from": float(candidate_value),
                "to": float(safe_value),
                "low": float(bounded_low),
                "high": float(bounded_high),
            }
    return {
        "weights": _normalize_model_selection_weights(bounded, defaults=normalized_current),
        "clamped": bool(clamps),
        "clamps": clamps,
        "limits": {
            "max_abs_delta": float(max_abs_delta),
            "max_relative_delta": float(max_rel_delta),
        },
    }


def _maybe_retune_model_selection_weights(
    *,
    now_utc: datetime,
    report_dir: Path,
    pending_report: Mapping[str, Any] | None,
    current_weights: Mapping[str, float],
) -> dict[str, Any]:
    enabled = bool(get_env("AI_TRADING_AFTER_HOURS_RETUNE_ENABLED", True, cast=bool))
    summary: dict[str, Any] = {
        "enabled": enabled,
        "retuned": False,
        "drift_summary": {},
    }
    if not enabled:
        summary["reason"] = "disabled"
        return summary
    max_reports = max(
        3,
        int(get_env("AI_TRADING_AFTER_HOURS_RETUNE_MAX_REPORTS", 60, cast=int)),
    )
    report_snapshots = _recent_report_snapshots(
        report_dir=report_dir,
        max_reports=max_reports,
        pending_report=pending_report,
    )
    drift_summary = _drift_breach_summary(report_snapshots)
    summary["drift_summary"] = drift_summary
    summary["report_count"] = len(report_snapshots)
    if not bool(drift_summary.get("ready", False)):
        summary["reason"] = str(drift_summary.get("reason", "not_ready"))
        return summary
    if not bool(drift_summary.get("breached", False)):
        summary["reason"] = "no_drift_breach"
        return summary
    tuning = _retune_selection_weights(
        report_snapshots,
        current_weights=current_weights,
    )
    summary["tuning"] = tuning
    if not bool(tuning.get("retuned", False)):
        summary["reason"] = "no_material_improvement"
        return summary
    cooldown = _retune_cooldown_allows(now_utc=now_utc)
    summary["cooldown"] = cooldown
    if not bool(cooldown.get("allowed", True)):
        summary["reason"] = "cooldown_active"
        return summary
    safety = _apply_selection_weight_safety_envelope(
        current_weights=current_weights,
        candidate_weights=dict(tuning["weights"]),
    )
    summary["safety_envelope"] = safety
    override_path = _write_model_selection_overrides(
        now_utc=now_utc,
        weights=safety["weights"],
        drift_summary=drift_summary,
    )
    summary.update(
        {
            "retuned": True,
            "reason": "drift_breach",
            "override_path": str(override_path),
            "weights": dict(safety["weights"]),
        }
    )
    return summary


def _threshold_by_regime(
    dataset: Any,
    probabilities: np.ndarray,
    *,
    default_threshold: float,
) -> dict[str, float]:
    regimes = dataset["regime"].astype(str).to_numpy()
    edge = dataset["realized_edge_bps"].astype(float).to_numpy()
    out: dict[str, float] = {}
    min_support = int(get_env("AI_TRADING_AFTER_HOURS_MIN_THRESHOLD_SUPPORT", 25, cast=int))
    drawdown_penalty_per_bps = float(
        get_env("AI_TRADING_AFTER_HOURS_THRESHOLD_DRAWDOWN_PENALTY", 0.003, cast=float)
    )
    drawdown_cap_bps = float(
        get_env(
            "AI_TRADING_AFTER_HOURS_THRESHOLD_MAX_DRAWDOWN_BPS",
            get_env("AI_TRADING_EDGE_TARGET_MAX_DRAWDOWN_BPS", 300.0, cast=float),
            cast=float,
        )
    )
    min_expectancy_bps = float(
        get_env("AI_TRADING_AFTER_HOURS_THRESHOLD_MIN_EXPECTANCY_BPS", 0.0, cast=float)
    )
    for regime in sorted({str(v) for v in regimes}):
        regime_mask = regimes == regime
        valid_mask = regime_mask & np.isfinite(probabilities)
        if int(np.sum(valid_mask)) < min_support:
            out[regime] = float(default_threshold)
            continue
        best_threshold = float(default_threshold)
        best_feasible_key = (-float("inf"), -float("inf"), -1)
        best_infeasible_key = (float("inf"), float("inf"), float("inf"))
        best_infeasible_threshold = float(default_threshold)
        feasible_found = False
        for threshold in _THRESHOLD_GRID:
            selected = valid_mask & (probabilities >= threshold)
            support = int(np.sum(selected))
            if support < min_support:
                continue
            selected_edges = edge[selected]
            expectancy_bps = (
                float(np.mean(selected_edges)) if selected_edges.size > 0 else 0.0
            )
            strategy_path = np.where(selected, edge, 0.0)
            drawdown_bps = float(_max_drawdown_bps(strategy_path))
            score = _score_expectancy_with_drawdown_penalty(
                expectancy_bps=expectancy_bps,
                max_drawdown_bps=drawdown_bps,
                penalty_per_bps=drawdown_penalty_per_bps,
            )
            if drawdown_bps <= drawdown_cap_bps:
                if expectancy_bps < min_expectancy_bps:
                    continue
                feasible_key = (
                    score,
                    expectancy_bps,
                    support,
                )
                if feasible_key > best_feasible_key:
                    best_feasible_key = feasible_key
                    best_threshold = float(threshold)
                    feasible_found = True
            else:
                infeasible_key = (
                    drawdown_bps,
                    -expectancy_bps,
                    -float(support),
                )
                if infeasible_key < best_infeasible_key:
                    best_infeasible_key = infeasible_key
                    best_infeasible_threshold = float(threshold)
        out[regime] = best_threshold if feasible_found else best_infeasible_threshold
    return out


def _apply_regime_calibration_threshold_adjustment(
    *,
    thresholds_by_regime: Mapping[str, float],
    regime_calibration: Mapping[str, Mapping[str, float]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    adjusted = {str(key): float(value) for key, value in thresholds_by_regime.items()}
    adjustments: dict[str, dict[str, float]] = {}
    if not bool(
        get_env(
            "AI_TRADING_AFTER_HOURS_REGIME_CALIBRATION_THRESHOLD_ADJUST_ENABLED",
            True,
            cast=bool,
        )
    ):
        return adjusted, adjustments
    ece_warn = _clamp(
        float(get_env("AI_TRADING_AFTER_HOURS_REGIME_ECE_WARN", 0.08, cast=float)),
        low=0.0,
        high=1.0,
    )
    ece_critical = _clamp(
        float(get_env("AI_TRADING_AFTER_HOURS_REGIME_ECE_CRITICAL", 0.14, cast=float)),
        low=ece_warn,
        high=1.0,
    )
    max_bump = _clamp(
        float(get_env("AI_TRADING_AFTER_HOURS_REGIME_THRESHOLD_MAX_BUMP", 0.08, cast=float)),
        low=0.0,
        high=0.4,
    )
    base_step = _clamp(
        float(get_env("AI_TRADING_AFTER_HOURS_REGIME_THRESHOLD_BUMP_STEP", 0.02, cast=float)),
        low=0.0,
        high=max_bump if max_bump > 0.0 else 0.0,
    )
    if base_step <= 0.0 or max_bump <= 0.0:
        return adjusted, adjustments
    for regime, metrics in regime_calibration.items():
        base_threshold = adjusted.get(str(regime))
        if base_threshold is None:
            continue
        try:
            ece = float(metrics.get("ece", 0.0))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(ece) or ece <= ece_warn:
            continue
        if ece_critical > ece_warn:
            severity = (ece - ece_warn) / (ece_critical - ece_warn)
        else:
            severity = 1.0
        severity = _clamp(float(severity), low=0.0, high=3.0)
        bump = min(max_bump, base_step * (1.0 + severity))
        new_threshold = _clamp(float(base_threshold) + float(bump), low=0.05, high=0.95)
        adjusted[str(regime)] = float(new_threshold)
        adjustments[str(regime)] = {
            "ece": float(ece),
            "threshold_before": float(base_threshold),
            "threshold_after": float(new_threshold),
            "threshold_bump": float(new_threshold - float(base_threshold)),
        }
    return adjusted, adjustments


def _parse_float_grid(raw: str, *, fallback: tuple[float, ...]) -> list[float]:
    tokens = [token.strip() for token in str(raw).split(",") if token.strip()]
    parsed: list[float] = []
    for token in tokens:
        try:
            parsed.append(float(token))
        except (TypeError, ValueError):
            continue
    if parsed:
        return parsed
    return list(fallback)


def _threshold_metrics_snapshot(
    dataset: Any,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    edge = dataset["realized_edge_bps"].astype(float).to_numpy()
    regimes = dataset["regime"].astype(str).to_numpy()
    valid_mask = np.isfinite(probabilities)
    selected = valid_mask & (probabilities >= threshold)
    support = int(np.sum(selected))
    selected_edges = edge[selected] if support > 0 else np.asarray([], dtype=float)
    strategy_path = np.where(selected, edge, 0.0)
    return {
        "threshold": float(threshold),
        "support": support,
        "mean_expectancy_bps": (
            float(np.mean(selected_edges)) if selected_edges.size > 0 else 0.0
        ),
        "max_drawdown_bps": float(_max_drawdown_bps(strategy_path)),
        "turnover_ratio": float(np.mean(selected)) if selected.size > 0 else 0.0,
        "mean_hit_rate": (
            float(np.mean(selected_edges > 0)) if selected_edges.size > 0 else 0.0
        ),
        "regime_metrics": _regime_summary(regimes, selected, edge),
    }


def _run_sensitivity_sweep(
    dataset: Any,
    probabilities: np.ndarray,
    *,
    default_threshold: float,
    targets: EdgeTargets,
) -> dict[str, Any]:
    enabled = bool(
        get_env("AI_TRADING_AFTER_HOURS_SENSITIVITY_SWEEP_ENABLED", True, cast=bool)
    )
    if not enabled:
        return {
            "enabled": False,
            "gate": True,
            "summary": {"reason": "disabled"},
            "scenarios": [],
        }

    deltas = _parse_float_grid(
        str(
            get_env(
                "AI_TRADING_AFTER_HOURS_SENSITIVITY_THRESHOLD_DELTAS",
                "-0.04,-0.02,0.0,0.02,0.04",
            )
            or ""
        ),
        fallback=(-0.04, -0.02, 0.0, 0.02, 0.04),
    )
    thresholds = sorted(
        {
            round(_clamp(default_threshold + delta, low=0.05, high=0.95), 4)
            for delta in deltas
        }
        | {round(_clamp(default_threshold, low=0.05, high=0.95), 4)}
    )
    min_support = int(get_env("AI_TRADING_AFTER_HOURS_MIN_THRESHOLD_SUPPORT", 25, cast=int))
    scenarios = [
        _threshold_metrics_snapshot(
            dataset,
            probabilities,
            threshold=threshold,
        )
        for threshold in thresholds
    ]

    valid_scenarios = [item for item in scenarios if int(item["support"]) >= min_support]
    if not valid_scenarios:
        return {
            "enabled": True,
            "gate": False,
            "summary": {
                "reason": "insufficient_support",
                "min_support": min_support,
                "valid_scenarios": 0,
                "total_scenarios": len(scenarios),
            },
            "scenarios": scenarios,
        }

    pass_count = 0
    expectancies: list[float] = []
    for scenario in valid_scenarios:
        expectancy = float(scenario["mean_expectancy_bps"])
        drawdown = float(scenario["max_drawdown_bps"])
        turnover = float(scenario["turnover_ratio"])
        gates = {
            "expectancy": expectancy >= targets.min_expectancy_bps,
            "drawdown": drawdown <= targets.max_drawdown_bps,
            "turnover": turnover <= targets.max_turnover_ratio,
        }
        scenario["edge_gates"] = gates
        scenario["edge_pass"] = bool(all(gates.values()))
        if scenario["edge_pass"]:
            pass_count += 1
        expectancies.append(expectancy)

    expectancy_std_bps = float(pstdev(expectancies)) if len(expectancies) > 1 else 0.0
    min_expectancy = float(min(expectancies))
    pass_ratio = float(pass_count) / float(max(1, len(valid_scenarios)))

    max_expectancy_std_bps = float(
        get_env("AI_TRADING_AFTER_HOURS_SWEEP_MAX_EXPECTANCY_STD_BPS", 6.0, cast=float)
    )
    min_scenario_expectancy_bps = float(
        get_env(
            "AI_TRADING_AFTER_HOURS_SWEEP_MIN_SCENARIO_EXPECTANCY_BPS",
            targets.min_expectancy_bps - 2.0,
            cast=float,
        )
    )
    min_pass_ratio = _clamp(
        float(get_env("AI_TRADING_AFTER_HOURS_SWEEP_MIN_PASS_RATIO", 0.6, cast=float)),
        low=0.0,
        high=1.0,
    )
    gate = bool(
        expectancy_std_bps <= max_expectancy_std_bps
        and min_expectancy >= min_scenario_expectancy_bps
        and pass_ratio >= min_pass_ratio
    )
    return {
        "enabled": True,
        "gate": gate,
        "summary": {
            "valid_scenarios": len(valid_scenarios),
            "total_scenarios": len(scenarios),
            "min_support": min_support,
            "expectancy_std_bps": expectancy_std_bps,
            "max_expectancy_std_bps": max_expectancy_std_bps,
            "min_scenario_expectancy_bps": min_scenario_expectancy_bps,
            "observed_min_expectancy_bps": min_expectancy,
            "min_pass_ratio": min_pass_ratio,
            "observed_pass_ratio": pass_ratio,
        },
        "scenarios": scenarios,
    }


def _edge_targets() -> EdgeTargets:
    return EdgeTargets(
        min_expectancy_bps=float(
            get_env("AI_TRADING_EDGE_TARGET_EXPECTANCY_BPS", 1.0, cast=float)
        ),
        max_drawdown_bps=float(
            get_env("AI_TRADING_EDGE_TARGET_MAX_DRAWDOWN_BPS", 300.0, cast=float)
        ),
        max_turnover_ratio=float(
            get_env("AI_TRADING_EDGE_TARGET_MAX_TURNOVER_RATIO", 0.45, cast=float)
        ),
        min_hit_rate_stability=float(
            get_env("AI_TRADING_EDGE_TARGET_MIN_HIT_RATE_STABILITY", 0.45, cast=float)
        ),
    )


def _meets_edge_targets(metrics: CandidateMetrics, targets: EdgeTargets) -> dict[str, bool]:
    return {
        "expectancy": metrics.mean_expectancy_bps >= targets.min_expectancy_bps,
        "drawdown": metrics.max_drawdown_bps <= targets.max_drawdown_bps,
        "turnover": metrics.turnover_ratio <= targets.max_turnover_ratio,
        "stability": metrics.hit_rate_stability >= targets.min_hit_rate_stability,
    }


def _promotion_policy_name() -> str:
    raw = str(get_env("AI_TRADING_AFTER_HOURS_PROMOTION_POLICY", "strict", cast=str) or "")
    normalized = raw.strip().lower()
    if normalized in {"legacy", "strict"}:
        return normalized
    return "strict"


def _promotion_score(*, expectancy_bps: float, max_drawdown_bps: float) -> float:
    drawdown_penalty = float(
        get_env("AI_TRADING_AFTER_HOURS_PROMOTION_DRAWDOWN_PENALTY", 0.003, cast=float)
    )
    return _score_expectancy_with_drawdown_penalty(
        expectancy_bps=expectancy_bps,
        max_drawdown_bps=max_drawdown_bps,
        penalty_per_bps=drawdown_penalty,
    )


def _after_hours_report_paths(report_dir: Path) -> list[Path]:
    if not report_dir.exists():
        return []
    return sorted(report_dir.glob("after_hours_training_*.json"))


def _prior_metrics_from_report_payload(
    payload: Mapping[str, Any],
    *,
    report_path: Path,
) -> dict[str, Any] | None:
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    try:
        mean_expectancy_bps = float(metrics.get("mean_expectancy_bps"))
        max_drawdown_bps = float(metrics.get("max_drawdown_bps"))
    except (TypeError, ValueError):
        return None
    model_payload = payload.get("model")
    model_id = None
    governance_status = None
    if isinstance(model_payload, Mapping):
        model_id = model_payload.get("model_id")
        governance_status = model_payload.get("governance_status")
    return {
        "report_path": str(report_path),
        "model_id": model_id,
        "governance_status": governance_status,
        "mean_expectancy_bps": mean_expectancy_bps,
        "max_drawdown_bps": max_drawdown_bps,
        "mean_hit_rate": float(metrics.get("mean_hit_rate", 0.0) or 0.0),
        "hit_rate_stability": float(metrics.get("hit_rate_stability", 0.0) or 0.0),
        "support": int(metrics.get("support", 0) or 0),
        "fold_count": int(metrics.get("fold_count", 0) or 0),
        "profitable_fold_count": int(metrics.get("profitable_fold_count", 0) or 0),
        "profitable_fold_ratio": float(metrics.get("profitable_fold_ratio", 0.0) or 0.0),
        "regime_calibration": payload.get("regime_calibration"),
        "ts": payload.get("ts"),
    }


def _report_is_production(payload: Mapping[str, Any]) -> bool:
    model_payload = payload.get("model")
    governance_status = (
        str(model_payload.get("governance_status", "") or "").strip().lower()
        if isinstance(model_payload, Mapping)
        else ""
    )
    if governance_status == "production":
        return True
    promotion_payload = payload.get("promotion")
    promotion_status = (
        str(promotion_payload.get("status", "") or "").strip().lower()
        if isinstance(promotion_payload, Mapping)
        else ""
    )
    return promotion_status == "production"


def _load_prior_model_metrics(*, report_dir: Path) -> dict[str, Any] | None:
    report_candidates: list[dict[str, Any]] = []
    for report_path in _after_hours_report_paths(report_dir):
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if not isinstance(payload, Mapping):
            continue
        parsed = _prior_metrics_from_report_payload(payload, report_path=report_path)
        if parsed is None:
            continue
        parsed["is_production"] = _report_is_production(payload)
        parsed["parsed_ts"] = _parse_ts(parsed.get("ts"))
        report_candidates.append(parsed)
    if not report_candidates:
        return None
    report_candidates.sort(
        key=lambda item: (
            item.get("parsed_ts") or datetime.min.replace(tzinfo=UTC),
            str(item.get("report_path", "")),
        )
    )
    latest = report_candidates[-1]
    latest_model_id = str(latest.get("model_id") or "")

    production_distinct = [
        item
        for item in report_candidates
        if bool(item.get("is_production", False))
        and str(item.get("model_id") or "") != latest_model_id
    ]
    if production_distinct:
        chosen = production_distinct[-1]
    else:
        distinct_any = [
            item
            for item in report_candidates
            if str(item.get("model_id") or "") != latest_model_id
        ]
        chosen = distinct_any[-1] if distinct_any else latest
    result = dict(chosen)
    result.pop("is_production", None)
    result.pop("parsed_ts", None)
    return result


def _promotion_gate_bundle(
    *,
    best: CandidateMetrics,
    rows: int,
    edge_gates: Mapping[str, bool],
    prior_metrics: Mapping[str, Any] | None = None,
    additional_gates: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    policy = _promotion_policy_name()
    strict_gates: dict[str, bool] = {}
    prior_model_comparison: dict[str, Any] = {
        "required": False,
        "available": False,
        "gate": True,
    }
    if policy == "strict":
        min_rows = max(
            1, int(get_env("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_ROWS", 800, cast=int))
        )
        min_support = max(
            1, int(get_env("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_SUPPORT", 80, cast=int))
        )
        min_folds = max(
            1, int(get_env("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_FOLDS", 4, cast=int))
        )
        min_hit_rate = float(
            get_env("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_HIT_RATE", 0.52, cast=float)
        )
        min_profitable_folds = max(
            0,
            int(
                get_env(
                    "AI_TRADING_AFTER_HOURS_PROMOTION_MIN_PROFITABLE_FOLDS",
                    3,
                    cast=int,
                )
            ),
        )
        min_profitable_fold_ratio = _clamp(
            float(
                get_env(
                    "AI_TRADING_AFTER_HOURS_PROMOTION_MIN_PROFITABLE_FOLD_RATIO",
                    0.5,
                    cast=float,
                )
            ),
            low=0.0,
            high=1.0,
        )
        require_prior_improvement = bool(
            get_env("AI_TRADING_AFTER_HOURS_PROMOTION_REQUIRE_PRIOR_IMPROVEMENT", True, cast=bool)
        )
        min_score_margin = float(
            get_env("AI_TRADING_AFTER_HOURS_PROMOTION_MIN_SCORE_MARGIN", 0.1, cast=float)
        )
        policy_min_oos_samples = max(
            1,
            int(
                get_env(
                    "AI_TRADING_POLICY_PROMOTION_MIN_OOS_SAMPLES",
                    min_support,
                    cast=int,
                )
            ),
        )
        policy_min_oos_net_bps = float(
            get_env(
                "AI_TRADING_POLICY_PROMOTION_MIN_OOS_NET_BPS",
                0.0,
                cast=float,
            )
        )
        candidate_score = _promotion_score(
            expectancy_bps=best.mean_expectancy_bps,
            max_drawdown_bps=best.max_drawdown_bps,
        )
        prior_gate = True
        prior_model_comparison = {
            "required": require_prior_improvement,
            "available": False,
            "gate": True,
            "margin_required": min_score_margin,
            "candidate_score": candidate_score,
        }
        if require_prior_improvement and prior_metrics is not None:
            try:
                prior_expectancy = float(prior_metrics.get("mean_expectancy_bps"))
                prior_drawdown = float(prior_metrics.get("max_drawdown_bps"))
            except (TypeError, ValueError):
                prior_gate = True
                prior_model_comparison["reason"] = "invalid_prior_metrics"
            else:
                prior_score = _promotion_score(
                    expectancy_bps=prior_expectancy,
                    max_drawdown_bps=prior_drawdown,
                )
                score_delta = candidate_score - prior_score
                prior_gate = score_delta >= min_score_margin
                prior_model_comparison.update(
                    {
                        "available": True,
                        "prior_score": prior_score,
                        "score_delta": score_delta,
                        "prior_report_path": prior_metrics.get("report_path"),
                        "prior_model_id": prior_metrics.get("model_id"),
                        "prior_governance_status": prior_metrics.get("governance_status"),
                    }
                )
        elif require_prior_improvement:
            prior_model_comparison["reason"] = "no_prior_metrics"
        prior_model_comparison["gate"] = bool(prior_gate)
        strict_gates = {
            "rows": int(rows) >= min_rows,
            "support": int(best.support) >= min_support,
            "fold_count": int(best.fold_count) >= min_folds,
            "hit_rate": float(best.mean_hit_rate) >= min_hit_rate,
            "profitable_folds": int(best.profitable_fold_count) >= min_profitable_folds,
            "profitable_fold_ratio": (
                float(best.profitable_fold_ratio) >= min_profitable_fold_ratio
            ),
            "policy_oos_samples": int(best.support) >= policy_min_oos_samples,
            "policy_oos_net": float(best.mean_expectancy_bps) >= policy_min_oos_net_bps,
            "prior_model_improvement": bool(prior_gate),
        }
    combined_gates: dict[str, bool] = {str(k): bool(v) for k, v in dict(edge_gates).items()}
    combined_gates.update(strict_gates)
    normalized_additional_gates: dict[str, bool] = {}
    if additional_gates:
        normalized_additional_gates = {
            str(key): bool(value) for key, value in dict(additional_gates).items()
        }
        combined_gates.update(normalized_additional_gates)
    require_sensitivity = bool(
        get_env("AI_TRADING_AFTER_HOURS_PROMOTION_REQUIRE_SENSITIVITY", True, cast=bool)
    )
    if not require_sensitivity and "sensitivity" in combined_gates:
        combined_gates["sensitivity"] = True
    gate_passed = all(combined_gates.values())
    auto_promote = bool(get_env("AI_TRADING_AFTER_HOURS_AUTO_PROMOTE", False, cast=bool))
    status = "production" if (auto_promote and gate_passed) else "shadow"
    return {
        "policy": policy,
        "auto_promote": auto_promote,
        "require_sensitivity": require_sensitivity,
        "edge_gates": {str(k): bool(v) for k, v in dict(edge_gates).items()},
        "strict_gates": strict_gates,
        "additional_gates": normalized_additional_gates,
        "prior_model_comparison": prior_model_comparison,
        "combined_gates": combined_gates,
        "gate_passed": gate_passed,
        "status": status,
    }


def _runtime_performance_go_no_go_gate() -> dict[str, Any]:
    """Evaluate runtime realized-performance go/no-go criteria for promotion."""

    enabled = bool(
        get_env(
            "AI_TRADING_AFTER_HOURS_PROMOTION_RUNTIME_GONOGO_ENABLED",
            False,
            cast=bool,
        )
    )
    if not enabled:
        return {
            "enabled": False,
            "gate_passed": True,
            "reason": "disabled",
            "checks": {},
            "failed_checks": [],
            "thresholds": {},
            "observed": {},
        }

    trade_history_configured = str(
        get_env(
            "AI_TRADING_RUNTIME_PERF_TRADE_HISTORY_PATH",
            "runtime/tca_records.jsonl",
            cast=str,
        )
        or ""
    ).strip()
    trade_history_path = resolve_runtime_artifact_path(
        trade_history_configured or "runtime/tca_records.jsonl",
        default_relative="runtime/tca_records.jsonl",
    )
    gate_summary_configured = str(
        get_env(
            "AI_TRADING_RUNTIME_PERF_GATE_SUMMARY_PATH",
            "runtime/gate_effectiveness_summary.json",
            cast=str,
        )
        or ""
    ).strip()
    gate_summary_path = resolve_runtime_artifact_path(
        gate_summary_configured or "runtime/gate_effectiveness_summary.json",
        default_relative="runtime/gate_effectiveness_summary.json",
    )
    thresholds = {
        "min_closed_trades": int(
            get_env("AI_TRADING_RUNTIME_GONOGO_MIN_CLOSED_TRADES", 20, cast=int)
        ),
        "min_profit_factor": float(
            get_env("AI_TRADING_RUNTIME_GONOGO_MIN_PROFIT_FACTOR", 1.1, cast=float)
        ),
        "min_win_rate": float(
            get_env("AI_TRADING_RUNTIME_GONOGO_MIN_WIN_RATE", 0.5, cast=float)
        ),
        "min_net_pnl": float(
            get_env("AI_TRADING_RUNTIME_GONOGO_MIN_NET_PNL", 0.0, cast=float)
        ),
        "min_acceptance_rate": float(
            get_env("AI_TRADING_RUNTIME_GONOGO_MIN_ACCEPTANCE_RATE", 0.05, cast=float)
        ),
        "min_expected_net_edge_bps": float(
            get_env(
                "AI_TRADING_RUNTIME_GONOGO_MIN_EXPECTED_NET_EDGE_BPS",
                -50.0,
                cast=float,
            )
        ),
        "require_pnl_available": bool(
            get_env("AI_TRADING_RUNTIME_GONOGO_REQUIRE_PNL_AVAILABLE", True, cast=bool)
        ),
        "require_gate_valid": bool(
            get_env("AI_TRADING_RUNTIME_GONOGO_REQUIRE_GATE_VALID", False, cast=bool)
        ),
    }

    try:
        from ai_trading.tools import runtime_performance_report as performance_report

        report = performance_report.build_report(
            trade_history_path=trade_history_path,
            gate_summary_path=gate_summary_path,
        )
        decision = performance_report.evaluate_go_no_go(report, thresholds=thresholds)
    except Exception as exc:
        logger.warning(
            "AFTER_HOURS_RUNTIME_GONOGO_FAILED",
            extra={"cause": exc.__class__.__name__, "detail": str(exc)},
        )
        return {
            "enabled": True,
            "gate_passed": False,
            "reason": "runtime_performance_eval_failed",
            "checks": {},
            "failed_checks": ["runtime_performance_eval_failed"],
            "thresholds": thresholds,
            "observed": {},
            "paths": {
                "trade_history": str(trade_history_path),
                "gate_summary": str(gate_summary_path),
            },
        }

    return {
        "enabled": True,
        "gate_passed": bool(decision.get("gate_passed")),
        "checks": dict(decision.get("checks", {})),
        "failed_checks": list(decision.get("failed_checks", [])),
        "thresholds": dict(decision.get("thresholds", {})),
        "observed": dict(decision.get("observed", {})),
        "paths": {
            "trade_history": str(trade_history_path),
            "gate_summary": str(gate_summary_path),
        },
    }


def _regime_calibration_diagnostics(
    *,
    current: Mapping[str, Any] | None,
    prior: Mapping[str, Any] | None,
) -> dict[str, Any]:
    enabled = bool(
        get_env(
            "AI_TRADING_ROADMAP_PHASE1_REGIME_CALIBRATION_GATE_ENABLED",
            True,
            cast=bool,
        )
    )
    max_brier_delta = _clamp(
        float(get_env("AI_TRADING_ROADMAP_PHASE1_MAX_REGIME_BRIER_DELTA", 0.02, cast=float)),
        low=0.0,
        high=1.0,
    )
    max_ece_delta = _clamp(
        float(get_env("AI_TRADING_ROADMAP_PHASE1_MAX_REGIME_ECE_DELTA", 0.03, cast=float)),
        low=0.0,
        high=1.0,
    )
    max_degraded_regimes = max(
        0, int(get_env("AI_TRADING_ROADMAP_PHASE1_MAX_DEGRADED_REGIMES", 0, cast=int))
    )
    min_support = max(
        1,
        int(get_env("AI_TRADING_ROADMAP_PHASE1_REGIME_CALIBRATION_MIN_SUPPORT", 25, cast=int)),
    )
    summary: dict[str, Any] = {
        "enabled": enabled,
        "gate": True,
        "thresholds": {
            "max_brier_delta": float(max_brier_delta),
            "max_ece_delta": float(max_ece_delta),
            "max_degraded_regimes": int(max_degraded_regimes),
            "min_support": int(min_support),
        },
        "compared_regimes": 0,
        "degraded_regimes": 0,
        "regimes": {},
    }
    if not enabled:
        summary["reason"] = "disabled"
        return summary
    if not isinstance(current, Mapping):
        summary["reason"] = "missing_current_calibration"
        return summary
    if not isinstance(prior, Mapping):
        summary["reason"] = "missing_prior_calibration"
        return summary
    degraded = 0
    compared = 0
    regime_details: dict[str, Any] = {}
    for regime, current_metrics in current.items():
        if not isinstance(current_metrics, Mapping):
            continue
        prior_metrics = prior.get(str(regime))
        if not isinstance(prior_metrics, Mapping):
            continue
        current_support = int(current_metrics.get("support", 0) or 0)
        prior_support = int(prior_metrics.get("support", 0) or 0)
        if current_support < min_support or prior_support < min_support:
            continue
        try:
            current_brier = float(current_metrics.get("brier_score", 0.0) or 0.0)
            prior_brier = float(prior_metrics.get("brier_score", 0.0) or 0.0)
            current_ece = float(current_metrics.get("ece", 0.0) or 0.0)
            prior_ece = float(prior_metrics.get("ece", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        brier_delta = current_brier - prior_brier
        ece_delta = current_ece - prior_ece
        materially_worse = bool(brier_delta > max_brier_delta or ece_delta > max_ece_delta)
        if materially_worse:
            degraded += 1
        compared += 1
        regime_details[str(regime)] = {
            "current_support": int(current_support),
            "prior_support": int(prior_support),
            "current_brier_score": float(current_brier),
            "prior_brier_score": float(prior_brier),
            "brier_delta": float(brier_delta),
            "current_ece": float(current_ece),
            "prior_ece": float(prior_ece),
            "ece_delta": float(ece_delta),
            "materially_worse": materially_worse,
        }
    summary["compared_regimes"] = int(compared)
    summary["degraded_regimes"] = int(degraded)
    summary["regimes"] = regime_details
    summary["gate"] = bool(degraded <= max_degraded_regimes)
    if compared <= 0:
        summary["reason"] = "no_overlap"
    return summary


def _phase1_week1_gate_bundle(
    *,
    best: CandidateMetrics,
    rows: int,
    sensitivity_sweep: Mapping[str, Any],
    prior_metrics: Mapping[str, Any] | None = None,
    regime_calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    enabled = bool(get_env("AI_TRADING_ROADMAP_PHASE1_ENABLED", True, cast=bool))
    require_for_promotion = bool(
        get_env("AI_TRADING_AFTER_HOURS_PROMOTION_REQUIRE_PHASE1_GATE", False, cast=bool)
    )
    require_prior_improvement = bool(
        get_env("AI_TRADING_ROADMAP_PHASE1_REQUIRE_PRIOR_IMPROVEMENT", True, cast=bool)
    )
    thresholds = {
        "min_rows": int(get_env("AI_TRADING_ROADMAP_PHASE1_MIN_ROWS", 1200, cast=int)),
        "min_support": int(
            get_env("AI_TRADING_ROADMAP_PHASE1_MIN_SUPPORT", 120, cast=int)
        ),
        "min_expectancy_bps": float(
            get_env("AI_TRADING_ROADMAP_PHASE1_MIN_EXPECTANCY_BPS", 1.5, cast=float)
        ),
        "max_drawdown_bps": float(
            get_env("AI_TRADING_ROADMAP_PHASE1_MAX_DRAWDOWN_BPS", 1800.0, cast=float)
        ),
        "max_turnover_ratio": float(
            get_env("AI_TRADING_ROADMAP_PHASE1_MAX_TURNOVER_RATIO", 0.35, cast=float)
        ),
        "min_hit_rate_stability": float(
            get_env("AI_TRADING_ROADMAP_PHASE1_MIN_HIT_RATE_STABILITY", 0.60, cast=float)
        ),
        "max_brier_score": float(
            get_env("AI_TRADING_ROADMAP_PHASE1_MAX_BRIER_SCORE", 0.27, cast=float)
        ),
        "min_profitable_fold_ratio": _clamp(
            float(
                get_env(
                    "AI_TRADING_ROADMAP_PHASE1_MIN_PROFITABLE_FOLD_RATIO",
                    0.45,
                    cast=float,
                )
            ),
            low=0.0,
            high=1.0,
        ),
        "min_prior_score_delta": float(
            get_env("AI_TRADING_ROADMAP_PHASE1_MIN_PRIOR_SCORE_DELTA", 0.15, cast=float)
        ),
    }
    metrics = {
        "rows": int(rows),
        "support": int(best.support),
        "mean_expectancy_bps": float(best.mean_expectancy_bps),
        "max_drawdown_bps": float(best.max_drawdown_bps),
        "turnover_ratio": float(best.turnover_ratio),
        "hit_rate_stability": float(best.hit_rate_stability),
        "brier_score": float(best.brier_score),
        "profitable_fold_ratio": float(best.profitable_fold_ratio),
        "sensitivity_gate": bool(sensitivity_sweep.get("gate", True)),
    }
    calibration_diagnostics = _regime_calibration_diagnostics(
        current=regime_calibration,
        prior=(
            prior_metrics.get("regime_calibration")
            if isinstance(prior_metrics, Mapping)
            else None
        ),
    )
    if not enabled:
        return {
            "enabled": False,
            "required_for_promotion": require_for_promotion,
            "gate_passed": True,
            "gates": {},
            "thresholds": thresholds,
            "metrics": metrics,
            "calibration_diagnostics": calibration_diagnostics,
            "prior_model_comparison": {
                "required": require_prior_improvement,
                "available": False,
                "gate": True,
                "reason": "phase1_disabled",
            },
        }
    candidate_score = _promotion_score(
        expectancy_bps=best.mean_expectancy_bps,
        max_drawdown_bps=best.max_drawdown_bps,
    )
    prior_gate = True
    prior_model_comparison: dict[str, Any] = {
        "required": require_prior_improvement,
        "available": False,
        "gate": True,
        "candidate_score": float(candidate_score),
        "margin_required": float(thresholds["min_prior_score_delta"]),
    }
    if require_prior_improvement:
        if prior_metrics is None:
            prior_gate = False
            prior_model_comparison["reason"] = "no_prior_metrics"
        else:
            try:
                prior_expectancy = float(prior_metrics.get("mean_expectancy_bps"))
                prior_drawdown = float(prior_metrics.get("max_drawdown_bps"))
            except (TypeError, ValueError):
                prior_gate = False
                prior_model_comparison["reason"] = "invalid_prior_metrics"
            else:
                prior_score = _promotion_score(
                    expectancy_bps=prior_expectancy,
                    max_drawdown_bps=prior_drawdown,
                )
                score_delta = float(candidate_score - prior_score)
                prior_gate = score_delta >= float(thresholds["min_prior_score_delta"])
                prior_model_comparison.update(
                    {
                        "available": True,
                        "prior_score": float(prior_score),
                        "score_delta": float(score_delta),
                        "prior_report_path": prior_metrics.get("report_path"),
                        "prior_model_id": prior_metrics.get("model_id"),
                        "prior_governance_status": prior_metrics.get("governance_status"),
                    }
                )
    prior_model_comparison["gate"] = bool(prior_gate)
    gates = {
        "rows": int(rows) >= int(thresholds["min_rows"]),
        "support": int(best.support) >= int(thresholds["min_support"]),
        "expectancy": float(best.mean_expectancy_bps) >= float(thresholds["min_expectancy_bps"]),
        "drawdown": float(best.max_drawdown_bps) <= float(thresholds["max_drawdown_bps"]),
        "turnover": float(best.turnover_ratio) <= float(thresholds["max_turnover_ratio"]),
        "stability": float(best.hit_rate_stability)
        >= float(thresholds["min_hit_rate_stability"]),
        "brier": float(best.brier_score) <= float(thresholds["max_brier_score"]),
        "profitable_fold_ratio": float(best.profitable_fold_ratio)
        >= float(thresholds["min_profitable_fold_ratio"]),
        "sensitivity": bool(sensitivity_sweep.get("gate", True)),
        "regime_calibration": bool(calibration_diagnostics.get("gate", True)),
        "prior_model_improvement": bool(prior_gate),
    }
    gate_passed = all(bool(value) for value in gates.values())
    return {
        "enabled": True,
        "required_for_promotion": require_for_promotion,
        "gate_passed": bool(gate_passed),
        "gates": gates,
        "thresholds": thresholds,
        "metrics": metrics,
        "calibration_diagnostics": calibration_diagnostics,
        "prior_model_comparison": prior_model_comparison,
    }


def _candidate_names() -> list[str]:
    names: list[str] = ["logreg"]
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: F401
    except Exception:
        pass
    else:
        names.append("histgb")
    try:
        import lightgbm  # noqa: F401
    except Exception:
        pass
    else:
        names.append("lightgbm")
    try:
        import xgboost  # noqa: F401
    except Exception:
        pass
    else:
        names.append("xgboost")
    return names


def _serialize_candidate_metrics(
    candidates: list[CandidateMetrics],
    *,
    best_name: str,
    selection_weights: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    ranked = sorted(
        candidates,
        key=lambda item: (
            _candidate_selection_score(item, weights=selection_weights),
            item.mean_expectancy_bps,
            -item.max_drawdown_bps,
            item.support,
        ),
        reverse=True,
    )
    for rank, item in enumerate(ranked, start=1):
        payload.append(
            {
                "rank": rank,
                "name": item.name,
                "selected": item.name == best_name,
                "fold_count": item.fold_count,
                "profitable_fold_count": item.profitable_fold_count,
                "profitable_fold_ratio": item.profitable_fold_ratio,
                "support": item.support,
                "mean_expectancy_bps": item.mean_expectancy_bps,
                "max_drawdown_bps": item.max_drawdown_bps,
                "turnover_ratio": item.turnover_ratio,
                "mean_hit_rate": item.mean_hit_rate,
                "hit_rate_stability": item.hit_rate_stability,
                "brier_score": item.brier_score,
                "selection_score": _candidate_selection_score(item, weights=selection_weights),
                "regime_metrics": item.regime_metrics,
                "regime_calibration": item.regime_calibration,
            }
        )
    return payload


def _training_feature_stats(dataset: Any) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for feature in FEATURE_COLUMNS:
        if feature not in dataset.columns:
            continue
        values = np.asarray(dataset[feature], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size <= 0:
            continue
        std = float(np.std(finite))
        stats[str(feature)] = {
            "support": float(finite.size),
            "mean": float(np.mean(finite)),
            "std": std if std > 0.0 else 0.0,
            "p05": float(np.quantile(finite, 0.05)),
            "p95": float(np.quantile(finite, 0.95)),
        }
    return stats


def _fit_final_model(name: str, dataset: Any, *, seed: int):
    model = _fit_candidate_model(name, seed=seed)
    X = dataset.loc[:, FEATURE_COLUMNS]
    y = dataset["label"].astype(int)
    model.fit(X, y)
    setattr(model, "training_feature_stats_", _training_feature_stats(dataset))
    setattr(model, "training_feature_stats_version_", "1")
    return model


def _dataset_fingerprint(dataset: Any, *, symbols: list[str], cost_floor_bps: float) -> str:
    start_ts = str(dataset["timestamp"].iloc[0])
    end_ts = str(dataset["timestamp"].iloc[-1])
    payload = {
        "symbols": symbols,
        "rows": int(len(dataset)),
        "start": start_ts,
        "end": end_ts,
        "cost_floor_bps": float(cost_floor_bps),
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _build_manifest_metadata(
    *,
    symbols: list[str],
    rows: int,
    lookback_days: int,
    default_threshold: float,
    thresholds_by_regime: Mapping[str, float],
    cost_floor_bps: float,
    dataset_fingerprint: str,
    sensitivity_sweep: Mapping[str, Any],
) -> dict[str, Any]:
    feature_hash = hashlib.sha256(
        json.dumps(list(FEATURE_COLUMNS), sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest_metadata = {
        "strategy": "after_hours_ml_edge",
        "symbols": list(symbols),
        "rows": int(rows),
        "lookback_days": int(lookback_days),
        "horizon_days": int(get_env("AI_TRADING_AFTER_HOURS_HORIZON_DAYS", 1, cast=int)),
        "embargo_days": int(get_env("AI_TRADING_AFTER_HOURS_EMBARGO_DAYS", 1, cast=int)),
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_hash": feature_hash,
        "default_threshold": float(default_threshold),
        "thresholds_by_regime": dict(thresholds_by_regime),
        "cost_floor_bps": float(cost_floor_bps),
        "cost_model_version": str(
            get_env("AI_TRADING_AFTER_HOURS_COST_MODEL_VERSION", "tca_floor_v1") or "tca_floor_v1"
        ),
        "data_sources": {
            "daily_source": str(get_env("DAILY_SOURCE", "unknown") or "unknown"),
            "minute_source": str(get_env("MINUTE_SOURCE", "unknown") or "unknown"),
            "data_provenance": str(get_env("DATA_PROVENANCE", "iex") or "iex"),
            "alpaca_data_feed": str(get_env("ALPACA_DATA_FEED", "unknown") or "unknown"),
        },
        "dataset_fingerprint": dataset_fingerprint,
        "sensitivity_sweep": {
            "enabled": bool(sensitivity_sweep.get("enabled", False)),
            "gate": bool(sensitivity_sweep.get("gate", True)),
            "summary": sensitivity_sweep.get("summary", {}),
        },
    }
    return validate_manifest_metadata(manifest_metadata)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, default=str, indent=2), encoding="utf-8")
    return path


def _write_after_hours_reports(
    *,
    report_dir: Path,
    now_utc: datetime,
    payload: Mapping[str, Any],
) -> tuple[Path, Path]:
    report_obj = dict(payload)
    timestamped_path = report_dir / f"after_hours_training_{now_utc.strftime('%Y%m%d_%H%M%S')}.json"
    daily_alias_path = report_dir / f"after_hours_training_{now_utc.strftime('%Y%m%d')}.json"
    _write_json(timestamped_path, report_obj)
    if bool(get_env("AI_TRADING_AFTER_HOURS_WRITE_DAILY_REPORT_ALIAS", True, cast=bool)):
        _write_json(daily_alias_path, report_obj)
    return timestamped_path, daily_alias_path


def _is_pytest_temp_path(path_value: Any) -> bool:
    normalized = str(path_value or "").strip().replace("\\", "/").lower()
    return "/pytest-of-" in normalized or "/tmp/pytest-" in normalized


def _sanitize_after_hours_training_state_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = dict(payload)
    removed_keys: list[str] = []
    for key in ("report_path", "daily_report_path"):
        value = sanitized.get(key)
        if value is None:
            continue
        if _is_pytest_temp_path(value):
            sanitized.pop(key, None)
            removed_keys.append(key)
    if removed_keys:
        logger.warning(
            "AFTER_HOURS_TRAINING_STATE_SANITIZED",
            extra={"removed_keys": removed_keys},
        )
    return sanitized


def _after_hours_training_state_path() -> Path:
    raw = str(
        get_env(
            "AI_TRADING_AFTER_HOURS_TRAINING_STATE_PATH",
            "runtime/after_hours_training_state.json",
            cast=str,
        )
        or ""
    ).strip()
    if not raw:
        raw = "runtime/after_hours_training_state.json"
    return _resolve_after_hours_output_path(
        raw,
        default_relative="runtime/after_hours_training_state.json",
    )


def _load_after_hours_training_state() -> Mapping[str, Any] | None:
    path = _after_hours_training_state_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        logger.warning(
            "AFTER_HOURS_TRAINING_STATE_READ_FAILED",
            extra={"path": str(path)},
        )
        return None
    if isinstance(payload, Mapping):
        return _sanitize_after_hours_training_state_payload(payload)
    return None


def _write_after_hours_training_state(payload: Mapping[str, Any]) -> Path | None:
    path = _after_hours_training_state_path()
    fallback = (paths.DATA_DIR / "runtime/after_hours_training_state.json").resolve()
    try:
        writable_dir = _resolve_writable_output_dir(
            requested=path.parent,
            fallback=fallback.parent,
            event_name="AFTER_HOURS_TRAINING_STATE_PATH_FALLBACK",
        )
    except RuntimeError:
        logger.warning(
            "AFTER_HOURS_TRAINING_STATE_WRITE_FAILED",
            extra={"path": str(path), "reason": "no_writable_directory"},
        )
        return None
    target = path if writable_dir == path.parent else (fallback if writable_dir == fallback.parent else writable_dir / path.name)
    sanitized_payload = _sanitize_after_hours_training_state_payload(payload)
    try:
        return _write_json(target, sanitized_payload)
    except OSError as exc:
        logger.warning(
            "AFTER_HOURS_TRAINING_STATE_WRITE_FAILED",
            extra={"path": str(target), "error": str(exc)},
        )
        return None


def _new_rows_since_training_state(
    dataset: Any,
    *,
    previous_state: Mapping[str, Any] | None,
) -> int:
    if dataset is None or getattr(dataset, "empty", True):
        return 0
    if not isinstance(previous_state, Mapping):
        return int(len(dataset))
    prev_raw = previous_state.get("max_label_ts")
    prev_label_ts = _parse_ts(prev_raw)
    if prev_label_ts is None:
        return int(len(dataset))
    import pandas as pd
    try:
        label_ts = pd.to_datetime(dataset["label_ts"], utc=True, errors="coerce")
    except Exception:
        return int(len(dataset))
    if label_ts is None:
        return int(len(dataset))
    valid = label_ts.dropna()
    if valid.empty:
        return 0
    return int((valid > prev_label_ts).sum())


def _maybe_promote_to_runtime_model_path(
    model_path: Path,
    *,
    manifest_metadata: Mapping[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    promote_enabled = bool(
        get_env("AI_TRADING_AFTER_HOURS_PROMOTE_MODEL_PATH", True, cast=bool)
    )
    if not promote_enabled:
        return None, None
    runtime_model_path_raw = str(get_env("AI_TRADING_MODEL_PATH", "") or "").strip()
    if not runtime_model_path_raw:
        runtime_model_path_raw = str(
            get_env(
                "AI_TRADING_AFTER_HOURS_RUNTIME_MODEL_PATH",
                "models/runtime/ml_latest.joblib",
                cast=str,
            )
            or ""
        ).strip()
    if not runtime_model_path_raw:
        return None, None
    runtime_model_path = _resolve_after_hours_output_path(
        runtime_model_path_raw,
        default_relative="models/runtime/ml_latest.joblib",
    )
    try:
        runtime_model_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_model_path.write_bytes(model_path.read_bytes())
        manifest_path = write_artifact_manifest(
            model_path=str(runtime_model_path),
            model_version=f"edge_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
            training_data_range=None,
            metadata=manifest_metadata,
        )
    except OSError as exc:
        logger.error(
            "AFTER_HOURS_RUNTIME_PROMOTION_FAILED",
            extra={
                "source_path": str(model_path),
                "target_path": str(runtime_model_path),
                "exc_type": exc.__class__.__name__,
                "error": str(exc),
            },
        )
        return None, None
    return str(runtime_model_path), str(manifest_path)


def _maybe_promote_rl_overlay_to_runtime_path(
    trained_model_path: Path,
) -> str | None:
    promote_enabled = bool(
        get_env("AI_TRADING_AFTER_HOURS_PROMOTE_RL_PATH", True, cast=bool)
    )
    if not promote_enabled:
        return None
    if not trained_model_path.is_file():
        return None
    runtime_model_path_raw = str(get_env("AI_TRADING_RL_MODEL_PATH", "") or "").strip()
    if not runtime_model_path_raw:
        runtime_model_path_raw = str(
            get_env(
                "AI_TRADING_AFTER_HOURS_RUNTIME_RL_MODEL_PATH",
                "models/runtime/rl_agent.zip",
                cast=str,
            )
            or ""
        ).strip()
    if not runtime_model_path_raw:
        return None
    runtime_model_path = _resolve_after_hours_output_path(
        runtime_model_path_raw,
        default_relative="models/runtime/rl_agent.zip",
    )
    try:
        runtime_model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trained_model_path, runtime_model_path)
    except OSError as exc:
        logger.error(
            "AFTER_HOURS_RL_PROMOTION_FAILED",
            extra={
                "source_path": str(trained_model_path),
                "target_path": str(runtime_model_path),
                "exc_type": exc.__class__.__name__,
                "error": str(exc),
            },
        )
        return None
    return str(runtime_model_path)


def _maybe_train_rl_overlay(
    dataset: Any,
    *,
    now_utc: datetime,
    baseline_expectancy_bps: float,
    dataset_fingerprint: str | None = None,
    feature_hash: str | None = None,
    governance_status: str = "shadow",
) -> dict[str, Any]:
    enabled = bool(get_env("AI_TRADING_AFTER_HOURS_RL_OVERLAY_ENABLED", False, cast=bool))
    if not enabled:
        return {"enabled": False, "trained": False, "recommend_use_rl_agent": False}
    try:
        from ai_trading.rl_trading.train import RLTrainer, train_multi_seed
    except Exception as exc:
        return {
            "enabled": True,
            "trained": False,
            "recommend_use_rl_agent": False,
            "error": str(exc),
        }
    rl_dataset = dataset
    if any(column not in rl_dataset.columns for column in FEATURE_COLUMNS):
        rl_dataset = _augment_training_features(rl_dataset.copy())
    data = rl_dataset.reindex(columns=list(FEATURE_COLUMNS)).astype(float).to_numpy()
    close_series = dataset.get("close")
    if close_series is not None:
        close_prices = close_series.astype(float).to_numpy()
    else:
        close_prices = data[:, 0]
    if data.shape[0] < 120:
        return {
            "enabled": True,
            "trained": False,
            "recommend_use_rl_agent": False,
            "reason": "insufficient_rows",
        }
    timesteps = int(get_env("AI_TRADING_AFTER_HOURS_RL_TIMESTEPS", 15000, cast=int))
    algorithm = str(get_env("AI_TRADING_AFTER_HOURS_RL_ALGO", "PPO") or "PPO").upper()
    requested_out_dir = _resolve_after_hours_output_path(
        str(get_env("AI_TRADING_AFTER_HOURS_RL_DIR", "models/after_hours_rl", cast=str) or ""),
        default_relative="models/after_hours_rl",
    )
    try:
        out_dir = _resolve_writable_output_dir(
            requested=requested_out_dir,
            fallback=(paths.MODELS_DIR / "after_hours_rl").resolve(),
            event_name="AFTER_HOURS_RL_DIR_FALLBACK",
        )
    except RuntimeError as exc:
        logger.error(
            "AFTER_HOURS_RL_DIR_UNWRITABLE",
            extra={"requested": str(requested_out_dir), "error": str(exc)},
        )
        return {
            "enabled": True,
            "trained": False,
            "recommend_use_rl_agent": False,
            "reason": "rl_dir_unwritable",
            "error": str(exc),
        }
    run_dir = out_dir / now_utc.strftime("%Y%m%d_%H%M%S")
    tags = ["after_hours", "rl_overlay", "cost_aware", "walk_forward"]
    multi_seed_summary: dict[str, Any] | None = None
    trainer = RLTrainer(
        algorithm=algorithm,
        total_timesteps=max(2000, timesteps),
        eval_freq=max(1000, timesteps // 5),
        early_stopping_patience=5,
        seed=int(get_env("AI_TRADING_SEED", 42, cast=int)),
    )
    env_params: dict[str, Any] = {
        "transaction_cost": float(get_env("AI_TRADING_RL_TRANSACTION_COST", 0.001, cast=float)),
        "slippage": float(get_env("AI_TRADING_RL_SLIPPAGE", 0.0005, cast=float)),
        "half_spread": float(get_env("AI_TRADING_RL_HALF_SPREAD", 0.0002, cast=float)),
        "price_series": close_prices,
        "register_model": True,
        "registry_strategy": "rl_overlay",
        "registry_model_type": algorithm.lower(),
        "registry_tags": tags,
        "registry_requested_status": governance_status,
    }
    if dataset_fingerprint:
        env_params["dataset_fingerprint"] = str(dataset_fingerprint)
    if feature_hash:
        env_params["feature_spec_hash"] = str(feature_hash)
    results = trainer.train(
        data=data,
        env_params=env_params,
        save_path=str(run_dir),
    )
    multi_seed_enabled = bool(
        get_env("AI_TRADING_AFTER_HOURS_RL_MULTI_SEED_ENABLED", False, cast=bool)
    )
    if multi_seed_enabled:
        seeds_raw = str(
            get_env("AI_TRADING_AFTER_HOURS_RL_MULTI_SEEDS", "11,23,37", cast=str) or ""
        )
        seeds: list[int] = []
        for token in seeds_raw.split(","):
            token_clean = token.strip()
            if not token_clean:
                continue
            try:
                seed_value = int(token_clean)
            except ValueError:
                continue
            if seed_value not in seeds:
                seeds.append(seed_value)
        if seeds:
            seed_run_root = run_dir / "seed_matrix"
            multi_seed_summary = train_multi_seed(
                data=data,
                seeds=seeds,
                algorithm=algorithm,
                total_timesteps=max(2000, timesteps),
                eval_freq=max(1000, timesteps // 5),
                early_stopping_patience=5,
                env_params=env_params,
                model_params=None,
                save_root=str(seed_run_root),
            )
    final_eval = results.get("final_evaluation", {}) if isinstance(results, dict) else {}
    mean_reward = float(final_eval.get("mean_reward", 0.0) or 0.0)
    rl_reward_target = float(get_env("AI_TRADING_RL_MIN_MEAN_REWARD", 0.0, cast=float))
    baseline_target = float(
        get_env("AI_TRADING_RL_REQUIRE_BASELINE_EXPECTANCY_BPS", 1.0, cast=float)
    )
    recommend = mean_reward >= rl_reward_target and baseline_expectancy_bps >= baseline_target
    trained_model_path = run_dir / f"model_{algorithm.lower()}.zip"
    trained_model_path_str = str(trained_model_path) if trained_model_path.is_file() else None
    promoted_model_path = (
        _maybe_promote_rl_overlay_to_runtime_path(trained_model_path)
        if trained_model_path_str
        else None
    )
    return {
        "enabled": True,
        "trained": True,
        "algorithm": algorithm,
        "timesteps": timesteps,
        "output_dir": str(run_dir),
        "trained_model_path": trained_model_path_str,
        "promoted_model_path": promoted_model_path,
        "mean_reward": mean_reward,
        "rl_reward_target": rl_reward_target,
        "baseline_expectancy_bps": baseline_expectancy_bps,
        "recommend_use_rl_agent": bool(recommend),
        "model_id": results.get("model_id") if isinstance(results, dict) else None,
        "governance_status": results.get("governance_status") if isinstance(results, dict) else None,
        "multi_seed_summary": multi_seed_summary,
    }


def run_after_hours_training(*, now: datetime | None = None) -> dict[str, Any]:
    """Run after-hours ML training and optional RL overlay training."""
    import pandas as pd

    now_utc = (now or datetime.now(UTC)).astimezone(UTC)
    now_ny = now_utc.astimezone(ZoneInfo("America/New_York"))
    now_minutes = (int(now_ny.hour) * 60) + int(now_ny.minute)
    close_minutes = 16 * 60
    catchup_end_minutes = (9 * 60) + 30
    catchup_enabled = bool(
        get_env("AI_TRADING_AFTER_HOURS_TRAINING_CATCHUP_ENABLED", True, cast=bool)
    )
    in_after_hours_window = now_minutes >= close_minutes
    in_catchup_window = catchup_enabled and now_minutes < catchup_end_minutes
    if not (in_after_hours_window or in_catchup_window):
        return {
            "status": "skipped",
            "reason": "before_market_close",
            "timestamp": now_utc.isoformat(),
        }
    symbols = _load_symbols()
    if not symbols:
        return {
            "status": "skipped",
            "reason": "no_symbols",
            "timestamp": now_utc.isoformat(),
        }
    tca_path = str(get_env("AI_TRADING_TCA_PATH", "runtime/tca_records.jsonl"))
    tca_records = _read_jsonl_records(
        tca_path,
        max_records=int(get_env("AI_TRADING_AFTER_HOURS_TCA_WINDOW", 20000, cast=int)),
    )
    fill_quality = _build_fill_quality_metrics(tca_records)
    cost_floor_bps = _estimate_cost_floor_bps(tca_records)
    lookback_days = int(get_env("AI_TRADING_AFTER_HOURS_LOOKBACK_DAYS", 540, cast=int))
    dataset = _build_training_dataset(
        symbols,
        lookback_days=lookback_days,
        cost_floor_bps=cost_floor_bps,
        now_utc=now_utc,
    )
    min_rows = int(get_env("AI_TRADING_AFTER_HOURS_MIN_ROWS", 250, cast=int))
    if dataset.empty or len(dataset) < min_rows:
        return {
            "status": "skipped",
            "reason": "insufficient_dataset",
            "rows": int(len(dataset)),
            "required_rows": min_rows,
            "timestamp": now_utc.isoformat(),
        }
    dataset_fp = _dataset_fingerprint(dataset, symbols=symbols, cost_floor_bps=cost_floor_bps)
    training_state = _load_after_hours_training_state()
    min_new_rows = max(
        1,
        int(get_env("AI_TRADING_AFTER_HOURS_MIN_NEW_ROWS_FOR_RETRAIN", 25, cast=int)),
    )
    new_rows = _new_rows_since_training_state(
        dataset,
        previous_state=training_state,
    )
    previous_fp = (
        str(training_state.get("dataset_fingerprint", "") or "").strip()
        if isinstance(training_state, Mapping)
        else ""
    )
    unchanged_dataset_fingerprint = bool(previous_fp) and previous_fp == dataset_fp
    force_retrain = bool(get_env("AI_TRADING_AFTER_HOURS_FORCE_RETRAIN", False, cast=bool))
    skip_no_new_signal_data = bool(
        get_env("AI_TRADING_AFTER_HOURS_SKIP_IF_NO_NEW_SIGNAL_DATA", True, cast=bool)
    )
    if (
        skip_no_new_signal_data
        and not force_retrain
        and isinstance(training_state, Mapping)
        and (unchanged_dataset_fingerprint or int(new_rows) < min_new_rows)
    ):
        return {
            "status": "skipped",
            "reason": "no_new_signal_data",
            "rows": int(len(dataset)),
            "new_rows": int(new_rows),
            "min_new_rows": int(min_new_rows),
            "unchanged_dataset_fingerprint": bool(unchanged_dataset_fingerprint),
            "dataset_fingerprint": str(dataset_fp),
            "timestamp": now_utc.isoformat(),
        }
    split_idx = max(20, int(len(dataset) * 0.7))
    horizon_days = int(get_env("AI_TRADING_AFTER_HOURS_HORIZON_DAYS", 1, cast=int))
    embargo_days = int(get_env("AI_TRADING_AFTER_HOURS_EMBARGO_DAYS", 1, cast=int))
    label_ts_series = pd.to_datetime(dataset["label_ts"], utc=True)
    timestamp_series = pd.to_datetime(dataset["timestamp"], utc=True)
    if split_idx < len(dataset):
        test_label_times = label_ts_series.iloc[split_idx:]
        if not test_label_times.empty:
            global_gap_days = max(1, horizon_days + embargo_days)
            cutoff = test_label_times.min() - pd.Timedelta(days=global_gap_days)
            train_label_times = label_ts_series[label_ts_series <= cutoff]
            if train_label_times.empty:
                logger.info(
                    "AFTER_HOURS_GLOBAL_LEAKAGE_GUARD_SKIPPED",
                    extra={
                        "reason": "insufficient_train_after_gap",
                        "global_gap_days": global_gap_days,
                        "rows": int(len(dataset)),
                    },
                )
            else:
                try:
                    run_leakage_guards(
                        feature_timestamps=timestamp_series,
                        label_timestamps=label_ts_series,
                        train_label_times=train_label_times,
                        test_label_times=test_label_times,
                        horizon_days=horizon_days,
                        embargo_days=embargo_days,
                    )
                except AssertionError as exc:
                    logger.info(
                        "AFTER_HOURS_GLOBAL_LEAKAGE_GUARD_SOFTFAIL",
                        extra={"error": str(exc)},
                    )
    seed = int(get_env("AI_TRADING_SEED", 42, cast=int))
    default_threshold = float(
        get_env("AI_TRADING_AFTER_HOURS_DEFAULT_THRESHOLD", 0.5, cast=float)
    )
    candidate_results: list[CandidateMetrics] = []
    for name in _candidate_names():
        metrics = _evaluate_candidate(
            name,
            dataset,
            seed=seed,
            threshold=default_threshold,
        )
        if metrics is not None:
            candidate_results.append(metrics)
    if not candidate_results:
        return {
            "status": "skipped",
            "reason": "no_candidate_models",
            "timestamp": now_utc.isoformat(),
        }
    selection_weights = _resolved_model_selection_weights()
    best = max(
        candidate_results,
        key=lambda item: (
            _candidate_selection_score(item, weights=selection_weights),
            item.mean_expectancy_bps,
            -item.max_drawdown_bps,
            item.support,
        ),
    )
    candidate_metrics_payload = _serialize_candidate_metrics(
        candidate_results,
        best_name=best.name,
        selection_weights=selection_weights,
    )
    requested_report_dir = _resolve_after_hours_output_path(
        str(get_env("AI_TRADING_AFTER_HOURS_REPORT_DIR", "runtime/research_reports", cast=str) or ""),
        default_relative="runtime/research_reports",
    )
    report_dir = _resolve_writable_output_dir(
        requested=requested_report_dir,
        fallback=(paths.DATA_DIR / "runtime/research_reports").resolve(),
        event_name="AFTER_HOURS_REPORT_DIR_FALLBACK",
    )
    prior_model_metrics = _load_prior_model_metrics(report_dir=report_dir)
    raw_thresholds_by_regime = _threshold_by_regime(
        dataset,
        best.oof_probabilities,
        default_threshold=default_threshold,
    )
    thresholds_by_regime, regime_threshold_adjustments = _apply_regime_calibration_threshold_adjustment(
        thresholds_by_regime=raw_thresholds_by_regime,
        regime_calibration=best.regime_calibration,
    )
    targets = _edge_targets()
    sensitivity_sweep = _run_sensitivity_sweep(
        dataset,
        best.oof_probabilities,
        default_threshold=default_threshold,
        targets=targets,
    )
    gate_map = _meets_edge_targets(best, targets)
    gate_map["sensitivity"] = bool(sensitivity_sweep.get("gate", True))
    phase1_week1 = _phase1_week1_gate_bundle(
        best=best,
        rows=int(len(dataset)),
        sensitivity_sweep=sensitivity_sweep,
        prior_metrics=prior_model_metrics,
        regime_calibration=best.regime_calibration,
    )
    runtime_performance_gate = _runtime_performance_go_no_go_gate()
    roadmap_additional_gates: dict[str, bool] = {}
    if bool(phase1_week1.get("required_for_promotion", False)):
        roadmap_additional_gates["phase1_week1"] = bool(phase1_week1.get("gate_passed", False))
    if bool(runtime_performance_gate.get("enabled", False)):
        roadmap_additional_gates["runtime_performance"] = bool(
            runtime_performance_gate.get("gate_passed", False)
        )
    promotion = _promotion_gate_bundle(
        best=best,
        rows=int(len(dataset)),
        edge_gates=gate_map,
        prior_metrics=prior_model_metrics,
        additional_gates=roadmap_additional_gates,
    )
    status = str(promotion["status"])
    logger.info(
        "AFTER_HOURS_PHASE1_GATE_EVAL",
        extra={
            "enabled": bool(phase1_week1.get("enabled", False)),
            "required_for_promotion": bool(phase1_week1.get("required_for_promotion", False)),
            "gate_passed": bool(phase1_week1.get("gate_passed", False)),
            "gates": dict(phase1_week1.get("gates", {})),
        },
    )
    logger.info(
        "AFTER_HOURS_PROMOTION_EVAL",
        extra={
            "policy": promotion["policy"],
            "status": status,
            "auto_promote": bool(promotion["auto_promote"]),
            "gate_passed": bool(promotion["gate_passed"]),
            "combined_gates": dict(promotion["combined_gates"]),
            "prior_model_comparison": dict(promotion.get("prior_model_comparison", {})),
            "runtime_performance_gate": runtime_performance_gate,
        },
    )

    manifest_metadata = _build_manifest_metadata(
        symbols=symbols,
        rows=int(len(dataset)),
        lookback_days=lookback_days,
        default_threshold=default_threshold,
        thresholds_by_regime=thresholds_by_regime,
        cost_floor_bps=cost_floor_bps,
        dataset_fingerprint=dataset_fp,
        sensitivity_sweep=sensitivity_sweep,
    )

    final_model = _fit_final_model(best.name, dataset, seed=seed)
    setattr(final_model, "edge_thresholds_by_regime_", thresholds_by_regime)
    setattr(final_model, "edge_global_threshold_", float(default_threshold))
    setattr(final_model, "regime_calibration_", best.regime_calibration)
    requested_model_dir = _resolve_after_hours_output_path(
        str(get_env("AI_TRADING_AFTER_HOURS_MODEL_DIR", "models/after_hours", cast=str) or ""),
        default_relative="models/after_hours",
    )
    fallback_model_dir = (paths.MODELS_DIR / "after_hours").resolve()
    model_dir = _resolve_writable_output_dir(
        requested=requested_model_dir,
        fallback=fallback_model_dir,
        event_name="AFTER_HOURS_MODEL_DIR_FALLBACK",
    )
    model_version = f"{best.name}_{now_utc.strftime('%Y%m%d_%H%M%S')}"
    model_path = model_dir / f"ml_edge_{model_version}.joblib"
    model_path = _dump_model_with_fallback(
        final_model,
        model_path,
        fallback_dir=(fallback_model_dir if fallback_model_dir != model_dir else None),
    )
    manifest_path = write_artifact_manifest(
        model_path=str(model_path),
        model_version=model_version,
        training_data_range={
            "start": str(dataset["timestamp"].iloc[0]),
            "end": str(dataset["timestamp"].iloc[-1]),
        },
        metadata=manifest_metadata,
    )
    registry = ModelRegistry()
    model_id = registry.register_model(
        final_model,
        strategy="ml_edge",
        model_type=best.name,
        metadata={
            "feature_columns": list(FEATURE_COLUMNS),
            "thresholds_by_regime": thresholds_by_regime,
            "default_threshold": default_threshold,
            "rows": int(len(dataset)),
            "symbols": symbols,
            "manifest_path": str(manifest_path),
            "model_path": str(model_path),
            "fill_quality": fill_quality,
            "sensitivity_sweep": sensitivity_sweep,
            "regime_calibration": best.regime_calibration,
            "regime_threshold_adjustments": regime_threshold_adjustments,
            "selection_score": _candidate_selection_score(best, weights=selection_weights),
            "manifest_metadata": manifest_metadata,
        },
        dataset_fingerprint=dataset_fp,
        tags=["after_hours", "cost_aware", "purged_walk_forward"],
        activate=True,
    )
    registry.update_governance_status(
        model_id,
        status,
        extra={
            "edge_gates": gate_map,
            "promotion": promotion,
            "edge_targets": {
                "min_expectancy_bps": targets.min_expectancy_bps,
                "max_drawdown_bps": targets.max_drawdown_bps,
                "max_turnover_ratio": targets.max_turnover_ratio,
                "min_hit_rate_stability": targets.min_hit_rate_stability,
                "sensitivity_gate": bool(sensitivity_sweep.get("gate", True)),
            },
            "metrics": {
                "mean_expectancy_bps": best.mean_expectancy_bps,
                "max_drawdown_bps": best.max_drawdown_bps,
                "turnover_ratio": best.turnover_ratio,
                "mean_hit_rate": best.mean_hit_rate,
                "hit_rate_stability": best.hit_rate_stability,
                "brier_score": best.brier_score,
                "selection_score": _candidate_selection_score(best, weights=selection_weights),
                "profitable_fold_count": best.profitable_fold_count,
                "profitable_fold_ratio": best.profitable_fold_ratio,
            },
            "prior_model_metrics": prior_model_metrics,
            "sensitivity_sweep": sensitivity_sweep,
            "roadmap": {"phase_1_week_1": phase1_week1},
            "runtime_performance_gate": runtime_performance_gate,
        },
    )
    promoted_model_path, promoted_manifest_path = _maybe_promote_to_runtime_model_path(
        model_path,
        manifest_metadata=manifest_metadata,
    )
    rl_overlay = _maybe_train_rl_overlay(
        dataset,
        now_utc=now_utc,
        baseline_expectancy_bps=best.mean_expectancy_bps,
        dataset_fingerprint=dataset_fp,
        feature_hash=str(manifest_metadata.get("feature_hash", "")),
        governance_status=status,
    )
    training_data_delta = {
        "new_rows": int(new_rows),
        "min_new_rows": int(min_new_rows),
        "unchanged_dataset_fingerprint": bool(unchanged_dataset_fingerprint),
    }
    report = {
        "ts": now_utc.isoformat(),
        "symbols": symbols,
        "rows": int(len(dataset)),
        "cost_floor_bps": float(cost_floor_bps),
        "feature_columns": list(FEATURE_COLUMNS),
        "model": {
            "model_id": model_id,
            "name": best.name,
            "model_path": str(model_path),
            "manifest_path": str(manifest_path),
            "manifest_metadata": manifest_metadata,
            "governance_status": status,
        },
        "metrics": {
            "mean_expectancy_bps": best.mean_expectancy_bps,
            "max_drawdown_bps": best.max_drawdown_bps,
            "turnover_ratio": best.turnover_ratio,
            "mean_hit_rate": best.mean_hit_rate,
            "hit_rate_stability": best.hit_rate_stability,
            "brier_score": best.brier_score,
            "selection_score": _candidate_selection_score(best, weights=selection_weights),
            "fold_count": best.fold_count,
            "profitable_fold_count": best.profitable_fold_count,
            "profitable_fold_ratio": best.profitable_fold_ratio,
            "support": best.support,
        },
        "fill_quality": fill_quality,
        "candidate_metrics": candidate_metrics_payload,
        "regime_metrics": best.regime_metrics,
        "regime_calibration": best.regime_calibration,
        "thresholds_by_regime": thresholds_by_regime,
        "thresholds_by_regime_raw": raw_thresholds_by_regime,
        "regime_threshold_adjustments": regime_threshold_adjustments,
        "sensitivity_sweep": sensitivity_sweep,
        "edge_gates": gate_map,
        "promotion": promotion,
        "roadmap": {"phase_1_week_1": phase1_week1},
        "runtime_performance_gate": runtime_performance_gate,
        "prior_model_metrics": prior_model_metrics,
        "training_data_delta": training_data_delta,
        "rl_overlay": rl_overlay,
        "selection_weights": dict(selection_weights),
        "runtime_promotion": {
            "model_path": promoted_model_path,
            "manifest_path": promoted_manifest_path,
        },
    }
    model_selection_retune = _maybe_retune_model_selection_weights(
        now_utc=now_utc,
        report_dir=report_dir,
        pending_report=report,
        current_weights=selection_weights,
    )
    report["model_selection_retune"] = model_selection_retune
    if bool(model_selection_retune.get("retuned", False)):
        logger.info(
            "AFTER_HOURS_SELECTION_RETUNED",
            extra={
                "override_path": str(model_selection_retune.get("override_path", "")),
                "report_count": int(model_selection_retune.get("report_count", 0) or 0),
            },
        )
    report_path, daily_report_path = _write_after_hours_reports(
        report_dir=report_dir,
        now_utc=now_utc,
        payload=report,
    )
    max_label_ts = _parse_ts(str(dataset["label_ts"].iloc[-1]))
    _write_after_hours_training_state(
        {
            "updated_at": now_utc.isoformat(),
            "rows": int(len(dataset)),
            "dataset_fingerprint": str(dataset_fp),
            "max_label_ts": (
                max_label_ts.isoformat()
                if isinstance(max_label_ts, datetime)
                else str(dataset["label_ts"].iloc[-1])
            ),
            "report_path": str(report_path),
            "daily_report_path": str(daily_report_path),
            "model_id": str(model_id),
            "model_name": str(best.name),
        }
    )
    logger.info(
        "AFTER_HOURS_TRAINING_COMPLETE",
        extra={
            "model_id": model_id,
            "model_name": best.name,
            "rows": int(len(dataset)),
            "expectancy_bps": best.mean_expectancy_bps,
            "brier_score": best.brier_score,
            "selection_score": _candidate_selection_score(best, weights=selection_weights),
            "report_path": str(report_path),
            "daily_report_path": str(daily_report_path),
            "governance_status": status,
        },
    )
    note_after_hours_training_complete()
    return {
        "status": "trained",
        "model_id": model_id,
        "model_name": best.name,
        "model_path": str(model_path),
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "daily_report_path": str(daily_report_path),
        "governance_status": status,
        "edge_gates": gate_map,
        "rows": int(len(dataset)),
        "selection_score": _candidate_selection_score(best, weights=selection_weights),
        "brier_score": best.brier_score,
        "candidate_metrics": candidate_metrics_payload,
        "sensitivity_sweep": sensitivity_sweep,
        "prior_model_metrics": prior_model_metrics,
        "selection_weights": dict(selection_weights),
        "model_selection_retune": model_selection_retune,
        "training_data_delta": training_data_delta,
        "promoted_model_path": promoted_model_path,
        "promoted_manifest_path": promoted_manifest_path,
        "promotion": promotion,
        "roadmap": {"phase_1_week_1": phase1_week1},
        "runtime_performance_gate": runtime_performance_gate,
        "rl_overlay": rl_overlay,
    }


__all__ = ["run_after_hours_training"]
