"""After-hours ML/RL training orchestration.

This module builds cost-aware datasets, runs purged walk-forward evaluation,
trains calibrated baseline ML models, and optionally trains an RL overlay.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, time as dt_time, timedelta
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

import numpy as np

from ai_trading.config.management import get_env
from ai_trading.data.fetch import get_daily_df
from ai_trading.data.splits import PurgedGroupTimeSeriesSplit
from ai_trading.features.indicators import (
    compute_atr,
    compute_macd,
    compute_sma,
    compute_vwap,
)
from ai_trading.indicators import rsi as rsi_indicator
from ai_trading.logging import get_logger
from ai_trading.model_registry import ModelRegistry
from ai_trading.models.artifacts import write_artifact_manifest
from ai_trading.research.leakage_tests import run_leakage_guards

logger = get_logger(__name__)

FEATURE_COLUMNS: tuple[str, ...] = (
    "rsi",
    "macd",
    "atr",
    "vwap",
    "sma_50",
    "sma_200",
)
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
    support: int
    mean_expectancy_bps: float
    max_drawdown_bps: float
    turnover_ratio: float
    mean_hit_rate: float
    hit_rate_stability: float
    regime_metrics: dict[str, dict[str, float]]
    oof_probabilities: np.ndarray


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
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

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
        path_raw = str(get_env("AI_TRADING_TICKERS_CSV", "") or "").strip()
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
    frame = compute_atr(frame)
    frame = compute_vwap(frame)
    frame = compute_sma(frame, windows=(50, 200))
    close_arr = frame["close"].astype(float).to_numpy()
    frame["rsi"] = _safe_rsi(close_arr)
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
        return 1.0 / (1.0 + np.exp(-scores))
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
    hit_stability = 1.0
    if len(fold_hits) > 1:
        hit_stability = max(0.0, 1.0 - float(pstdev(fold_hits)))
    return CandidateMetrics(
        name=name,
        fold_count=valid_folds,
        support=total_support,
        mean_expectancy_bps=float(mean(fold_edges)) if fold_edges else 0.0,
        max_drawdown_bps=float(max(fold_dd)) if fold_dd else 0.0,
        turnover_ratio=float(mean(fold_turnover)) if fold_turnover else 0.0,
        mean_hit_rate=float(mean(fold_hits)) if fold_hits else 0.0,
        hit_rate_stability=float(hit_stability),
        regime_metrics=regime_metrics,
        oof_probabilities=oof_probs,
    )


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
    for regime in sorted({str(v) for v in regimes}):
        regime_mask = regimes == regime
        valid_mask = regime_mask & np.isfinite(probabilities)
        if int(np.sum(valid_mask)) < min_support:
            out[regime] = float(default_threshold)
            continue
        best_threshold = float(default_threshold)
        best_score = -float("inf")
        for threshold in _THRESHOLD_GRID:
            selected = valid_mask & (probabilities >= threshold)
            support = int(np.sum(selected))
            if support < min_support:
                continue
            score = float(np.mean(edge[selected]))
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        out[regime] = best_threshold
    return out


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


def _candidate_names() -> list[str]:
    names: list[str] = ["logreg"]
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
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    ranked = sorted(
        candidates,
        key=lambda item: (
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
                "support": item.support,
                "mean_expectancy_bps": item.mean_expectancy_bps,
                "max_drawdown_bps": item.max_drawdown_bps,
                "turnover_ratio": item.turnover_ratio,
                "mean_hit_rate": item.mean_hit_rate,
                "hit_rate_stability": item.hit_rate_stability,
                "regime_metrics": item.regime_metrics,
            }
        )
    return payload


def _fit_final_model(name: str, dataset: Any, *, seed: int):
    model = _fit_candidate_model(name, seed=seed)
    X = dataset.loc[:, FEATURE_COLUMNS]
    y = dataset["label"].astype(int)
    model.fit(X, y)
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


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, default=str, indent=2), encoding="utf-8")
    return path


def _maybe_promote_to_runtime_model_path(model_path: Path) -> tuple[str | None, str | None]:
    promote_enabled = bool(
        get_env("AI_TRADING_AFTER_HOURS_PROMOTE_MODEL_PATH", True, cast=bool)
    )
    if not promote_enabled:
        return None, None
    runtime_model_path_raw = str(get_env("AI_TRADING_MODEL_PATH", "") or "").strip()
    if not runtime_model_path_raw:
        return None, None
    runtime_model_path = Path(runtime_model_path_raw)
    runtime_model_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_model_path.write_bytes(model_path.read_bytes())
    manifest_path = write_artifact_manifest(
        model_path=str(runtime_model_path),
        model_version=f"edge_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
        training_data_range=None,
    )
    return str(runtime_model_path), str(manifest_path)


def _maybe_train_rl_overlay(
    dataset: Any,
    *,
    now_utc: datetime,
    baseline_expectancy_bps: float,
) -> dict[str, Any]:
    enabled = bool(get_env("AI_TRADING_AFTER_HOURS_RL_OVERLAY_ENABLED", False, cast=bool))
    if not enabled:
        return {"enabled": False, "trained": False, "recommend_use_rl_agent": False}
    try:
        from ai_trading.rl_trading.train import RLTrainer
    except Exception as exc:
        return {
            "enabled": True,
            "trained": False,
            "recommend_use_rl_agent": False,
            "error": str(exc),
        }
    data = dataset.loc[:, FEATURE_COLUMNS].astype(float).to_numpy()
    if data.shape[0] < 120:
        return {
            "enabled": True,
            "trained": False,
            "recommend_use_rl_agent": False,
            "reason": "insufficient_rows",
        }
    timesteps = int(get_env("AI_TRADING_AFTER_HOURS_RL_TIMESTEPS", 15000, cast=int))
    algorithm = str(get_env("AI_TRADING_AFTER_HOURS_RL_ALGO", "PPO") or "PPO").upper()
    out_dir = Path(str(get_env("AI_TRADING_AFTER_HOURS_RL_DIR", "models/after_hours_rl")))
    run_dir = out_dir / now_utc.strftime("%Y%m%d_%H%M%S")
    trainer = RLTrainer(
        algorithm=algorithm,
        total_timesteps=max(2000, timesteps),
        eval_freq=max(1000, timesteps // 5),
        early_stopping_patience=5,
        seed=int(get_env("AI_TRADING_SEED", 42, cast=int)),
    )
    results = trainer.train(
        data=data,
        env_params={
            "transaction_cost": float(get_env("AI_TRADING_RL_TRANSACTION_COST", 0.001, cast=float)),
            "slippage": float(get_env("AI_TRADING_RL_SLIPPAGE", 0.0005, cast=float)),
            "half_spread": float(get_env("AI_TRADING_RL_HALF_SPREAD", 0.0002, cast=float)),
        },
        save_path=str(run_dir),
    )
    final_eval = results.get("final_evaluation", {}) if isinstance(results, dict) else {}
    mean_reward = float(final_eval.get("mean_reward", 0.0) or 0.0)
    rl_reward_target = float(get_env("AI_TRADING_RL_MIN_MEAN_REWARD", 0.0, cast=float))
    baseline_target = float(
        get_env("AI_TRADING_RL_REQUIRE_BASELINE_EXPECTANCY_BPS", 1.0, cast=float)
    )
    recommend = mean_reward >= rl_reward_target and baseline_expectancy_bps >= baseline_target
    return {
        "enabled": True,
        "trained": True,
        "algorithm": algorithm,
        "timesteps": timesteps,
        "output_dir": str(run_dir),
        "mean_reward": mean_reward,
        "rl_reward_target": rl_reward_target,
        "baseline_expectancy_bps": baseline_expectancy_bps,
        "recommend_use_rl_agent": bool(recommend),
    }


def run_after_hours_training(*, now: datetime | None = None) -> dict[str, Any]:
    """Run after-hours ML training and optional RL overlay training."""
    import joblib
    import pandas as pd

    now_utc = (now or datetime.now(UTC)).astimezone(UTC)
    now_ny = now_utc.astimezone(ZoneInfo("America/New_York"))
    if now_ny.time() < dt_time(16, 0):
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
    best = max(
        candidate_results,
        key=lambda item: (item.mean_expectancy_bps, -item.max_drawdown_bps, item.support),
    )
    candidate_metrics_payload = _serialize_candidate_metrics(
        candidate_results,
        best_name=best.name,
    )
    thresholds_by_regime = _threshold_by_regime(
        dataset,
        best.oof_probabilities,
        default_threshold=default_threshold,
    )
    final_model = _fit_final_model(best.name, dataset, seed=seed)
    setattr(final_model, "edge_thresholds_by_regime_", thresholds_by_regime)
    setattr(final_model, "edge_global_threshold_", float(default_threshold))
    model_dir = Path(str(get_env("AI_TRADING_AFTER_HOURS_MODEL_DIR", "models/after_hours")))
    model_dir.mkdir(parents=True, exist_ok=True)
    model_version = f"{best.name}_{now_utc.strftime('%Y%m%d_%H%M%S')}"
    model_path = model_dir / f"ml_edge_{model_version}.joblib"
    joblib.dump(final_model, model_path)
    manifest_path = write_artifact_manifest(
        model_path=str(model_path),
        model_version=model_version,
        training_data_range={
            "start": str(dataset["timestamp"].iloc[0]),
            "end": str(dataset["timestamp"].iloc[-1]),
        },
    )
    dataset_fp = _dataset_fingerprint(dataset, symbols=symbols, cost_floor_bps=cost_floor_bps)
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
        },
        dataset_fingerprint=dataset_fp,
        tags=["after_hours", "cost_aware", "purged_walk_forward"],
        activate=True,
    )
    targets = _edge_targets()
    gate_map = _meets_edge_targets(best, targets)
    gates_passed = all(gate_map.values())
    status = "shadow"
    auto_promote = bool(get_env("AI_TRADING_AFTER_HOURS_AUTO_PROMOTE", False, cast=bool))
    if gates_passed and auto_promote:
        status = "production"
    registry.update_governance_status(
        model_id,
        status,
        extra={
            "edge_gates": gate_map,
            "edge_targets": {
                "min_expectancy_bps": targets.min_expectancy_bps,
                "max_drawdown_bps": targets.max_drawdown_bps,
                "max_turnover_ratio": targets.max_turnover_ratio,
                "min_hit_rate_stability": targets.min_hit_rate_stability,
            },
            "metrics": {
                "mean_expectancy_bps": best.mean_expectancy_bps,
                "max_drawdown_bps": best.max_drawdown_bps,
                "turnover_ratio": best.turnover_ratio,
                "mean_hit_rate": best.mean_hit_rate,
                "hit_rate_stability": best.hit_rate_stability,
            },
        },
    )
    promoted_model_path, promoted_manifest_path = _maybe_promote_to_runtime_model_path(model_path)
    rl_overlay = _maybe_train_rl_overlay(
        dataset,
        now_utc=now_utc,
        baseline_expectancy_bps=best.mean_expectancy_bps,
    )
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
            "governance_status": status,
        },
        "metrics": {
            "mean_expectancy_bps": best.mean_expectancy_bps,
            "max_drawdown_bps": best.max_drawdown_bps,
            "turnover_ratio": best.turnover_ratio,
            "mean_hit_rate": best.mean_hit_rate,
            "hit_rate_stability": best.hit_rate_stability,
            "fold_count": best.fold_count,
            "support": best.support,
        },
        "fill_quality": fill_quality,
        "candidate_metrics": candidate_metrics_payload,
        "regime_metrics": best.regime_metrics,
        "thresholds_by_regime": thresholds_by_regime,
        "edge_gates": gate_map,
        "rl_overlay": rl_overlay,
        "runtime_promotion": {
            "model_path": promoted_model_path,
            "manifest_path": promoted_manifest_path,
        },
    }
    report_dir = Path(
        str(get_env("AI_TRADING_AFTER_HOURS_REPORT_DIR", "runtime/research_reports"))
    )
    report_path = _write_json(
        report_dir / f"after_hours_training_{now_utc.strftime('%Y%m%d')}.json",
        report,
    )
    logger.info(
        "AFTER_HOURS_TRAINING_COMPLETE",
        extra={
            "model_id": model_id,
            "model_name": best.name,
            "rows": int(len(dataset)),
            "expectancy_bps": best.mean_expectancy_bps,
            "report_path": str(report_path),
            "governance_status": status,
        },
    )
    return {
        "status": "trained",
        "model_id": model_id,
        "model_name": best.name,
        "model_path": str(model_path),
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "governance_status": status,
        "edge_gates": gate_map,
        "rows": int(len(dataset)),
        "candidate_metrics": candidate_metrics_payload,
        "promoted_model_path": promoted_model_path,
        "promoted_manifest_path": promoted_manifest_path,
        "rl_overlay": rl_overlay,
    }


__all__ = ["run_after_hours_training"]
