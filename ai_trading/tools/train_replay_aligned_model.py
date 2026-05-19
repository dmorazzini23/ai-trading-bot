"""Train an offline replay-aligned edge model from local OHLCV bars."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ai_trading.data.historical_bars import HistoricalBarLoadReport, load_historical_bars
from ai_trading.features.indicators import (
    compute_atr,
    compute_macd,
    compute_macds,
    compute_sma,
    compute_vwap,
)
from ai_trading.logging import get_logger
from ai_trading.models.artifacts import write_artifact_manifest
from ai_trading.config.management import get_env
from ai_trading.paths import CACHE_DIR
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path
from ai_trading.tools.offline_replay import (
    LiveCostReplayModel,
    _augment_model_features,
    _load_live_cost_replay_model,
    _replay_session_regime,
    _replay_slippage_bps,
    _safe_rsi,
    _sanitize_model_feature_index,
)

logger = get_logger(__name__)

REPLAY_ALIGNED_FEATURE_COLUMNS: tuple[str, ...] = (
    "rsi",
    "macd",
    "atr",
    "vwap",
    "sma_50",
    "sma_200",
    "signal",
    "atr_pct",
    "vwap_distance",
    "sma_spread",
    "macd_signal_gap",
    "rsi_centered",
)
_FEATURE_CACHE_SCHEMA_VERSION = "replay_aligned_features_v1"


@dataclass(frozen=True)
class ReplayAlignedTrainingConfig:
    data_dir: str
    symbols: tuple[str, ...]
    horizon_bars: int
    label_objective: str
    fee_bps: float
    slippage_bps: float
    min_net_edge_bps: float
    train_fraction: float
    model_type: str
    edge_global_threshold: float | None
    live_cost_model_path: str | None = None
    live_cost_model_requested: bool = False
    live_cost_model_usable: bool = False
    training_cache_enabled: bool = True
    training_cache_dir: str | None = None


def _resolve_symbol_paths(data_dir: Path, symbols: str) -> dict[str, Path]:
    requested = {item.strip().upper() for item in symbols.split(",") if item.strip()}
    paths: dict[str, Path] = {}
    for csv_path in sorted(data_dir.glob("*.csv")):
        symbol = csv_path.stem.upper()
        if requested and symbol not in requested:
            continue
        paths[symbol] = csv_path
    if not paths:
        raise ValueError("No matching CSV files found for replay-aligned training")
    return paths


def _feature_frame(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    work = _sanitize_model_feature_index(frame.copy(), symbol=symbol)
    work = compute_macd(work)
    work = compute_macds(work)
    work = compute_atr(work)
    work = compute_vwap(work)
    work = compute_sma(work, windows=(50, 200))
    close_arr = pd.to_numeric(work.get("close"), errors="coerce").to_numpy(dtype=float)
    work["rsi"] = _safe_rsi(close_arr)
    work = _augment_model_features(work)
    for name in REPLAY_ALIGNED_FEATURE_COLUMNS:
        if name not in work.columns:
            work[name] = np.nan
    features = work[list(REPLAY_ALIGNED_FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    return cast(pd.DataFrame, features.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0))


def _env_bool(name: str, default: bool) -> bool:
    raw = get_env(name, None, cast=str, resolve_aliases=False)
    if raw in (None, ""):
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _training_cache_dir(raw: str | Path | None = None) -> Path:
    configured = str(raw or "").strip() or str(
        get_env(
            "AI_TRADING_REPLAY_ALIGNED_TRAINING_CACHE_DIR",
            "",
            cast=str,
            resolve_aliases=False,
        )
        or ""
    ).strip()
    return Path(configured).expanduser() if configured else CACHE_DIR / "training" / "replay_aligned"


def _resolve_output_dir(path: str | Path) -> Path:
    target = Path(path).expanduser()
    if target.is_absolute():
        return target
    return resolve_runtime_artifact_path(
        target,
        default_relative=str(target),
        for_write=True,
    )


def _resolve_input_dir(path: str | Path) -> Path:
    target = Path(path).expanduser()
    if target.is_absolute():
        return target
    return resolve_runtime_artifact_path(
        target,
        default_relative=str(target),
        for_write=False,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _symbol_feature_cache_key(csv_path: Path, *, timestamp_col: str, symbol: str) -> str:
    stat = csv_path.stat()
    payload = {
        "schema": _FEATURE_CACHE_SCHEMA_VERSION,
        "path": str(csv_path.resolve()),
        "symbol": symbol,
        "timestamp_col": timestamp_col,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _file_sha256(csv_path),
        "feature_columns": list(REPLAY_ALIGNED_FEATURE_COLUMNS),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_or_build_symbol_features(
    symbol: str,
    csv_path: Path,
    *,
    timestamp_col: str,
    use_training_cache: bool,
    training_cache_dir: Path | None,
    allow_research_synthetic_timestamps: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, HistoricalBarLoadReport]:
    cache_dir = training_cache_dir or _training_cache_dir()
    cache_path = cache_dir / f"{symbol}_{_symbol_feature_cache_key(csv_path, timestamp_col=timestamp_col, symbol=symbol)}.pkl"
    frame, report = load_historical_bars(
        csv_path,
        timestamp_col=timestamp_col,
        require_timestamp=True,
        allow_research_synthetic=allow_research_synthetic_timestamps,
    )
    if use_training_cache and cache_path.exists():
        try:
            cached = pd.read_pickle(cache_path)
        except (OSError, ValueError, TypeError, AttributeError):
            cached = None
        if isinstance(cached, dict):
            frame = cached.get("frame")
            features = cached.get("features")
            if isinstance(frame, pd.DataFrame) and isinstance(features, pd.DataFrame):
                return frame, features, report
    if frame.empty:
        return frame, pd.DataFrame(), report
    try:
        raw = pd.read_csv(csv_path)
    except (OSError, ValueError):
        raw = pd.DataFrame()
    if not raw.empty:
        lower_map = {str(col).lower(): col for col in raw.columns}
        spread_col = lower_map.get("spread_bps")
        ts_col = lower_map.get(str(timestamp_col).lower()) or lower_map.get("timestamp")
        if spread_col is not None and ts_col is not None:
            spread_index = pd.to_datetime(raw[ts_col], errors="coerce", utc=True, format="mixed")
            spread = pd.Series(pd.to_numeric(raw[spread_col], errors="coerce").to_numpy(), index=spread_index)
            frame["spread_bps"] = spread.reindex(frame.index).fillna(0.0).to_numpy(dtype=float)
    features = _feature_frame(frame, symbol=symbol)
    if use_training_cache:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            pd.to_pickle({"frame": frame, "features": features}, cache_path)
        except (OSError, ValueError, TypeError):
            logger.debug(
                "REPLAY_ALIGNED_TRAINING_CACHE_WRITE_FAILED",
                extra={"path": str(cache_path), "symbol": symbol},
                exc_info=True,
            )
    return frame, features, report


def _normalize_label_objective(value: str) -> str:
    normalized = str(value or "net_markout").strip().lower().replace("-", "_")
    aliases = {
        "net": "net_markout",
        "markout": "net_markout",
        "spread": "spread_adjusted",
        "spread_adjusted_markout": "spread_adjusted",
        "risk": "risk_adjusted",
        "risk_adjusted_markout": "risk_adjusted",
        "excursion": "mae_mfe",
        "mae_mfe_markout": "mae_mfe",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {"net_markout", "spread_adjusted", "risk_adjusted", "mae_mfe"}
    if normalized not in allowed:
        raise ValueError(
            "Unsupported label objective: "
            f"{value}. Expected one of {', '.join(sorted(allowed))}."
        )
    return normalized


def _excursion_bps(
    close: pd.Series,
    *,
    horizon_bars: int,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    horizon = max(1, int(horizon_bars))
    close_float = pd.to_numeric(close, errors="coerce").astype(float)
    high_float = pd.to_numeric(high, errors="coerce").astype(float) if high is not None else close_float
    low_float = pd.to_numeric(low, errors="coerce").astype(float) if low is not None else close_float
    base = close_float.replace(0.0, np.nan)
    adverse_returns: list[pd.Series] = []
    favorable_returns: list[pd.Series] = []
    for offset in range(1, horizon + 1):
        adverse_returns.append(((low_float.shift(-offset) / base) - 1.0) * 10000.0)
        favorable_returns.append(((high_float.shift(-offset) / base) - 1.0) * 10000.0)
    max_adverse = pd.concat(adverse_returns, axis=1).min(axis=1, skipna=True)
    max_favorable = pd.concat(favorable_returns, axis=1).max(axis=1, skipna=True)
    return cast(pd.Series, max_adverse), cast(pd.Series, max_favorable)


def _label_score(
    *,
    objective: str,
    net_long_bps: pd.Series,
    spread_adjusted_long_bps: pd.Series,
    max_adverse_excursion_bps: pd.Series,
    max_favorable_excursion_bps: pd.Series,
    round_trip_cost_bps: pd.Series,
) -> pd.Series:
    normalized = _normalize_label_objective(objective)
    if normalized == "net_markout":
        return net_long_bps
    if normalized == "spread_adjusted":
        return spread_adjusted_long_bps
    adverse_penalty = max_adverse_excursion_bps.clip(upper=0.0).abs()
    favorable_credit = max_favorable_excursion_bps.clip(lower=0.0)
    if normalized == "risk_adjusted":
        return net_long_bps - (0.75 * adverse_penalty) + (0.25 * favorable_credit)
    return net_long_bps - adverse_penalty + (0.50 * favorable_credit) - (0.50 * round_trip_cost_bps)


def _build_symbol_dataset(
    symbol: str,
    csv_path: Path,
    *,
    timestamp_col: str,
    horizon_bars: int,
    label_objective: str,
    fee_bps: float,
    slippage_bps: float,
    min_net_edge_bps: float,
    live_cost_model: LiveCostReplayModel | None = None,
    use_training_cache: bool = True,
    training_cache_dir: Path | None = None,
    allow_research_synthetic_timestamps: bool = False,
) -> pd.DataFrame:
    frame, features, report = _load_or_build_symbol_features(
        symbol,
        csv_path,
        timestamp_col=timestamp_col,
        use_training_cache=use_training_cache,
        training_cache_dir=training_cache_dir,
        allow_research_synthetic_timestamps=allow_research_synthetic_timestamps,
    )
    reports = getattr(_build_symbol_dataset, "_load_reports", None)
    if isinstance(reports, dict):
        reports[symbol] = report
    if frame.empty:
        return pd.DataFrame()
    if not frame.index.is_monotonic_increasing:
        frame = frame.sort_index(kind="mergesort")
        features = features.reindex(frame.index)
    close = pd.to_numeric(frame["close"], errors="coerce").astype(float)
    future_close = close.shift(-int(horizon_bars))
    gross_long_bps = ((future_close / close.replace(0.0, np.nan)) - 1.0) * 10000.0
    if "spread_bps" in frame.columns:
        spread_cost_bps = pd.to_numeric(frame["spread_bps"], errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)
    else:
        spread_cost_bps = pd.Series(0.0, index=frame.index, dtype=float)
    cost_cfg = cast(
        Any,
        argparse.Namespace(
            live_cost_model=live_cost_model,
            slippage_bps=max(0.0, float(slippage_bps)),
        ),
    )
    entry_slippage = pd.Series(
        [
            _replay_slippage_bps(
                cost_cfg,
                symbol=symbol,
                side="buy",
                ts=ts,
            )
            if live_cost_model is not None
            else max(0.0, float(slippage_bps))
            for ts in frame.index
        ],
        index=frame.index,
        dtype=float,
    )
    exit_slippage = pd.Series(
        [
            _replay_slippage_bps(
                cost_cfg,
                symbol=symbol,
                side="sell",
                ts=ts,
            )
            if live_cost_model is not None
            else max(0.0, float(slippage_bps))
            for ts in frame.index
        ],
        index=frame.index,
        dtype=float,
    )
    round_trip_cost_bps = (
        spread_cost_bps + (2.0 * max(0.0, float(fee_bps))) + entry_slippage + exit_slippage
    )
    spread_adjusted_long_bps = gross_long_bps - spread_cost_bps
    net_long_bps = gross_long_bps - round_trip_cost_bps
    max_adverse_excursion_bps, max_favorable_excursion_bps = _excursion_bps(
        close,
        horizon_bars=horizon_bars,
        high=frame["high"] if "high" in frame.columns else None,
        low=frame["low"] if "low" in frame.columns else None,
    )
    risk_adjusted_net_bps = _label_score(
        objective="risk_adjusted",
        net_long_bps=net_long_bps,
        spread_adjusted_long_bps=spread_adjusted_long_bps,
        max_adverse_excursion_bps=max_adverse_excursion_bps,
        max_favorable_excursion_bps=max_favorable_excursion_bps,
        round_trip_cost_bps=round_trip_cost_bps,
    )
    normalized_objective = _normalize_label_objective(label_objective)
    label_score_bps = _label_score(
        objective=normalized_objective,
        net_long_bps=net_long_bps,
        spread_adjusted_long_bps=spread_adjusted_long_bps,
        max_adverse_excursion_bps=max_adverse_excursion_bps,
        max_favorable_excursion_bps=max_favorable_excursion_bps,
        round_trip_cost_bps=round_trip_cost_bps,
    )
    out = features.copy()
    out["symbol"] = symbol
    out["timestamp"] = frame.index
    label_end_timestamp = pd.Series(frame.index, index=frame.index).shift(-int(horizon_bars))
    out["label_end_timestamp"] = label_end_timestamp.to_numpy()
    out["session_regime"] = [_replay_session_regime(ts) for ts in frame.index]
    out["gross_long_bps"] = gross_long_bps.to_numpy(dtype=float)
    out["spread_cost_bps"] = spread_cost_bps.to_numpy(dtype=float)
    out["spread_adjusted_long_bps"] = spread_adjusted_long_bps.to_numpy(dtype=float)
    out["entry_slippage_bps"] = entry_slippage.to_numpy(dtype=float)
    out["exit_slippage_bps"] = exit_slippage.to_numpy(dtype=float)
    out["round_trip_cost_bps"] = round_trip_cost_bps.to_numpy(dtype=float)
    out["net_long_bps"] = net_long_bps.to_numpy(dtype=float)
    out["max_adverse_excursion_bps"] = max_adverse_excursion_bps.to_numpy(dtype=float)
    out["max_favorable_excursion_bps"] = max_favorable_excursion_bps.to_numpy(dtype=float)
    out["risk_adjusted_net_bps"] = risk_adjusted_net_bps.to_numpy(dtype=float)
    out["label_score_bps"] = label_score_bps.to_numpy(dtype=float)
    out["label_objective"] = normalized_objective
    out["target"] = (out["label_score_bps"] > float(min_net_edge_bps)).astype(int)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            *REPLAY_ALIGNED_FEATURE_COLUMNS,
            "timestamp",
            "label_end_timestamp",
            "net_long_bps",
            "label_score_bps",
            "target",
        ]
    )
    return cast(pd.DataFrame, out)


def build_training_dataset(
    *,
    data_dir: Path,
    symbols: str = "",
    timestamp_col: str = "timestamp",
    horizon_bars: int = 1,
    label_objective: str = "net_markout",
    fee_bps: float = 1.0,
    slippage_bps: float = 2.0,
    min_net_edge_bps: float = 0.0,
    live_cost_model: LiveCostReplayModel | None = None,
    use_training_cache: bool | None = None,
    training_cache_dir: Path | None = None,
    allow_research_synthetic_timestamps: bool = False,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    load_reports: dict[str, HistoricalBarLoadReport] = {}
    setattr(_build_symbol_dataset, "_load_reports", load_reports)
    cache_enabled = _env_bool(
        "AI_TRADING_REPLAY_ALIGNED_TRAINING_CACHE_ENABLED",
        True,
    ) if use_training_cache is None else bool(use_training_cache)
    try:
        for symbol, csv_path in _resolve_symbol_paths(data_dir, symbols).items():
            symbol_rows = _build_symbol_dataset(
                symbol,
                csv_path,
                timestamp_col=timestamp_col,
                horizon_bars=horizon_bars,
                label_objective=label_objective,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                min_net_edge_bps=min_net_edge_bps,
                live_cost_model=live_cost_model,
                use_training_cache=cache_enabled,
                training_cache_dir=training_cache_dir,
                allow_research_synthetic_timestamps=allow_research_synthetic_timestamps,
            )
            if not symbol_rows.empty:
                rows.append(symbol_rows)
    finally:
        setattr(_build_symbol_dataset, "_load_reports", None)
    if not rows:
        return pd.DataFrame()
    dataset = pd.concat(rows, axis=0, ignore_index=True)
    dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], errors="coerce", utc=True)
    dataset["label_end_timestamp"] = pd.to_datetime(
        dataset["label_end_timestamp"],
        errors="coerce",
        utc=True,
    )
    dataset = dataset.dropna(subset=["timestamp", "label_end_timestamp"]).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    dataset.attrs["load_reports"] = {
        symbol: report.as_dict()
        for symbol, report in sorted(load_reports.items())
    }
    return cast(pd.DataFrame, dataset)


def _training_authority(dataset: pd.DataFrame) -> dict[str, Any]:
    reports_raw = dataset.attrs.get("load_reports")
    reports = reports_raw if isinstance(reports_raw, Mapping) else {}
    timestamp_authoritative = all(
        bool(report.get("timestamp_authoritative"))
        for report in reports.values()
        if isinstance(report, Mapping)
    )
    source_providers = sorted(
        {
            str(provider).strip().lower()
            for report in reports.values()
            if isinstance(report, Mapping)
            for provider in report.get("source_providers", ())
            if str(provider).strip()
        }
    )
    research_synthetic = any(
        bool(report.get("research_synthetic"))
        for report in reports.values()
        if isinstance(report, Mapping)
    )
    return {
        "runtime_authority": False,
        "promotion_authority": False,
        "live_money_authority": False,
        "research_only": True,
        "timestamp_authoritative": bool(timestamp_authoritative and bool(reports)),
        "research_synthetic": bool(research_synthetic),
        "source_providers": source_providers,
    }


def _make_model(model_type: str, *, random_state: int) -> Any:
    normalized = str(model_type or "logistic").strip().lower()
    if normalized == "logistic":
        return Pipeline(
            steps=[
                ("standardscaler", StandardScaler()),
                (
                    "logisticregression",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if normalized == "random_forest":
        return RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        )
    if normalized == "hist_gradient":
        estimator = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            l2_regularization=0.05,
            min_samples_leaf=20,
            random_state=random_state,
        )
        return CalibratedClassifierCV(estimator=estimator, cv=3)
    raise ValueError(f"Unsupported model type: {model_type}")


def _feature_importance(model: Any) -> list[dict[str, Any]]:
    """Return lightweight feature attribution for candidate triage artifacts."""
    estimator = model
    if isinstance(model, Pipeline):
        estimator = model.steps[-1][1] if model.steps else model
    raw: Any = None
    if hasattr(estimator, "coef_"):
        coef = np.asarray(getattr(estimator, "coef_"), dtype=float)
        if coef.ndim == 2 and coef.shape[0] >= 1:
            raw = coef[0]
    elif hasattr(estimator, "feature_importances_"):
        raw = np.asarray(getattr(estimator, "feature_importances_"), dtype=float)
    if raw is None:
        return []
    values = np.asarray(raw, dtype=float).reshape(-1)
    if values.size != len(REPLAY_ALIGNED_FEATURE_COLUMNS):
        return []
    rows = [
        {
            "feature": feature,
            "importance": float(abs(value)),
            "signed_weight": float(value),
        }
        for feature, value in zip(REPLAY_ALIGNED_FEATURE_COLUMNS, values, strict=True)
        if np.isfinite(value)
    ]
    rows.sort(key=lambda item: cast(float, item["importance"]), reverse=True)
    return rows


def _live_cost_request_metadata(
    args: argparse.Namespace,
    live_cost_model: LiveCostReplayModel | None,
) -> dict[str, Any]:
    explicit_path = getattr(args, "live_cost_model_json", None)
    requested_flag = getattr(args, "use_live_cost_model", None)
    requested = explicit_path is not None or bool(requested_flag)
    path: Path | None = Path(explicit_path).expanduser() if explicit_path is not None else None
    if path is None and bool(requested_flag):
        path = resolve_runtime_artifact_path(
            str(
                get_env(
                    "AI_TRADING_LIVE_COST_MODEL_PATH",
                    "runtime/live_cost_model_latest.json",
                    cast=str,
                    resolve_aliases=False,
                )
                or "runtime/live_cost_model_latest.json"
            ),
            default_relative="runtime/live_cost_model_latest.json",
        )
    out: dict[str, Any] = {
        "requested": bool(requested),
        "enabled": live_cost_model is not None,
        "usable": live_cost_model is not None,
        "path": live_cost_model.path if live_cost_model is not None else (str(path) if path is not None else None),
        "bucket_count": live_cost_model.bucket_count if live_cost_model is not None else 0,
        "generated_at": live_cost_model.generated_at if live_cost_model is not None else None,
        "status": live_cost_model.status if live_cost_model is not None else None,
        "freshness_status": live_cost_model.freshness_status if live_cost_model is not None else None,
        "source_sha256": live_cost_model.source_sha256 if live_cost_model is not None else None,
        "reason": "loaded" if live_cost_model is not None else ("not_requested" if not requested else "not_loaded"),
    }
    if live_cost_model is not None or not requested or path is None:
        return out
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        out["reason"] = "artifact_unavailable"
        return out
    if not isinstance(payload, Mapping) or payload.get("artifact_type") != "live_cost_model":
        out["reason"] = "invalid_artifact_type"
        return out
    status = payload.get("status")
    if isinstance(status, Mapping):
        out["status"] = status.get("status")
        out["available"] = bool(status.get("available"))
        if not bool(status.get("available")):
            out["reason"] = "status_unavailable"
            return out
        status_text = str(status.get("status") or "").strip().lower()
        if status_text != "ready":
            out["reason"] = f"status_{status_text or 'missing'}"
            return out
    else:
        out["reason"] = "status_missing"
        return out
    rows = payload.get("by_symbol_side_session")
    if not isinstance(rows, list):
        out["reason"] = "buckets_missing"
        return out
    sufficient = [row for row in rows if isinstance(row, Mapping) and bool(row.get("sufficient_samples"))]
    out["bucket_count"] = len(sufficient)
    out["reason"] = "insufficient_bucket_samples" if not sufficient else "not_loaded"
    return out


def _positive_class_index(model: Any) -> int:
    classes = getattr(model, "classes_", None)
    if classes is None:
        return 1
    try:
        class_list = list(classes)
    except TypeError:
        return 1
    for idx, value in enumerate(class_list):
        if int(value) == 1:
            return idx
    return min(1, max(0, len(class_list) - 1))


def _evaluate_probabilities(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, Any]:
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    out: dict[str, Any] = {
        "rows": int(y_arr.size),
        "positive_rate": float(y_arr.mean()) if y_arr.size else 0.0,
        "mean_probability": float(p_arr.mean()) if p_arr.size else 0.0,
    }
    if y_arr.size and len(set(y_arr.tolist())) >= 2:
        out["roc_auc"] = float(roc_auc_score(y_arr, p_arr))
        out["log_loss"] = float(log_loss(y_arr, np.column_stack([1.0 - p_arr, p_arr]), labels=[0, 1]))
        out["brier_score"] = float(brier_score_loss(y_arr, p_arr))
    else:
        out["roc_auc"] = None
        out["log_loss"] = None
        out["brier_score"] = None
    return out


def _threshold_report(
    dataset: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    allow_short_labels: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    score = (2.0 * p) - 1.0
    confidence = np.maximum(p, 1.0 - p) if allow_short_labels else p
    net = pd.to_numeric(dataset["net_long_bps"], errors="coerce").to_numpy(dtype=float)
    for confidence_threshold in (0.52, 0.58, 0.62, 0.66):
        for entry_threshold in (0.05, 0.10, 0.15, 0.20):
            if allow_short_labels:
                mask = (confidence >= confidence_threshold) & (np.abs(score) >= entry_threshold)
            else:
                mask = (confidence >= confidence_threshold) & (score >= entry_threshold)
            if not bool(mask.any()):
                mean_net = None
                total_net = 0.0
                positive_rate = None
            else:
                signed_net = np.where(score[mask] >= 0.0, net[mask], -net[mask]) if allow_short_labels else net[mask]
                mean_net = float(np.nanmean(signed_net))
                total_net = float(np.nansum(signed_net))
                positive_rate = float(np.nanmean(signed_net > 0.0))
            rows.append(
                {
                    "confidence_threshold": float(confidence_threshold),
                    "entry_score_threshold": float(entry_threshold),
                    "candidates": int(mask.sum()),
                    "mean_net_markout_bps": mean_net,
                    "total_net_markout_bps": total_net,
                    "positive_rate": positive_rate,
                }
            )
    rows.sort(
        key=lambda item: (
            item["mean_net_markout_bps"] is not None and int(item.get("candidates", 0) or 0) > 0,
            float(item["mean_net_markout_bps"] or -1e9),
            int(item.get("candidates", 0) or 0),
        ),
        reverse=True,
    )
    return rows


def _threshold_report_by_regime(
    dataset: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    min_samples: int = 25,
    allow_short_labels: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    if "session_regime" not in dataset.columns:
        return {}
    reports: dict[str, list[dict[str, Any]]] = {}
    regimes = dataset["session_regime"].astype(str).str.lower()
    for regime in sorted(regime for regime in regimes.unique().tolist() if regime):
        mask = regimes == regime
        if int(mask.sum()) < int(min_samples):
            continue
        reports[regime] = _threshold_report(
            dataset.loc[mask].copy(),
            probabilities[mask.to_numpy()],
            allow_short_labels=allow_short_labels,
        )
    return reports


def _best_thresholds_by_regime(
    reports: Mapping[str, list[dict[str, Any]]],
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for regime, rows in reports.items():
        if not rows:
            continue
        best = next(
            (
                row
                for row in rows
                if int(row.get("candidates", 0) or 0) > 0
                and row.get("mean_net_markout_bps") is not None
            ),
            None,
        )
        if best is None:
            continue
        value = best.get("confidence_threshold")
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        thresholds[str(regime)] = float(np.clip(parsed, 0.0, 0.99))
    return thresholds


def _optional_threshold(value: Any) -> float | None:
    if value in (None, ""):
        return None
    threshold = float(value)
    if threshold <= 0.0:
        return None
    return float(np.clip(threshold, 0.0, 0.99))


def _attach_model_metadata(
    model: Any,
    *,
    edge_global_threshold: float | None,
    edge_thresholds_by_regime: Mapping[str, float] | None = None,
) -> None:
    for name, value in (
        ("edge_score_orientation_", "direct"),
        ("edge_score_semantics_", "long_probability"),
        ("replay_aligned_objective_", "one_bar_net_markout"),
        ("replay_label_sides_", np.asarray(["buy"], dtype=object)),
        ("supports_short_scores_", False),
        ("feature_names_in_", np.asarray(REPLAY_ALIGNED_FEATURE_COLUMNS, dtype=object)),
        ("classes_", np.asarray(getattr(model, "classes_", np.asarray([0, 1])), dtype=int)),
    ):
        if hasattr(model, name) and name in {"feature_names_in_", "classes_"}:
            continue
        try:
            setattr(model, name, value)
        except AttributeError:
            logger.debug(
                "REPLAY_ALIGNED_MODEL_METADATA_READONLY",
                extra={"attribute": name, "model_type": type(model).__name__},
            )
    if edge_global_threshold is not None:
        try:
            setattr(model, "edge_global_threshold_", float(edge_global_threshold))
        except AttributeError:
            logger.debug(
                "REPLAY_ALIGNED_MODEL_METADATA_READONLY",
                extra={"attribute": "edge_global_threshold_", "model_type": type(model).__name__},
            )
    if edge_thresholds_by_regime:
        try:
            setattr(
                model,
                "edge_thresholds_by_regime_",
                {str(key): float(value) for key, value in edge_thresholds_by_regime.items()},
            )
        except AttributeError:
            logger.debug(
                "REPLAY_ALIGNED_MODEL_METADATA_READONLY",
                extra={"attribute": "edge_thresholds_by_regime_", "model_type": type(model).__name__},
            )


def _split_train_validation_with_purge(
    dataset: pd.DataFrame,
    *,
    train_fraction: float,
    horizon_bars: int,
    embargo_bars: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cutoff_idx = max(1, min(len(dataset) - 1, int(len(dataset) * float(train_fraction))))
    train = dataset.iloc[:cutoff_idx].copy()
    validation = dataset.iloc[cutoff_idx:].copy()
    validation_start = pd.to_datetime(validation["timestamp"], errors="coerce", utc=True).min()
    purged_train_rows = 0
    embargoed_train_rows = 0
    if not pd.isna(validation_start) and "label_end_timestamp" in train.columns:
        label_end = pd.to_datetime(train["label_end_timestamp"], errors="coerce", utc=True)
        keep_mask = label_end < validation_start
        purged_train_rows = int((~keep_mask).sum())
        train = train.loc[keep_mask].copy()
    embargo_count = max(0, int(0 if embargo_bars is None else embargo_bars))
    if embargo_count > 0 and not train.empty:
        embargoed_train_rows = int(min(embargo_count, len(train)))
        train = train.iloc[:-embargoed_train_rows].copy() if embargoed_train_rows < len(train) else train.iloc[0:0].copy()
    diagnostics = {
        "initial_train_rows": int(cutoff_idx),
        "initial_validation_rows": int(len(dataset) - cutoff_idx),
        "purged_train_rows": int(purged_train_rows),
        "embargoed_train_rows": int(embargoed_train_rows),
        "embargo_bars": int(embargo_count),
        "horizon_bars": int(horizon_bars),
        "validation_start": None if pd.isna(validation_start) else str(validation_start),
    }
    return train, validation, diagnostics


def train_replay_aligned_model(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = _resolve_input_dir(args.data_dir)
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    live_cost_model = _load_live_cost_replay_model(
        argparse.Namespace(
            live_cost_model_json=getattr(args, "live_cost_model_json", None),
            use_live_cost_model=getattr(args, "use_live_cost_model", None),
        )
    )
    dataset = build_training_dataset(
        data_dir=data_dir,
        symbols=str(args.symbols or ""),
        timestamp_col=str(args.timestamp_col),
        horizon_bars=int(args.horizon_bars),
        label_objective=str(getattr(args, "label_objective", "net_markout")),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slippage_bps),
        min_net_edge_bps=float(args.min_net_edge_bps),
        live_cost_model=live_cost_model,
        use_training_cache=getattr(args, "training_cache", None),
        training_cache_dir=(
            _training_cache_dir(getattr(args, "training_cache_dir", None))
            if getattr(args, "training_cache_dir", None)
            else None
        ),
        allow_research_synthetic_timestamps=bool(
            getattr(args, "allow_research_synthetic_timestamps", False)
        ),
    )
    if dataset.empty:
        raise RuntimeError("Replay-aligned training dataset is empty")
    if dataset["target"].nunique() < 2:
        raise RuntimeError("Replay-aligned target has fewer than two classes")

    train, validation, split_diagnostics = _split_train_validation_with_purge(
        dataset,
        train_fraction=float(args.train_fraction),
        horizon_bars=int(args.horizon_bars),
    )
    if train.empty or validation.empty:
        raise RuntimeError("Train/validation split is empty after horizon purge")
    if train["target"].nunique() < 2 or validation["target"].nunique() < 2:
        raise RuntimeError("Train/validation split must contain both target classes")

    model = _make_model(str(args.model_type), random_state=int(args.random_state))
    X_train = train[list(REPLAY_ALIGNED_FEATURE_COLUMNS)].astype(float)
    y_train = train["target"].astype(int)
    X_validation = validation[list(REPLAY_ALIGNED_FEATURE_COLUMNS)].astype(float)
    y_validation = validation["target"].astype(int)
    model.fit(X_train, y_train)
    positive_index = _positive_class_index(model)
    probabilities = np.asarray(model.predict_proba(X_validation), dtype=float)[:, positive_index]

    edge_global_threshold = _optional_threshold(getattr(args, "edge_global_threshold", None))
    validation_report = _evaluate_probabilities(y_validation, probabilities)
    threshold_report = _threshold_report(validation, probabilities)
    threshold_report_by_regime = _threshold_report_by_regime(validation, probabilities)
    edge_thresholds_by_regime = _best_thresholds_by_regime(threshold_report_by_regime)
    _attach_model_metadata(
        model,
        edge_global_threshold=edge_global_threshold,
        edge_thresholds_by_regime=edge_thresholds_by_regime,
    )
    feature_importance = _feature_importance(model)
    live_cost_metadata = _live_cost_request_metadata(args, live_cost_model)

    model_name = str(args.model_name or f"replay_aligned_{args.model_type}").strip()
    model_path = output_dir / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    config = ReplayAlignedTrainingConfig(
        data_dir=str(data_dir),
        symbols=tuple(sorted(dataset["symbol"].astype(str).str.upper().unique().tolist())),
        horizon_bars=int(args.horizon_bars),
        label_objective=_normalize_label_objective(
            str(getattr(args, "label_objective", "net_markout"))
        ),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slippage_bps),
        min_net_edge_bps=float(args.min_net_edge_bps),
        train_fraction=float(args.train_fraction),
        model_type=str(args.model_type),
        edge_global_threshold=edge_global_threshold,
        live_cost_model_path=live_cost_model.path if live_cost_model is not None else None,
        live_cost_model_requested=bool(live_cost_metadata.get("requested")),
        live_cost_model_usable=bool(live_cost_metadata.get("usable")),
        training_cache_enabled=bool(
            _env_bool("AI_TRADING_REPLAY_ALIGNED_TRAINING_CACHE_ENABLED", True)
            if getattr(args, "training_cache", None) is None
            else getattr(args, "training_cache", True)
        ),
        training_cache_dir=str(_training_cache_dir(getattr(args, "training_cache_dir", None))),
    )
    manifest_path = write_artifact_manifest(
        model_path=model_path,
        model_version=f"replay_aligned_{args.model_type}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
        training_data_range={
            "start": str(dataset["timestamp"].min()),
            "end": str(dataset["timestamp"].max()),
        },
        metadata={
            "strategy": "replay_aligned_markout",
            "feature_columns": list(REPLAY_ALIGNED_FEATURE_COLUMNS),
            "objective": f"{config.horizon_bars}_bar_{config.label_objective}_binary",
            "config": asdict(config),
            "authority": _training_authority(dataset),
            "thresholds_by_regime": edge_thresholds_by_regime,
            "feature_importance": feature_importance[:25],
            "live_cost_model": live_cost_metadata,
        },
    )
    report = {
        "schema_version": "1.0.0",
        "artifact_type": "replay_aligned_training_report",
        "generated_at": datetime.now(UTC).isoformat(),
        "authority": _training_authority(dataset),
        "model_path": str(model_path),
        "manifest_path": str(manifest_path),
        "config": asdict(config),
        "dataset": {
            "load_reports": dataset.attrs.get("load_reports", {}),
            "rows": int(len(dataset)),
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "symbols": int(dataset["symbol"].nunique()),
            "positive_rate": float(dataset["target"].mean()),
            "train_positive_rate": float(train["target"].mean()),
            "validation_positive_rate": float(validation["target"].mean()),
            "mean_round_trip_cost_bps": float(dataset["round_trip_cost_bps"].mean()),
            "mean_entry_slippage_bps": float(dataset["entry_slippage_bps"].mean()),
            "mean_exit_slippage_bps": float(dataset["exit_slippage_bps"].mean()),
            "mean_max_adverse_excursion_bps": float(dataset["max_adverse_excursion_bps"].mean()),
            "mean_max_favorable_excursion_bps": float(dataset["max_favorable_excursion_bps"].mean()),
            "mean_risk_adjusted_net_bps": float(dataset["risk_adjusted_net_bps"].mean()),
            "mean_label_score_bps": float(dataset["label_score_bps"].mean()),
            "split_purge": split_diagnostics,
        },
        "live_cost_model": live_cost_metadata,
        "feature_importance": feature_importance[:25],
        "validation": validation_report,
        "threshold_sweep": threshold_report,
        "threshold_sweep_by_regime": threshold_report_by_regime,
        "thresholds_by_regime": edge_thresholds_by_regime,
        "recommendation": "evaluate_candidate_with_offline_replay_before_promotion",
    }
    report_path = output_dir / f"{model_name}_training_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    logger.info(
        "REPLAY_ALIGNED_MODEL_TRAINED",
        extra={
            "model_path": str(model_path),
            "report_path": str(report_path),
            "rows": int(len(dataset)),
            "validation_roc_auc": validation_report.get("roc_auc"),
        },
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an offline replay-aligned edge model from local OHLCV bars."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--timestamp-col", type=str, default="timestamp")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument(
        "--model-type",
        choices=("logistic", "random_forest", "hist_gradient"),
        default="hist_gradient",
    )
    parser.add_argument("--horizon-bars", type=int, default=1)
    parser.add_argument(
        "--label-objective",
        choices=("net_markout", "spread_adjusted", "risk_adjusted", "mae_mfe"),
        default="net_markout",
        help=(
            "Training label objective. net_markout and spread_adjusted use cost-adjusted "
            "future markout; risk_adjusted and mae_mfe include adverse/favorable excursion."
        ),
    )
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument(
        "--live-cost-model-json",
        type=Path,
        default=None,
        help="Optional live cost model artifact for replay-aligned training labels.",
    )
    parser.add_argument(
        "--use-live-cost-model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use AI_TRADING_LIVE_COST_MODEL_PATH for training labels when no explicit artifact is provided.",
    )
    parser.add_argument("--min-net-edge-bps", type=float, default=0.0)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument(
        "--edge-global-threshold",
        type=float,
        default=None,
        help="Optional model-carried minimum live confidence threshold.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--training-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Cache replay-aligned feature frames across horizon/objective training runs.",
    )
    parser.add_argument("--training-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-research-synthetic-timestamps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow non-timestamped CSVs only for explicitly research-only synthetic training.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    train_replay_aligned_model(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
