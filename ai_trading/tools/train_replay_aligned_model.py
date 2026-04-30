"""Train an offline replay-aligned edge model from local OHLCV bars."""

from __future__ import annotations

import argparse
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

from ai_trading.data.historical_bars import load_historical_bars
from ai_trading.features.indicators import (
    compute_atr,
    compute_macd,
    compute_macds,
    compute_sma,
    compute_vwap,
)
from ai_trading.logging import get_logger
from ai_trading.models.artifacts import write_artifact_manifest
from ai_trading.tools.offline_replay import (
    _augment_model_features,
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


@dataclass(frozen=True)
class ReplayAlignedTrainingConfig:
    data_dir: str
    symbols: tuple[str, ...]
    horizon_bars: int
    fee_bps: float
    slippage_bps: float
    min_net_edge_bps: float
    train_fraction: float
    model_type: str
    edge_global_threshold: float | None


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


def _build_symbol_dataset(
    symbol: str,
    csv_path: Path,
    *,
    timestamp_col: str,
    horizon_bars: int,
    fee_bps: float,
    slippage_bps: float,
    min_net_edge_bps: float,
) -> pd.DataFrame:
    frame, _report = load_historical_bars(csv_path, timestamp_col=timestamp_col)
    if frame.empty:
        return pd.DataFrame()
    features = _feature_frame(frame, symbol=symbol)
    close = pd.to_numeric(frame["close"], errors="coerce").astype(float)
    future_close = close.shift(-int(horizon_bars))
    gross_long_bps = ((future_close / close.replace(0.0, np.nan)) - 1.0) * 10000.0
    round_trip_cost_bps = (2.0 * max(0.0, float(fee_bps))) + (
        2.0 * max(0.0, float(slippage_bps))
    )
    net_long_bps = gross_long_bps - round_trip_cost_bps
    out = features.copy()
    out["symbol"] = symbol
    out["timestamp"] = frame.index
    out["gross_long_bps"] = gross_long_bps.to_numpy(dtype=float)
    out["net_long_bps"] = net_long_bps.to_numpy(dtype=float)
    out["target"] = (out["net_long_bps"] > float(min_net_edge_bps)).astype(int)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[*REPLAY_ALIGNED_FEATURE_COLUMNS, "net_long_bps", "target"]
    )
    return cast(pd.DataFrame, out)


def build_training_dataset(
    *,
    data_dir: Path,
    symbols: str = "",
    timestamp_col: str = "timestamp",
    horizon_bars: int = 1,
    fee_bps: float = 1.0,
    slippage_bps: float = 2.0,
    min_net_edge_bps: float = 0.0,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for symbol, csv_path in _resolve_symbol_paths(data_dir, symbols).items():
        symbol_rows = _build_symbol_dataset(
            symbol,
            csv_path,
            timestamp_col=timestamp_col,
            horizon_bars=horizon_bars,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            min_net_edge_bps=min_net_edge_bps,
        )
        if not symbol_rows.empty:
            rows.append(symbol_rows)
    if not rows:
        return pd.DataFrame()
    dataset = pd.concat(rows, axis=0, ignore_index=True)
    dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], errors="coerce", utc=True)
    dataset = dataset.dropna(subset=["timestamp"]).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return cast(pd.DataFrame, dataset)


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


def _threshold_report(dataset: pd.DataFrame, probabilities: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    score = (2.0 * p) - 1.0
    confidence = np.maximum(p, 1.0 - p)
    net = pd.to_numeric(dataset["net_long_bps"], errors="coerce").to_numpy(dtype=float)
    for confidence_threshold in (0.52, 0.58, 0.62, 0.66):
        for entry_threshold in (0.05, 0.10, 0.15, 0.20):
            mask = (confidence >= confidence_threshold) & (np.abs(score) >= entry_threshold)
            if not bool(mask.any()):
                mean_net = None
                total_net = 0.0
                positive_rate = None
            else:
                signed_net = np.where(score[mask] >= 0.0, net[mask], -net[mask])
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
    rows.sort(key=lambda item: (item["mean_net_markout_bps"] is None, item["mean_net_markout_bps"] or -1e9), reverse=True)
    return rows


def _optional_threshold(value: Any) -> float | None:
    if value in (None, ""):
        return None
    threshold = float(value)
    if threshold <= 0.0:
        return None
    return float(np.clip(threshold, 0.0, 0.99))


def _attach_model_metadata(model: Any, *, edge_global_threshold: float | None) -> None:
    for name, value in (
        ("edge_score_orientation_", "direct"),
        ("replay_aligned_objective_", "one_bar_net_markout"),
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


def train_replay_aligned_model(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_training_dataset(
        data_dir=data_dir,
        symbols=str(args.symbols or ""),
        timestamp_col=str(args.timestamp_col),
        horizon_bars=int(args.horizon_bars),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slippage_bps),
        min_net_edge_bps=float(args.min_net_edge_bps),
    )
    if dataset.empty:
        raise RuntimeError("Replay-aligned training dataset is empty")
    if dataset["target"].nunique() < 2:
        raise RuntimeError("Replay-aligned target has fewer than two classes")

    cutoff_idx = max(1, min(len(dataset) - 1, int(len(dataset) * float(args.train_fraction))))
    train = dataset.iloc[:cutoff_idx].copy()
    validation = dataset.iloc[cutoff_idx:].copy()
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
    _attach_model_metadata(model, edge_global_threshold=edge_global_threshold)

    model_name = str(args.model_name or f"replay_aligned_{args.model_type}").strip()
    model_path = output_dir / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    config = ReplayAlignedTrainingConfig(
        data_dir=str(data_dir),
        symbols=tuple(sorted(dataset["symbol"].astype(str).str.upper().unique().tolist())),
        horizon_bars=int(args.horizon_bars),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slippage_bps),
        min_net_edge_bps=float(args.min_net_edge_bps),
        train_fraction=float(args.train_fraction),
        model_type=str(args.model_type),
        edge_global_threshold=edge_global_threshold,
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
            "objective": "one_bar_net_markout_binary",
            "config": asdict(config),
        },
    )
    validation_report = _evaluate_probabilities(y_validation, probabilities)
    threshold_report = _threshold_report(validation, probabilities)
    report = {
        "schema_version": "1.0.0",
        "artifact_type": "replay_aligned_training_report",
        "generated_at": datetime.now(UTC).isoformat(),
        "model_path": str(model_path),
        "manifest_path": str(manifest_path),
        "config": asdict(config),
        "dataset": {
            "rows": int(len(dataset)),
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "symbols": int(dataset["symbol"].nunique()),
            "positive_rate": float(dataset["target"].mean()),
            "train_positive_rate": float(train["target"].mean()),
            "validation_positive_rate": float(validation["target"].mean()),
        },
        "validation": validation_report,
        "threshold_sweep": threshold_report,
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
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--min-net-edge-bps", type=float, default=0.0)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument(
        "--edge-global-threshold",
        type=float,
        default=None,
        help="Optional model-carried minimum live confidence threshold.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    train_replay_aligned_model(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
