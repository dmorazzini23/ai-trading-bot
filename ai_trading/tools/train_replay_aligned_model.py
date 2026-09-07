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
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
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
from ai_trading.replay.live_cost_alignment import evaluate_cost_scenarios
from ai_trading.models.artifacts import write_artifact_manifest
from ai_trading.models.contracts import (
    DAY_SLEEVE_ML_BAR_TIMEFRAME,
    infer_day_sleeve_regimes,
)
from ai_trading.config.management import get_env
from ai_trading.paths import CACHE_DIR
from ai_trading.research.walk_forward import (
    ContiguousWalkForwardConfig,
    contiguous_walk_forward_splits,
)
from ai_trading.research.model_selection import (
    chronological_development_holdout,
    trailing_nested_selection_split,
)
from ai_trading.registry.manifest import (
    MARKET_REGIME_CLASSIFIER_ID,
    derive_market_regime_policy,
    evaluate_market_regime_policy,
)
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
_GOVERNED_HISTORICAL_SYMBOLS = frozenset({"AAPL", "AMZN", "MSFT"})
_HISTORICAL_AUTHORITY_REQUIRED: dict[str, Any] = {
    "research_only": True,
    "evidence_type": "historical_research",
    "promotion_eligible": False,
    "promotion_authority": False,
    "live_money_authority": False,
    "runtime_fill_authority": False,
}


class ContinuousEdgeEstimator:
    """Fit continuous post-cost edge while exposing replay-compatible scores."""

    def __init__(
        self,
        *,
        family: str,
        random_state: int,
        min_net_edge_bps: float = 0.0,
    ) -> None:
        self.family = str(family)
        self.random_state = int(random_state)
        self.min_net_edge_bps = float(min_net_edge_bps)
        self.continuous_edge_objective_ = True
        self.classes_ = np.asarray([0, 1], dtype=int)
        self._model: Any = None
        self._score_scale_bps = 1.0

    def _new_model(self) -> Any:
        if self.family == "edge_linear":
            from sklearn.linear_model import Ridge

            return Pipeline(
                steps=[
                    ("standardscaler", StandardScaler()),
                    ("ridge", Ridge(alpha=4.0)),
                ]
            )
        if self.family in {"edge_hist_gradient", "edge_rank"}:
            return HistGradientBoostingRegressor(
                loss="squared_error",
                max_iter=250,
                learning_rate=0.05,
                l2_regularization=0.10,
                min_samples_leaf=20,
                random_state=self.random_state,
            )
        raise ValueError(f"Unsupported continuous edge family: {self.family}")

    def fit(
        self,
        features: pd.DataFrame,
        target_edge_bps: pd.Series,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> ContinuousEdgeEstimator:
        target = pd.to_numeric(target_edge_bps, errors="coerce").to_numpy(dtype=float)
        if self.family == "edge_rank":
            target = pd.Series(target).rank(method="average", pct=True).to_numpy(dtype=float)
        finite = target[np.isfinite(target)]
        if finite.size == 0:
            raise ValueError("Continuous edge target has no finite values")
        self._score_scale_bps = max(
            float(np.quantile(np.abs(finite - np.median(finite)), 0.75)),
            1.0e-6,
        )
        self._model = self._new_model()
        weights = (
            np.asarray(sample_weight, dtype=float)
            if sample_weight is not None
            else np.ones(len(target), dtype=float)
        )
        if isinstance(self._model, Pipeline):
            final_step = self._model.steps[-1][0]
            self._model.fit(
                features,
                target,
                **{f"{final_step}__sample_weight": weights},
            )
        else:
            self._model.fit(features, target, sample_weight=weights)
        return self

    def predict_edge_bps(self, features: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Continuous edge estimator is not fitted")
        return np.asarray(self._model.predict(features), dtype=float)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        prediction = self.predict_edge_bps(features)
        if self.family == "edge_rank":
            positive = np.clip(prediction, 0.0, 1.0)
        else:
            logits = np.clip(
                (prediction - self.min_net_edge_bps) / self._score_scale_bps,
                -40.0,
                40.0,
            )
            positive = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack((1.0 - positive, positive))

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(features)[:, 1] >= 0.5).astype(int)


@dataclass(frozen=True)
class ReplayAlignedTrainingConfig:
    data_dir: str
    symbols: tuple[str, ...]
    horizon_bars: int
    label_objective: str
    fee_bps: float
    slippage_bps: float
    min_net_edge_bps: float
    max_training_invalid_rate: float
    train_fraction: float
    model_type: str
    edge_global_threshold: float | None
    live_cost_model_path: str | None = None
    live_cost_model_requested: bool = False
    live_cost_model_usable: bool = False
    training_cache_enabled: bool = True
    training_cache_dir: str | None = None
    walk_forward_folds: int = 5
    walk_forward_embargo_bars: int = 1
    walk_forward_embargo_percent: float = 0.0
    edge_weight_max: float = 5.0
    edge_weight_quantile: float = 0.90
    evaluation_folds: int | None = None
    nested_validation_fraction: float = 0.20
    nested_min_support: int = 25
    evaluate_holdout: bool = True


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


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_shadow_markout_overrides(
    *,
    jsonl_path: Path | None,
    manifest_path: Path | None,
    horizon_bars: int,
) -> tuple[dict[tuple[str, pd.Timestamp], dict[str, Any]], dict[str, Any]]:
    """Load strictly research-only markouts keyed to governed bar timestamps."""

    diagnostics: dict[str, Any] = {
        "requested": bool(jsonl_path is not None or manifest_path is not None),
        "usable": False,
        "training_ingestion_enabled": False,
        "matched_rows": 0,
        "reason": "not_requested",
        "evidence_partition": "shadow",
        "promotion_eligible": False,
        "runtime_authority": False,
        "promotion_authority": False,
        "live_money_authority": False,
    }
    if jsonl_path is None and manifest_path is None:
        return {}, diagnostics
    if jsonl_path is None or manifest_path is None:
        diagnostics["reason"] = "incomplete_request"
        return {}, diagnostics
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        diagnostics["reason"] = "manifest_unreadable"
        return {}, diagnostics
    required_false = (
        "fill_based_evidence",
        "promotion_eligible",
        "runtime_authority",
        "promotion_authority",
        "live_money_authority",
    )
    if not (
        isinstance(manifest, Mapping)
        and manifest.get("artifact_type")
        == "shadow_markout_replay_input_manifest"
        and manifest.get("schema_version") == "1.0.0"
        and manifest.get("evidence_type") == "shadow_counterfactual"
        and manifest.get("evidence_partition") == "shadow"
        and manifest.get("research_only") is True
        and all(manifest.get(field) is False for field in required_false)
    ):
        diagnostics["reason"] = "manifest_authority_contract_failed"
        return {}, diagnostics
    try:
        expected_output = _resolve_manifest_path(
            manifest.get("output_jsonl"), relative_to=manifest_path.parent
        ).resolve()
        resolved_jsonl = jsonl_path.expanduser().resolve()
        expected_hash = str(manifest.get("content_sha256") or "").strip().lower()
        if expected_output != resolved_jsonl:
            diagnostics["reason"] = "manifest_output_path_mismatch"
            return {}, diagnostics
        if not expected_hash or _file_sha256(resolved_jsonl) != expected_hash:
            diagnostics["reason"] = "content_hash_mismatch"
            return {}, diagnostics
    except (OSError, ValueError):
        diagnostics["reason"] = "evidence_unreadable"
        return {}, diagnostics

    overrides: dict[tuple[str, pd.Timestamp], dict[str, Any]] = {}
    outcome_ids: set[str] = set()
    parsed_rows = 0
    try:
        with resolved_jsonl.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                parsed_rows += 1
                if not isinstance(row, Mapping):
                    raise ValueError(f"invalid shadow row {line_number}")
                if not (
                    row.get("schema_version") == "1.0.0"
                    and row.get("evidence_type") == "shadow_counterfactual"
                    and row.get("evidence_partition") == "shadow"
                    and row.get("research_only") is True
                    and all(row.get(field) is False for field in required_false)
                ):
                    raise ValueError(
                        f"shadow authority contract failed at row {line_number}"
                    )
                if int(row.get("horizon_bars") or 0) != int(horizon_bars):
                    continue
                symbol = str(row.get("symbol") or "").strip().upper()
                if symbol not in _GOVERNED_HISTORICAL_SYMBOLS:
                    raise ValueError(f"ungoverned shadow symbol at row {line_number}")
                timestamp = pd.to_datetime(
                    row.get("decision_timestamp"), errors="coerce", utc=True
                )
                label_end = pd.to_datetime(
                    row.get("label_end_timestamp"), errors="coerce", utc=True
                )
                edge = float(row.get("net_markout_bps"))
                outcome_id = str(row.get("outcome_id") or "").strip()
                if pd.isna(timestamp) or pd.isna(label_end) or not np.isfinite(edge):
                    raise ValueError(f"invalid shadow label at row {line_number}")
                if not outcome_id or outcome_id in outcome_ids:
                    raise ValueError(f"duplicate shadow outcome at row {line_number}")
                outcome_ids.add(outcome_id)
                key = (symbol, cast(pd.Timestamp, timestamp))
                if key in overrides:
                    raise ValueError(f"duplicate shadow timestamp at row {line_number}")
                overrides[key] = {
                    "label_score_bps": edge,
                    "label_end_timestamp": label_end,
                    "outcome_id": outcome_id,
                }
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        diagnostics.update(
            {"reason": "row_contract_failed", "error_type": type(exc).__name__}
        )
        return {}, diagnostics
    if parsed_rows != int(manifest.get("row_count") or -1):
        diagnostics.update(
            {
                "reason": "row_count_mismatch",
                "manifest_rows": int(manifest.get("row_count") or 0),
                "parsed_rows": parsed_rows,
            }
        )
        return {}, diagnostics
    diagnostics.update(
        {
            "usable": True,
            "training_ingestion_enabled": bool(overrides),
            "reason": "validated_research_only",
            "manifest_path": str(manifest_path),
            "jsonl_path": str(resolved_jsonl),
            "content_sha256": expected_hash,
            "row_count": parsed_rows,
            "horizon_rows": len(overrides),
        }
    )
    return overrides, diagnostics


def _apply_shadow_markout_overrides(
    dataset: pd.DataFrame,
    *,
    jsonl_path: Path | None,
    manifest_path: Path | None,
    horizon_bars: int,
    label_objective: str,
    min_net_edge_bps: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    overrides, diagnostics = _load_shadow_markout_overrides(
        jsonl_path=jsonl_path,
        manifest_path=manifest_path,
        horizon_bars=horizon_bars,
    )
    if _normalize_label_objective(label_objective) != "net_markout":
        diagnostics.update(
            {
                "training_ingestion_enabled": False,
                "reason": (
                    "objective_not_compatible"
                    if diagnostics.get("usable")
                    else diagnostics.get("reason")
                ),
            }
        )
        return dataset, diagnostics
    if not overrides:
        return dataset, diagnostics
    result = dataset.copy()
    result["label_source"] = "historical_bar_cost_adjusted"
    result["shadow_outcome_id"] = None
    matched = 0
    for index, row in result.iterrows():
        timestamp = pd.to_datetime(row.get("timestamp"), errors="coerce", utc=True)
        key = (str(row.get("symbol") or "").strip().upper(), timestamp)
        override = overrides.get(key)
        if override is None:
            continue
        edge = float(override["label_score_bps"])
        result.at[index, "net_long_bps"] = edge
        result.at[index, "label_score_bps"] = edge
        result.at[index, "label_end_timestamp"] = override["label_end_timestamp"]
        result.at[index, "target"] = int(edge > float(min_net_edge_bps))
        result.at[index, "label_source"] = "shadow_counterfactual"
        result.at[index, "shadow_outcome_id"] = override["outcome_id"]
        matched += 1
    result.attrs.update(dataset.attrs)
    diagnostics.update(
        {
            "matched_rows": matched,
            "training_ingestion_enabled": matched > 0,
            "reason": "ingested" if matched > 0 else "no_timestamp_matches",
        }
    )
    result.attrs["shadow_markout_evidence"] = diagnostics
    return result, diagnostics


def _validated_historical_authority(
    payload: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    authority_raw = payload.get("authority")
    if not isinstance(authority_raw, Mapping):
        raise ValueError(f"{source} is missing historical research authority")
    authority = dict(authority_raw)
    mismatches = [
        key
        for key, expected in _HISTORICAL_AUTHORITY_REQUIRED.items()
        if authority.get(key) != expected
    ]
    if mismatches:
        raise ValueError(
            f"{source} has invalid historical research authority fields: "
            + ",".join(sorted(mismatches))
        )
    return authority


def _resolve_manifest_path(raw: Any, *, relative_to: Path) -> Path:
    if raw in (None, ""):
        raise ValueError("historical acquisition path is missing")
    path = Path(str(raw or "")).expanduser()
    return path if path.is_absolute() else (relative_to / path).resolve()


def _local_training_coverage(
    symbol_paths: dict[str, Path], *, timestamp_col: str,
) -> dict[str, Any]:
    """Describe actual input coverage before fitting; never infer quality from hashes."""
    coverage: dict[str, Any] = {}
    for symbol, path in sorted(symbol_paths.items()):
        frame, load_report = load_historical_bars(
            path, timestamp_col=timestamp_col, require_timestamp=True,
        )
        index = pd.DatetimeIndex(frame.index)
        local_dates = index.tz_convert("America/New_York").date
        deltas = index.to_series().diff().dt.total_seconds()
        same_date = pd.Series(local_dates, index=index).eq(
            pd.Series(local_dates, index=index).shift()
        )
        within_day = deltas[same_date & deltas.gt(0)]
        cadence = float(within_day.median()) if not within_day.empty else None
        coverage[symbol] = {
            "start": index.min().isoformat(),
            "end": index.max().isoformat(),
            "rows": len(frame),
            "observed_session_dates": sorted({str(day) for day in local_dates}),
            "observed_session_count": len(set(local_dates)),
            "median_intraday_cadence_seconds": cadence,
            "intraday_gap_count": int(within_day.gt(cadence * 1.5).sum()) if cadence else 0,
            "max_intraday_gap_seconds": float(within_day.max()) if not within_day.empty else None,
            "regime_distribution": dict(pd.Series(infer_day_sleeve_regimes(frame["close"])).value_counts().items()),
            "load_diagnostics": load_report.as_dict(),
        }
    return coverage


def _resolve_training_input(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    acquisition_raw = getattr(args, "acquisition_manifest_json", None)
    symbols_text = str(getattr(args, "symbols", "") or "")
    requested_symbols = {
        token.strip().upper()
        for token in symbols_text.split(",")
        if token.strip()
    }
    if acquisition_raw in (None, ""):
        data_dir_raw = getattr(args, "data_dir", None)
        if data_dir_raw in (None, ""):
            raise ValueError(
                "one of --data-dir or --acquisition-manifest-json is required"
            )
        data_dir = _resolve_input_dir(data_dir_raw)
        symbol_paths = _resolve_symbol_paths(data_dir, symbols_text)
        missing_symbols = requested_symbols - set(symbol_paths)
        if missing_symbols:
            raise ValueError(f"historical training symbols missing: {','.join(sorted(missing_symbols))}")
        coverage = _local_training_coverage(
            symbol_paths, timestamp_col=str(getattr(args, "timestamp_col", "timestamp")),
        )
        local_identities: list[dict[str, Any]] = [
            {
                "symbol": symbol,
                "content_sha256": _file_sha256(csv_path),
            }
            for symbol, csv_path in sorted(symbol_paths.items())
        ]
        return data_dir, {
            "mode": "local_historical_csv",
            "quality_passed": False,
            "quality_status": "unverified_completeness",
            "quality_reason": "acquisition_manifest_missing",
            "coverage": coverage,
            "dataset_hash": _canonical_sha256(local_identities),
            "symbols": sorted(symbol_paths),
            "authority": dict(_HISTORICAL_AUTHORITY_REQUIRED),
        }

    if requested_symbols - _GOVERNED_HISTORICAL_SYMBOLS:
        invalid = ",".join(sorted(requested_symbols - _GOVERNED_HISTORICAL_SYMBOLS))
        raise ValueError(f"replay historical training symbols are not governed: {invalid}")

    acquisition_path = _resolve_input_dir(acquisition_raw)
    try:
        acquisition_payload = json.loads(acquisition_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"unable to read historical acquisition manifest: {acquisition_path}"
        ) from exc
    if not isinstance(acquisition_payload, Mapping):
        raise ValueError("historical acquisition manifest must contain a JSON object")
    if acquisition_payload.get("quality_passed") is not True:
        raise ValueError("historical acquisition failed its completeness quality gate")
    authority = _validated_historical_authority(
        acquisition_payload,
        source="historical acquisition",
    )
    data_dir = _resolve_manifest_path(
        acquisition_payload.get("dataset_dir"),
        relative_to=acquisition_path.parent,
    )
    manifest_path = _resolve_manifest_path(
        acquisition_payload.get("manifest_path"),
        relative_to=acquisition_path.parent,
    )
    if not data_dir.is_dir():
        raise ValueError(f"historical acquisition dataset directory is missing: {data_dir}")
    if manifest_path.resolve().parent != data_dir.resolve():
        raise ValueError("historical dataset provenance is outside the dataset root")
    try:
        dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"unable to read historical dataset provenance: {manifest_path}"
        ) from exc
    if not isinstance(dataset_manifest, Mapping):
        raise ValueError("historical dataset provenance must contain a JSON object")
    if dataset_manifest.get("quality_passed") is not True:
        raise ValueError("historical dataset provenance failed its quality gate")
    _validated_historical_authority(
        dataset_manifest,
        source="historical dataset provenance",
    )
    cache_key = str(acquisition_payload.get("cache_key") or "").strip()
    manifest_cache_key = str(dataset_manifest.get("dataset_cache_key") or "").strip()
    if not cache_key or cache_key != manifest_cache_key:
        raise ValueError("historical acquisition cache key does not match provenance")

    symbol_rows_raw = acquisition_payload.get("symbols")
    if not isinstance(symbol_rows_raw, list) or not symbol_rows_raw:
        raise ValueError("historical acquisition contains no symbol datasets")
    manifest_symbol_rows_raw = dataset_manifest.get("symbols")
    if not isinstance(manifest_symbol_rows_raw, list):
        raise ValueError("historical dataset provenance contains no symbol datasets")
    manifest_symbol_rows = {
        str(row.get("symbol") or "").strip().upper(): row
        for row in manifest_symbol_rows_raw
        if isinstance(row, Mapping)
    }
    acquisition_identities: list[dict[str, Any]] = []
    completeness: dict[str, Any] = {}
    acquired_symbols: set[str] = set()
    resolved_root = data_dir.resolve()
    for raw_row in symbol_rows_raw:
        if not isinstance(raw_row, Mapping):
            raise ValueError("historical acquisition symbol entry is invalid")
        symbol = str(raw_row.get("symbol") or "").strip().upper()
        if not symbol or symbol not in _GOVERNED_HISTORICAL_SYMBOLS:
            raise ValueError(f"historical acquisition contains ungoverned symbol: {symbol}")
        if symbol in acquired_symbols:
            raise ValueError(f"historical acquisition contains duplicate symbol: {symbol}")
        if raw_row.get("quality_passed") is not True:
            raise ValueError(f"historical acquisition quality failed for {symbol}")
        csv_path = _resolve_manifest_path(
            raw_row.get("csv_path"),
            relative_to=data_dir,
        ).resolve()
        if csv_path.parent != resolved_root or not csv_path.is_file():
            raise ValueError(f"historical acquisition CSV is outside dataset root: {symbol}")
        expected_hash = str(raw_row.get("content_sha256") or "").strip().lower()
        actual_hash = _file_sha256(csv_path)
        if not expected_hash or actual_hash != expected_hash:
            raise ValueError(f"historical acquisition content hash mismatch for {symbol}")
        manifest_row = manifest_symbol_rows.get(symbol)
        if (
            not isinstance(manifest_row, Mapping)
            or manifest_row.get("quality_passed") is not True
            or str(manifest_row.get("content_sha256") or "").strip().lower()
            != actual_hash
        ):
            raise ValueError(
                f"historical acquisition does not match provenance for {symbol}"
            )
        acquired_symbols.add(symbol)
        completeness[symbol] = dict(raw_row.get("completeness") or {})
        acquisition_identities.append(
            {
                "symbol": symbol,
                "content_sha256": actual_hash,
                "row_count": int(raw_row.get("row_count") or 0),
            }
        )
    if requested_symbols and not requested_symbols.issubset(acquired_symbols):
        missing = ",".join(sorted(requested_symbols - acquired_symbols))
        raise ValueError(f"requested historical symbols are missing: {missing}")
    if acquired_symbols != set(manifest_symbol_rows):
        raise ValueError("historical acquisition symbol set does not match provenance")

    return data_dir, {
        "mode": "governed_historical_backfill",
        "output_json": str(acquisition_path),
        "output_sha256": _file_sha256(acquisition_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _file_sha256(manifest_path),
        "dataset_cache_key": cache_key,
        "dataset_hash": _canonical_sha256(
            sorted(acquisition_identities, key=lambda row: row["symbol"])
        ),
        "quality_passed": True,
        "symbols": sorted(acquired_symbols),
        "completeness": completeness,
        "coverage": _local_training_coverage(
            _resolve_symbol_paths(data_dir, symbols_text),
            timestamp_col=str(getattr(args, "timestamp_col", "timestamp")),
        ),
        "dataset_identity": dict(dataset_manifest.get("dataset_identity") or {}),
        "authority": authority,
    }


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
        "execution": "execution_adjusted",
        "execution_aware": "execution_adjusted",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {
        "net_markout",
        "spread_adjusted",
        "risk_adjusted",
        "mae_mfe",
        "execution_adjusted",
    }
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
    execution_adjusted_long_bps: pd.Series | None = None,
) -> pd.Series:
    normalized = _normalize_label_objective(objective)
    if normalized == "net_markout":
        return net_long_bps
    if normalized == "spread_adjusted":
        return spread_adjusted_long_bps
    if normalized == "execution_adjusted":
        return (
            execution_adjusted_long_bps
            if execution_adjusted_long_bps is not None
            else net_long_bps
        )
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
    next_low = (
        pd.to_numeric(frame["low"], errors="coerce").shift(-1)
        if "low" in frame.columns
        else future_close
    )
    passive_limit = close * (1.0 - (spread_cost_bps / 20000.0))
    passive_fill_probability_proxy = (next_low <= passive_limit).astype(float)
    passive_fill_probability_proxy.loc[next_low.isna()] = np.nan
    opportunity_cost_bps = net_long_bps.clip(lower=0.0) * (
        1.0 - passive_fill_probability_proxy
    )
    execution_adjusted_net_bps = (
        net_long_bps * passive_fill_probability_proxy
    ) - opportunity_cost_bps
    normalized_objective = _normalize_label_objective(label_objective)
    label_score_bps = _label_score(
        objective=normalized_objective,
        net_long_bps=net_long_bps,
        spread_adjusted_long_bps=spread_adjusted_long_bps,
        max_adverse_excursion_bps=max_adverse_excursion_bps,
        max_favorable_excursion_bps=max_favorable_excursion_bps,
        round_trip_cost_bps=round_trip_cost_bps,
        execution_adjusted_long_bps=execution_adjusted_net_bps,
    )
    out = features.copy()
    out["close"] = close.to_numpy(dtype=float)
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
    out["net_edge_after_cost_bps"] = net_long_bps.to_numpy(dtype=float)
    out["spread_adjusted_markout_bps"] = spread_adjusted_long_bps.to_numpy(dtype=float)
    out["max_adverse_excursion_bps"] = max_adverse_excursion_bps.to_numpy(dtype=float)
    out["max_favorable_excursion_bps"] = max_favorable_excursion_bps.to_numpy(dtype=float)
    out["mae_bps"] = max_adverse_excursion_bps.to_numpy(dtype=float)
    out["mfe_bps"] = max_favorable_excursion_bps.to_numpy(dtype=float)
    out["passive_fill_probability_proxy"] = passive_fill_probability_proxy.to_numpy(dtype=float)
    out["opportunity_cost_bps"] = opportunity_cost_bps.to_numpy(dtype=float)
    out["execution_adjusted_net_bps"] = execution_adjusted_net_bps.to_numpy(dtype=float)
    out["live_cost_adjusted_net_edge_bps"] = (
        net_long_bps.to_numpy(dtype=float)
        if live_cost_model is not None
        else np.full(len(out), np.nan, dtype=float)
    )
    out["risk_adjusted_net_bps"] = risk_adjusted_net_bps.to_numpy(dtype=float)
    out["label_score_bps"] = label_score_bps.to_numpy(dtype=float)
    out["label_objective"] = normalized_objective
    out["target"] = (out["label_score_bps"] > float(min_net_edge_bps)).astype(int)
    required_columns = [
            *REPLAY_ALIGNED_FEATURE_COLUMNS,
            "timestamp",
            "label_end_timestamp",
            "net_long_bps",
            "label_score_bps",
            "target",
    ]
    out = out.replace([np.inf, -np.inf], np.nan)
    expected_rows = max(0, len(out) - int(horizon_bars))
    eligible = out.iloc[:expected_rows] if expected_rows else out.iloc[0:0]
    invalid_mask = eligible[required_columns].isna().any(axis=1)
    invalid_rows = int(invalid_mask.sum())
    quality_report = {
        "symbol": symbol,
        "raw_rows": int(len(out)),
        "expected_labeled_rows": int(expected_rows),
        "valid_labeled_rows": int(expected_rows - invalid_rows),
        "quarantined_rows": invalid_rows,
        "unexpected_invalid_rate": (
            float(invalid_rows / expected_rows) if expected_rows else 0.0
        ),
        "missing_rate_by_required_column": {
            column: float(eligible[column].isna().mean()) if expected_rows else 0.0
            for column in required_columns
        },
        "zero_rate_by_feature": {
            column: float((pd.to_numeric(eligible[column], errors="coerce") == 0.0).mean())
            if expected_rows
            else 0.0
            for column in REPLAY_ALIGNED_FEATURE_COLUMNS
        },
        "quarantine_examples": [
            str(value)
            for value in eligible.loc[invalid_mask, "timestamp"].head(10).tolist()
        ],
    }
    out = out.dropna(subset=required_columns)
    out.attrs["quality_report"] = quality_report
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
    quality_reports: dict[str, dict[str, Any]] = {}
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
            quality = symbol_rows.attrs.get("quality_report")
            if isinstance(quality, Mapping):
                quality_reports[symbol] = dict(quality)
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
    total_expected = sum(
        int(report.get("expected_labeled_rows", 0) or 0)
        for report in quality_reports.values()
    )
    total_quarantined = sum(
        int(report.get("quarantined_rows", 0) or 0)
        for report in quality_reports.values()
    )
    dataset.attrs["quality_report"] = {
        "status": "complete",
        "symbols": quality_reports,
        "expected_labeled_rows": total_expected,
        "valid_labeled_rows": total_expected - total_quarantined,
        "quarantined_rows": total_quarantined,
        "unexpected_invalid_rate": (
            float(total_quarantined / total_expected) if total_expected else 0.0
        ),
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
        "evidence_type": "historical_research",
        "promotion_eligible": False,
        "runtime_fill_authority": False,
        "timestamp_authoritative": bool(timestamp_authoritative and bool(reports)),
        "research_synthetic": bool(research_synthetic),
        "source_providers": source_providers,
    }


def _make_model(
    model_type: str,
    *,
    random_state: int,
    min_net_edge_bps: float = 0.0,
) -> Any:
    normalized = str(model_type or "logistic").strip().lower()
    if normalized in {"logistic", "meta_label"}:
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
    if normalized in {"edge_linear", "edge_hist_gradient", "edge_rank"}:
        return ContinuousEdgeEstimator(
            family=normalized,
            random_state=random_state,
            min_net_edge_bps=min_net_edge_bps,
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def _make_candidate_model(
    model_type: str,
    *,
    random_state: int,
    min_net_edge_bps: float,
) -> Any:
    if str(model_type).strip().lower() in {
        "edge_linear",
        "edge_hist_gradient",
        "edge_rank",
    }:
        return _make_model(
            model_type,
            random_state=random_state,
            min_net_edge_bps=min_net_edge_bps,
        )
    return _make_model(model_type, random_state=random_state)


def _edge_magnitude_sample_weights(
    dataset: pd.DataFrame,
    *,
    min_net_edge_bps: float,
    max_weight: float,
    scaling_quantile: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return bounded weights derived only from the supplied fit partition."""

    bounded_max = max(1.0, float(max_weight))
    bounded_quantile = float(np.clip(float(scaling_quantile), 0.50, 1.0))
    edge = pd.to_numeric(dataset["label_score_bps"], errors="coerce").to_numpy(
        dtype=float
    )
    distance = np.abs(edge - float(min_net_edge_bps))
    finite = np.isfinite(distance)
    finite_distance = distance[finite]
    scale_bps = (
        float(np.quantile(finite_distance, bounded_quantile))
        if finite_distance.size
        else 0.0
    )
    if not np.isfinite(scale_bps) or scale_bps <= 0.0:
        weights = np.ones(len(dataset), dtype=float)
    else:
        normalized = np.clip(
            np.nan_to_num(distance / scale_bps, nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        )
        weights = 1.0 + ((bounded_max - 1.0) * normalized)
    weights = np.clip(weights, 1.0, bounded_max).astype(float)
    return weights, {
        "objective": "bounded_post_cost_edge_weighted_binary",
        "source": "abs_label_score_bps_minus_min_net_edge_bps",
        "partition_local": True,
        "rows": int(len(dataset)),
        "min_net_edge_bps": float(min_net_edge_bps),
        "scaling_quantile": bounded_quantile,
        "scale_bps": scale_bps,
        "min_weight": float(np.min(weights)) if weights.size else 1.0,
        "mean_weight": float(np.mean(weights)) if weights.size else 1.0,
        "max_weight": float(np.max(weights)) if weights.size else 1.0,
        "configured_max_weight": bounded_max,
    }


def _fit_weighted_binary_model(
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    sample_weight: np.ndarray,
) -> None:
    """Fit supported binary estimators without changing their serving contract."""

    if isinstance(model, Pipeline):
        final_step = model.steps[-1][0] if model.steps else ""
        if not final_step:
            raise ValueError("Replay-aligned pipeline has no final estimator")
        model.fit(
            features,
            target,
            **{f"{final_step}__sample_weight": np.asarray(sample_weight, dtype=float)},
        )
        return
    model.fit(features, target, sample_weight=np.asarray(sample_weight, dtype=float))


def _fit_replay_model(
    model: Any,
    features: pd.DataFrame,
    dataset: pd.DataFrame,
    *,
    sample_weight: np.ndarray,
) -> None:
    if isinstance(model, ContinuousEdgeEstimator):
        model.fit(
            features,
            dataset["net_long_bps"].astype(float),
            sample_weight=sample_weight,
        )
        return
    _fit_weighted_binary_model(
        model,
        features,
        dataset["target"].astype(int),
        sample_weight=sample_weight,
    )


def _feature_importance(model: Any) -> list[dict[str, Any]]:
    """Return lightweight feature attribution for candidate triage artifacts."""
    estimator = model._model if isinstance(model, ContinuousEdgeEstimator) else model
    if isinstance(estimator, Pipeline):
        estimator = estimator.steps[-1][1] if estimator.steps else estimator
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


def _heldout_feature_autopsy(
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    random_state: int,
) -> dict[str, Any]:
    """Measure feature usefulness only on untouched chronological holdout rows."""

    if features.empty or target.nunique() < 2:
        return {
            "status": "insufficient_support",
            "rows": int(len(features)),
            "features": [],
        }
    def _roc_auc_scorer(estimator: Any, values: pd.DataFrame, labels: pd.Series) -> float:
        probabilities = np.asarray(estimator.predict_proba(values), dtype=float)
        return float(roc_auc_score(labels, probabilities[:, _positive_class_index(estimator)]))

    measured = permutation_importance(
        model,
        features,
        target.astype(int),
        scoring=_roc_auc_scorer,
        n_repeats=5,
        random_state=int(random_state),
        n_jobs=1,
    )
    rows = [
        {
            "feature": feature,
            "importance_mean": float(mean),
            "importance_std": float(std),
            "helpful_on_holdout": bool(mean > 0.0),
        }
        for feature, mean, std in zip(
            REPLAY_ALIGNED_FEATURE_COLUMNS,
            measured.importances_mean,
            measured.importances_std,
            strict=True,
        )
    ]
    rows.sort(
        key=lambda item: abs(cast(float, item["importance_mean"])),
        reverse=True,
    )
    return {
        "status": "complete",
        "method": "chronological_holdout_permutation_importance",
        "scoring": "roc_auc",
        "repeats": 5,
        "rows": int(len(features)),
        "features": rows,
        "nonpositive_feature_count": sum(
            not bool(row["helpful_on_holdout"]) for row in rows
        ),
        "selection_authority": False,
        "promotion_authority": False,
    }


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
    """Evaluate development-only percentile selections.

    Class probabilities are ranking scores for this one-sided, imbalanced
    target.  They are not interpreted as signed edge via ``2p - 1``.
    """

    rows: list[dict[str, Any]] = []
    p = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    net = pd.to_numeric(dataset["net_long_bps"], errors="coerce").to_numpy(dtype=float)
    for percentile_threshold in (0.70, 0.80, 0.90, 0.95):
        probability_cutoff = float(np.quantile(p, percentile_threshold))
        if allow_short_labels:
            lower_cutoff = float(np.quantile(p, 1.0 - percentile_threshold))
            long_mask = p >= probability_cutoff
            short_mask = p <= lower_cutoff
            mask = long_mask | short_mask
            signed_net = np.where(long_mask[mask], net[mask], -net[mask])
        else:
            mask = p >= probability_cutoff
            signed_net = net[mask]
        selected = signed_net[np.isfinite(signed_net)]
        rows.append(
            {
                # Retained as the frozen probability cutoff consumed by replay.
                "confidence_threshold": probability_cutoff,
                "entry_score_threshold": 0.0,
                "score_percentile_threshold": float(percentile_threshold),
                "probability_cutoff": probability_cutoff,
                "score_semantics": "positive_class_probability_rank",
                "selection_scope": "development_only",
                "candidates": int(selected.size),
                "mean_net_markout_bps": (
                    float(np.mean(selected)) if selected.size else None
                ),
                "total_net_markout_bps": float(np.sum(selected)),
                "positive_rate": (
                    float(np.mean(selected > 0.0)) if selected.size else None
                ),
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


def _select_nested_threshold(
    dataset: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    min_support: int,
    fixed_confidence_threshold: float | None,
) -> dict[str, Any] | None:
    """Choose a feasible post-cost threshold from inner validation only."""

    reports = _threshold_report(dataset, probabilities)
    requested_percentile = (
        float(np.clip(fixed_confidence_threshold, 0.0, 0.99))
        if fixed_confidence_threshold is not None
        else None
    )
    feasible = [
        row
        for row in reports
        if int(row.get("candidates", 0) or 0) >= max(1, int(min_support))
        and row.get("mean_net_markout_bps") is not None
        and float(row.get("mean_net_markout_bps") or 0.0) > 0.0
        and (
            requested_percentile is None
            or abs(
                float(row.get("score_percentile_threshold", 0.0))
                - requested_percentile
            )
            < 1e-9
        )
    ]
    if not feasible and requested_percentile is not None:
        probability = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
        cutoff = float(np.quantile(probability, requested_percentile))
        net = pd.to_numeric(dataset["net_long_bps"], errors="coerce").to_numpy(dtype=float)
        selected = net[(probability >= cutoff) & np.isfinite(net)]
        selected_mean = float(np.mean(selected)) if selected.size else None
        row = {
            "confidence_threshold": cutoff,
            "entry_score_threshold": 0.0,
            "score_percentile_threshold": requested_percentile,
            "probability_cutoff": cutoff,
            "score_semantics": "positive_class_probability_rank",
            "selection_scope": "development_only",
            "candidates": int(selected.size),
            "mean_net_markout_bps": selected_mean,
            "total_net_markout_bps": float(np.sum(selected)),
            "positive_rate": float(np.mean(selected > 0.0)) if selected.size else None,
        }
        if (
            selected.size >= max(1, int(min_support))
            and float(selected_mean or 0.0) > 0.0
        ):
            feasible.append(row)
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda row: (
            float(row.get("mean_net_markout_bps") or -1e12),
            float(row.get("positive_rate") or 0.0),
            int(row.get("candidates", 0) or 0),
        ),
    )


def _abstention_diagnostics(
    dataset: pd.DataFrame, probabilities: np.ndarray, *, min_support: int,
    selected_threshold: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Explain the existing threshold decision without changing its selection policy."""
    scores = np.asarray(probabilities, dtype=float)
    finite = np.isfinite(scores)
    if not finite.all():
        return {"status": "invalid_scores", "rows": len(dataset), "invalid_scores": int((~finite).sum())}
    rows = _threshold_report(dataset, scores)
    gross = pd.to_numeric(dataset["gross_long_bps"], errors="coerce").to_numpy(dtype=float) if "gross_long_bps" in dataset else None
    costs = pd.to_numeric(dataset["round_trip_cost_bps"], errors="coerce").to_numpy(dtype=float) if "round_trip_cost_bps" in dataset else None
    reason_counts: dict[str, int] = {}
    for row in rows:
        mask = scores >= float(row["probability_cutoff"])
        selected_gross = gross[mask] if gross is not None else np.asarray([], dtype=float)
        selected_costs = costs[mask] if costs is not None else np.asarray([], dtype=float)
        gross_mean = float(np.mean(selected_gross)) if len(selected_gross) and np.isfinite(selected_gross).all() else None
        cost_mean = float(np.mean(selected_costs)) if len(selected_costs) and np.isfinite(selected_costs).all() else None
        net_mean = row["mean_net_markout_bps"]
        reasons = []
        if int(row["candidates"]) < min_support:
            reasons.append("insufficient_support")
        if net_mean is None or not np.isfinite(float(net_mean)):
            reasons.append("net_edge_unavailable")
        elif float(net_mean) <= 0:
            reasons.append("costs_exceed_positive_gross_edge" if gross_mean is not None and gross_mean > 0 else "gross_edge_nonpositive" if gross_mean is not None else "net_edge_nonpositive")
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        row.update({"mean_gross_markout_bps": gross_mean, "mean_round_trip_cost_bps": cost_mean, "rejection_reasons": reasons})
    return {
        "status": "threshold_selected" if selected_threshold else "abstained",
        "rows": len(dataset), "minimum_support": min_support,
        "score_quantiles": {str(q): float(np.quantile(scores, q)) for q in (0, 0.5, 0.9, 0.95, 1)} if len(scores) else {},
        "tested_percentiles": rows, "rejected_threshold_counts": reason_counts,
        "rejection_counts_overlap": True,
        "selected_threshold": dict(selected_threshold or {}),
        "scope": "inner_validation_only", "policy_changed": False,
    }


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
    horizon_bars: int = 1,
    label_objective: str = "net_markout",
) -> None:
    continuous = isinstance(model, ContinuousEdgeEstimator)
    for name, value in (
        ("edge_score_orientation_", "direct"),
        (
            "edge_score_semantics_",
            "continuous_edge_rank_score"
            if continuous
            else "positive_class_probability_rank",
        ),
        ("edge_threshold_selection_scope_", "development_only_percentile"),
        (
            "replay_aligned_objective_",
            f"{max(1, int(horizon_bars))}_bar_{_normalize_label_objective(label_objective)}",
        ),
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


def _ranking_separation(
    dataset: pd.DataFrame,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    scores = np.asarray(probabilities, dtype=float)
    net = pd.to_numeric(dataset["net_long_bps"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(scores) & np.isfinite(net)
    if not bool(valid.any()):
        return {
            "rows": 0,
            "quantile_fraction": 0.20,
            "high_score_mean_net_edge_bps": None,
            "low_score_mean_net_edge_bps": None,
            "high_minus_low_net_edge_bps": None,
        }
    valid_scores = scores[valid]
    valid_net = net[valid]
    bucket_size = max(1, int(np.ceil(valid_scores.size * 0.20)))
    order = np.argsort(valid_scores, kind="stable")
    low_mean = float(np.mean(valid_net[order[:bucket_size]]))
    high_mean = float(np.mean(valid_net[order[-bucket_size:]]))
    return {
        "rows": int(valid_scores.size),
        "quantile_fraction": 0.20,
        "high_score_mean_net_edge_bps": high_mean,
        "low_score_mean_net_edge_bps": low_mean,
        "high_minus_low_net_edge_bps": float(high_mean - low_mean),
    }


def _selected_post_cost_metrics(
    dataset: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    confidence_threshold: float,
    entry_score_threshold: float,
) -> tuple[dict[str, Any], np.ndarray]:
    probability = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    selected = probability >= float(confidence_threshold)
    net = pd.to_numeric(dataset["net_long_bps"], errors="coerce").to_numpy(dtype=float)
    selected_net = net[selected & np.isfinite(net)]
    gains = float(selected_net[selected_net > 0.0].sum()) if selected_net.size else 0.0
    losses = float(-selected_net[selected_net < 0.0].sum()) if selected_net.size else 0.0
    cumulative = np.cumsum(selected_net) if selected_net.size else np.asarray([], dtype=float)
    running_peak = np.maximum.accumulate(np.concatenate([np.asarray([0.0]), cumulative]))
    equity = np.concatenate([np.asarray([0.0]), cumulative])
    max_drawdown = float(np.max(running_peak - equity)) if equity.size else 0.0
    metrics = {
        "selected_candidates": int(selected_net.size),
        "trades": int(selected_net.size),
        "mean_post_cost_net_edge_bps": (
            float(np.mean(selected_net)) if selected_net.size else None
        ),
        "total_post_cost_net_edge_bps": float(np.sum(selected_net)),
        "gross_positive_net_edge_bps": gains,
        "gross_negative_net_edge_bps": losses,
        "profit_factor": (float(gains / losses) if losses > 0.0 else None),
        "max_drawdown_bps": max_drawdown,
        "hit_rate": (
            float(np.mean(selected_net > 0.0)) if selected_net.size else None
        ),
        "confidence_threshold": float(confidence_threshold),
        "entry_score_threshold": float(entry_score_threshold),
        "score_semantics": "positive_class_probability_rank",
        "threshold_selection_scope": "development_only_percentile",
        "ranking_separation": _ranking_separation(dataset, probability),
    }
    return metrics, selected


def _regime_post_cost_metrics(
    dataset: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    confidence_threshold: float,
    entry_score_threshold: float,
) -> tuple[str | None, dict[str, dict[str, Any]]]:
    regime_col = next(
        (name for name in ("market_regime", "session_regime") if name in dataset.columns),
        None,
    )
    if regime_col is None:
        return None, {}
    regimes = dataset[regime_col].astype(str).str.strip().str.lower()
    out: dict[str, dict[str, Any]] = {}
    for regime in sorted(value for value in regimes.unique().tolist() if value):
        mask = (regimes == regime).to_numpy()
        metrics, _ = _selected_post_cost_metrics(
            dataset.loc[mask].copy(),
            np.asarray(probabilities, dtype=float)[mask],
            confidence_threshold=confidence_threshold,
            entry_score_threshold=entry_score_threshold,
        )
        metrics["oos_rows"] = int(mask.sum())
        out[regime] = metrics
    return regime_col, out


def _fold_market_regimes(
    history: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.Series, dict[str, Any]]:
    """Label OOS rows with the canonical past-only live-serving classifier."""

    if "close" not in history.columns or "close" not in test.columns:
        raise ValueError("Canonical market-regime inference requires close prices")
    combined = history.copy()
    combined["_walk_forward_test_row"] = combined.index.isin(test.index)
    combined = combined.sort_values(
        ["symbol", "timestamp"], kind="mergesort"
    )
    result = pd.Series("sideways", index=test.index, dtype="object")
    for _, symbol_rows in combined.groupby("symbol", sort=True):
        symbol_rows = symbol_rows.sort_values("timestamp", kind="mergesort")
        close = pd.to_numeric(symbol_rows["close"], errors="coerce").ffill().bfill()
        labels = np.asarray(infer_day_sleeve_regimes(close.to_numpy(dtype=float)), dtype=object)
        test_mask = symbol_rows["_walk_forward_test_row"].astype(bool).to_numpy()
        result.loc[symbol_rows.index[test_mask]] = labels[test_mask]
    counts = result.astype(str).value_counts().sort_index()
    return (
        result,
        {
            "market_regime_classifier": "day_sleeve_past_only_v1",
            "canonical_helper": "ai_trading.models.contracts.infer_day_sleeve_regimes",
            "method": "canonical_past_only_close_inference",
            "raw_close_return_proxy_used": False,
            "bar_timeframe": DAY_SLEEVE_ML_BAR_TIMEFRAME,
            "past_only": True,
            "history_scope": "full_causal_symbol_history_through_fold_test_end",
            "test_label_counts": {
                str(regime): int(count) for regime, count in counts.items()
            },
        },
    )


def _walk_forward_qualification(
    folds: list[dict[str, Any]],
    *,
    required_folds: int,
    min_trades: int,
    min_mean_net_edge_bps: float,
    min_profitable_fold_ratio: float,
    min_ranking_separation_bps: float,
) -> dict[str, Any]:
    supported = [fold for fold in folds if int(fold.get("trades", 0) or 0) > 0]
    fold_edges = [float(fold["mean_post_cost_net_edge_bps"]) for fold in supported]
    total_trades = sum(int(fold.get("trades", 0) or 0) for fold in folds)
    total_edge = sum(float(fold.get("total_post_cost_net_edge_bps", 0.0) or 0.0) for fold in folds)
    mean_edge = float(total_edge / total_trades) if total_trades else None
    profitable_folds = sum(edge > 0.0 for edge in fold_edges)
    profitable_ratio = float(profitable_folds / len(folds)) if folds else 0.0
    separation_values = [
        float(value)
        for fold in folds
        for value in [
            cast(Mapping[str, Any], fold.get("ranking_separation", {})).get(
                "high_minus_low_net_edge_bps"
            )
        ]
        if value is not None
    ]
    mean_separation = (
        float(np.mean(separation_values)) if separation_values else None
    )
    edge_confidence_lower_bound = None
    if fold_edges:
        edge_confidence_lower_bound = float(np.mean(fold_edges))
        if len(fold_edges) > 1:
            edge_confidence_lower_bound -= float(
                1.96 * np.std(fold_edges, ddof=1) / np.sqrt(len(fold_edges))
            )
    reasons: list[str] = []
    if len(folds) < int(required_folds):
        reasons.append(f"insufficient_folds:{len(folds)}<{int(required_folds)}")
    if total_trades < int(min_trades):
        reasons.append(f"insufficient_support:{total_trades}<{int(min_trades)}")
    if mean_edge is None or mean_edge <= float(min_mean_net_edge_bps):
        reasons.append(
            f"nonpositive_or_below_minimum_net_edge:{mean_edge}"
        )
    if (
        edge_confidence_lower_bound is None
        or edge_confidence_lower_bound <= float(min_mean_net_edge_bps)
    ):
        reasons.append(
            "net_edge_confidence_lower_bound_below_minimum:"
            f"{edge_confidence_lower_bound}"
        )
    if profitable_ratio < float(min_profitable_fold_ratio):
        reasons.append(
            "unstable_profitable_folds:"
            f"{profitable_ratio:.6f}<{float(min_profitable_fold_ratio):.6f}"
        )
    if mean_separation is None or mean_separation <= float(min_ranking_separation_bps):
        reasons.append(f"insufficient_ranking_separation:{mean_separation}")
    edge_std = float(np.std(fold_edges)) if len(fold_edges) > 1 else 0.0
    stability_score = (
        float(max(0.0, 1.0 - (edge_std / max(abs(float(mean_edge or 0.0)), 1.0))))
        if fold_edges
        else 0.0
    )
    worst_fold = min(
        supported,
        key=lambda fold: float(fold.get("mean_post_cost_net_edge_bps", 0.0) or 0.0),
        default=None,
    )
    return {
        "evidence_qualified": not reasons,
        "qualification_reasons": reasons,
        "fold_count": int(len(folds)),
        "supported_fold_count": int(len(supported)),
        "profitable_fold_count": int(profitable_folds),
        "profitable_fold_ratio": profitable_ratio,
        "selected_candidates": int(total_trades),
        "trades": int(total_trades),
        "mean_post_cost_net_edge_bps": mean_edge,
        "total_post_cost_net_edge_bps": float(total_edge),
        "fold_edge_std_bps": edge_std,
        "fold_edge_confidence_lower_bound_bps": edge_confidence_lower_bound,
        "confidence_level": 0.95,
        "stability_score": stability_score,
        "mean_ranking_high_minus_low_bps": mean_separation,
        "worst_fold": (
            {
                "fold_index": int(worst_fold.get("fold_index", 0) or 0),
                "mean_post_cost_net_edge_bps": worst_fold.get(
                    "mean_post_cost_net_edge_bps"
                ),
            }
            if worst_fold is not None
            else None
        ),
        "governance_status": "shadow",
        "promotion_authority": False,
        "live_money_authority": False,
    }


def _aggregate_market_regime_results(
    oos_frame: pd.DataFrame,
    *,
    min_trades: int,
    min_profitable_fold_ratio: float,
    min_mean_net_edge_bps: float,
    min_ranking_separation_bps: float,
    group_column: str = "market_regime",
) -> dict[str, dict[str, Any]]:
    regime_names = sorted(oos_frame[group_column].astype(str).unique().tolist())
    out: dict[str, dict[str, Any]] = {}
    regime_min_trades = max(25, int(np.ceil(int(min_trades) / max(1, len(regime_names)))))
    for regime in regime_names:
        regime_frame = oos_frame.loc[
            oos_frame[group_column].astype(str) == regime
        ].copy()
        selected_frame = regime_frame.loc[
            regime_frame["walk_forward_selected"].astype(bool)
        ].copy()
        selected_net = pd.to_numeric(
            selected_frame["net_long_bps"], errors="coerce"
        ).dropna().to_numpy(dtype=float)
        trades = int(selected_net.size)
        total_edge = float(np.sum(selected_net))
        gross_positive = float(selected_net[selected_net > 0.0].sum())
        gross_negative = float(-selected_net[selected_net < 0.0].sum())
        fold_edges = [
            float(pd.to_numeric(group["net_long_bps"], errors="coerce").mean())
            for _, group in selected_frame.groupby("walk_forward_fold")
            if not group.empty
        ]
        fold_count = int(regime_frame["walk_forward_fold"].nunique())
        supported_fold_count = int(selected_frame["walk_forward_fold"].nunique())
        profitable = sum(edge > 0.0 for edge in fold_edges)
        profitable_ratio = float(profitable / fold_count) if fold_count else 0.0
        mean_edge = float(total_edge / trades) if trades else 0.0
        cumulative = np.cumsum(selected_net) if trades else np.asarray([], dtype=float)
        equity = np.concatenate([np.asarray([0.0]), cumulative])
        running_peak = np.maximum.accumulate(equity)
        max_drawdown = float(np.max(running_peak - equity)) if equity.size else 0.0
        ranking = _ranking_separation(
            regime_frame,
            pd.to_numeric(
                regime_frame["walk_forward_probability"], errors="coerce"
            ).to_numpy(dtype=float),
        )
        ranking_separation = ranking.get("high_minus_low_net_edge_bps")
        reasons: list[str] = []
        if trades < regime_min_trades:
            reasons.append(f"insufficient_regime_support:{trades}<{regime_min_trades}")
        if mean_edge <= float(min_mean_net_edge_bps):
            reasons.append(f"nonpositive_regime_net_edge:{mean_edge}")
        if profitable_ratio < float(min_profitable_fold_ratio):
            reasons.append(
                "unstable_regime_folds:"
                f"{profitable_ratio:.6f}<{float(min_profitable_fold_ratio):.6f}"
            )
        if (
            ranking_separation is None
            or float(ranking_separation) <= float(min_ranking_separation_bps)
        ):
            reasons.append(f"insufficient_regime_ranking_separation:{ranking_separation}")
        out[regime] = {
            "oos_rows": int(len(regime_frame)),
            "support": int(trades),
            "fold_count": fold_count,
            "supported_fold_count": supported_fold_count,
            "profitable_fold_count": int(profitable),
            "profitable_fold_ratio": profitable_ratio,
            "trades": int(trades),
            "mean_post_cost_net_edge_bps": mean_edge,
            "total_post_cost_net_edge_bps": float(total_edge),
            "profit_factor": (
                float(gross_positive / gross_negative) if gross_negative > 0.0 else None
            ),
            "max_drawdown_bps": max_drawdown,
            "hit_rate": float(np.mean(selected_net > 0.0)) if trades else None,
            "ranking_separation": ranking,
            "evidence_qualified": not reasons,
            "qualification_reasons": reasons,
            "shadow_disposition": "observe" if not reasons else "abstain",
            "promotion_authority": False,
            "live_money_authority": False,
        }
    return out


def _research_cost_scenarios(args: argparse.Namespace) -> tuple[list[float], dict[str, Any]]:
    costs = [float(value) for value in str(getattr(args, "cost_scenarios_bps", "0,3,6,10,20")).split(",")]
    path = getattr(args, "live_cost_model_json", None)
    evidence: dict[str, Any] = {"available": False, "reason": "quote_prior_missing", "runtime_fill_authority": False}
    if path is None:
        return costs, evidence
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    prior = payload.get("research_cost_prior", {})
    if prior.get("evidence_type") != "quote_derived_research_prior":
        return costs, evidence
    if any(prior.get(key) for key in ("runtime_fill_authority", "promotion_eligible", "promotion_authority", "live_money_authority")):
        return costs, {**evidence, "reason": "quote_prior_authority_invalid"}
    now = datetime.now(UTC)
    values = []
    symbols = {value.strip().upper() for value in str(getattr(args, "symbols", "")).split(",") if value.strip()}
    for row in prior.get("by_symbol_side_session", []):
        if symbols and row.get("symbol") not in symbols:
            continue
        timestamp = pd.to_datetime(row.get("last_observed_at"), utc=True, errors="coerce")
        if pd.isna(timestamp) or not -300 <= (now - timestamp).total_seconds() <= 96 * 3600:
            continue
        value = row.get("p90_prior_cost_bps")
        if row.get("sufficient_samples") and value is not None and np.isfinite(float(value)) and float(value) >= 0:
            values.append(float(value))
    if not values:
        return costs, {**evidence, "reason": "fresh_sufficient_quote_buckets_missing"}
    estimate = float(np.median(values))
    return [*costs, estimate], {
        "available": True, "source_path": str(path), "source_sha256": _file_sha256(Path(path)),
        "evidence_type": "quote_derived_research_prior", "aggregation": "median_of_recent_bucket_p90_costs",
        "cost_bps": estimate, "bucket_count": len(values), "max_age_hours": 96,
        "runtime_fill_authority": False, "promotion_authority": False,
        "limitation": "constant research stress scenario, not historical fill calibration or candidate-weighted costs",
    }


def _controlled_fold_comparisons(
    *, args: argparse.Namespace, train: pd.DataFrame, test: pd.DataFrame,
    nested_fit: pd.DataFrame, nested_selection: pd.DataFrame,
    candidate_selected: np.ndarray, market_regimes: np.ndarray,
    fold_index: int,
) -> list[dict[str, Any]]:
    """Predeclared controls use identical outer tests and inner-only threshold selection."""
    disabled = set(getattr(args, "disabled_experiments", ()))
    min_edge = float(getattr(args, "min_net_edge_bps", 0.0))
    masks = {
        "candidate": candidate_selected,
        "always_long": np.ones(len(test), dtype=bool),
        "cash": np.zeros(len(test), dtype=bool),
        "momentum": test["sma_spread"].to_numpy(dtype=float) > 0.0,
        "abstain_volatile": candidate_selected & (market_regimes != "volatile"),
    }
    removals = [value.strip() for value in str(getattr(args, "experiment_feature_removals", "macd_signal_gap")).split(",") if value.strip()]
    invalid = set(removals) - set(REPLAY_ALIGNED_FEATURE_COLUMNS)
    if invalid:
        raise ValueError(f"Unknown experiment features: {sorted(invalid)}")
    for feature in removals:
        name = f"remove_{feature}"
        if name in disabled:
            continue
        columns = [column for column in REPLAY_ALIGNED_FEATURE_COLUMNS if column != feature]
        model = _make_candidate_model(
            str(args.model_type), random_state=int(args.random_state) + fold_index,
            min_net_edge_bps=min_edge,
        )
        weights, _ = _edge_magnitude_sample_weights(
            nested_fit, min_net_edge_bps=min_edge,
            max_weight=float(getattr(args, "edge_weight_max", 5.0)),
            scaling_quantile=float(getattr(args, "edge_weight_quantile", 0.9)),
        )
        _fit_replay_model(model, nested_fit[columns].astype(float), nested_fit, sample_weight=weights)
        probabilities = np.asarray(model.predict_proba(nested_selection[columns].astype(float)))[:, _positive_class_index(model)]
        threshold = _select_nested_threshold(
            nested_selection, probabilities,
            min_support=min(int(getattr(args, "nested_min_support", 25)), max(1, len(nested_selection) // 4)),
            fixed_confidence_threshold=_optional_threshold(getattr(args, "edge_global_threshold", None)),
        )
        weights, _ = _edge_magnitude_sample_weights(
            train, min_net_edge_bps=min_edge,
            max_weight=float(getattr(args, "edge_weight_max", 5.0)),
            scaling_quantile=float(getattr(args, "edge_weight_quantile", 0.9)),
        )
        model = _make_candidate_model(
            str(args.model_type), random_state=int(args.random_state) + fold_index,
            min_net_edge_bps=min_edge,
        )
        _fit_replay_model(model, train[columns].astype(float), train, sample_weight=weights)
        probabilities = np.asarray(model.predict_proba(test[columns].astype(float)))[:, _positive_class_index(model)]
        masks[name] = probabilities >= float(threshold["confidence_threshold"]) if threshold else np.zeros(len(test), dtype=bool)
    net = test["net_long_bps"].to_numpy(dtype=float)
    return [
        {
            "name": name, "fold_index": fold_index,
            "opportunities": len(test), "trades": int(mask.sum()),
            "total_net_edge_bps": float(net[mask].sum()),
            "mean_net_edge_bps": float(net[mask].mean()) if mask.any() else None,
            "net_edge_per_opportunity_bps": float(net[mask].sum() / len(test)),
        }
        for name, mask in masks.items() if name not in disabled or name == "candidate"
    ]


def _summarize_controlled_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate = {row["fold_index"]: row for row in rows if row["name"] == "candidate"}
    results = []
    for name in sorted({row["name"] for row in rows}):
        folds = [row for row in rows if row["name"] == name]
        edges = np.asarray([row["net_edge_per_opportunity_bps"] for row in folds])
        deltas = np.asarray([row["net_edge_per_opportunity_bps"] - candidate[row["fold_index"]]["net_edge_per_opportunity_bps"] for row in folds])
        opportunities = sum(row["opportunities"] for row in folds)
        total = sum(row["total_net_edge_bps"] for row in folds)
        profitable = float(np.mean(edges > 0.0))
        improvement = float(np.mean(deltas))
        accepted = bool(len(folds) >= 2 and total > 0.0 and improvement > 0.0 and profitable >= 0.60 and float(np.mean(deltas > 0.0)) >= 0.60)
        results.append({
            "name": name, "hypothesis": f"{name} improves net edge per common opportunity over the candidate without reducing fold consistency",
            "acceptance_criterion": {"minimum_folds": 2, "net_edge_bps_gt": 0.0, "paired_improvement_bps_gt": 0.0, "profitable_fold_ratio_gte": 0.60, "improving_fold_ratio_gte": 0.60},
            "folds": folds, "opportunities": opportunities,
            "trades": sum(row["trades"] for row in folds),
            "net_edge_per_opportunity_bps": total / opportunities if opportunities else None,
            "mean_paired_improvement_bps": improvement,
            "profitable_fold_ratio": profitable, "fold_edge_std_bps": float(np.std(edges)),
            "accepted": accepted, "conclusive": len(folds) >= 2,
        })
    results.sort(key=lambda row: (row["net_edge_per_opportunity_bps"], -row["fold_edge_std_bps"]), reverse=True)
    return {"scope": "development_outer_folds_only", "selection_scope": "nested_inner_validation_only", "metric": "equal_notional_markout_per_common_opportunity_not_portfolio_return", "variants": results, "promotion_authority": False}


def _run_fold_local_walk_forward(
    dataset: pd.DataFrame,
    *,
    args: argparse.Namespace,
    edge_global_threshold: float | None,
    cost_model_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], Any, pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    requested_folds = max(2, int(getattr(args, "walk_forward_folds", 5) or 5))
    embargo_bars = max(1, int(getattr(args, "walk_forward_embargo_bars", 1) or 1))
    embargo_percent = max(
        0.0, float(getattr(args, "walk_forward_embargo_percent", 0.0) or 0.0)
    )
    split_config = ContiguousWalkForwardConfig(
        folds=requested_folds,
        horizon_bars=int(args.horizon_bars),
        embargo_bars=embargo_bars,
        embargo_percent=embargo_percent,
    )
    all_splits = contiguous_walk_forward_splits(dataset, split_config)
    evaluation_folds_raw = getattr(args, "evaluation_folds", None)
    evaluation_folds = (
        requested_folds
        if evaluation_folds_raw in (None, 0, "")
        else max(1, min(requested_folds, int(evaluation_folds_raw)))
    )
    splits = all_splits[:evaluation_folds]
    fold_reports: list[dict[str, Any]] = []
    oos_frames: list[pd.DataFrame] = []
    last_bundle: tuple[Any, pd.DataFrame, pd.DataFrame, np.ndarray] | None = None
    edge_weight_max = max(
        1.0,
        float(getattr(args, "edge_weight_max", 5.0) or 5.0),
    )
    edge_weight_quantile = float(
        np.clip(
            float(getattr(args, "edge_weight_quantile", 0.90) or 0.90),
            0.50,
            1.0,
        )
    )
    min_net_edge_bps = float(
        getattr(args, "min_net_edge_bps", 0.0) or 0.0
    )
    nested_validation_fraction = float(
        getattr(args, "nested_validation_fraction", 0.20) or 0.20
    )
    nested_min_support = max(
        1, int(getattr(args, "nested_min_support", 25) or 25)
    )
    selected_thresholds: list[tuple[float, float]] = []
    comparison_rows: list[dict[str, Any]] = []
    for fold, train, test in splits:
        if train["target"].nunique() < 2:
            continue
        try:
            nested = trailing_nested_selection_split(
                train,
                validation_fraction=nested_validation_fraction,
                embargo_bars=embargo_bars,
            )
        except ValueError:
            continue
        if nested.fit["target"].nunique() < 2:
            continue
        selector_model = _make_candidate_model(
            str(args.model_type),
            random_state=int(args.random_state) + int(fold.fold_index),
            min_net_edge_bps=min_net_edge_bps,
        )
        nested_fit_features = nested.fit[
            list(REPLAY_ALIGNED_FEATURE_COLUMNS)
        ].astype(float)
        nested_selection_features = nested.selection[
            list(REPLAY_ALIGNED_FEATURE_COLUMNS)
        ].astype(float)
        selector_weights, selector_weight_report = _edge_magnitude_sample_weights(
            nested.fit,
            min_net_edge_bps=min_net_edge_bps,
            max_weight=edge_weight_max,
            scaling_quantile=edge_weight_quantile,
        )
        _fit_replay_model(
            selector_model,
            nested_fit_features,
            nested.fit,
            sample_weight=selector_weights,
        )
        if isinstance(selector_model, ContinuousEdgeEstimator):
            selector_weight_report = selector_weight_report | {
                "objective": "bounded_continuous_post_cost_edge_weighting"
            }
        selector_positive_index = _positive_class_index(selector_model)
        selection_probabilities = np.asarray(
            selector_model.predict_proba(nested_selection_features), dtype=float
        )[:, selector_positive_index]
        best_nested_threshold = _select_nested_threshold(
            nested.selection,
            selection_probabilities,
            min_support=min(nested_min_support, max(1, len(nested.selection) // 4)),
            fixed_confidence_threshold=edge_global_threshold,
        )
        abstention = _abstention_diagnostics(
            nested.selection, selection_probabilities,
            min_support=min(nested_min_support, max(1, len(nested.selection) // 4)),
            selected_threshold=best_nested_threshold,
        )
        threshold_feasible = best_nested_threshold is not None
        confidence_threshold = float(
            best_nested_threshold.get("confidence_threshold", 1.0)
            if best_nested_threshold is not None
            else 1.0
        )
        entry_score_threshold = float(
            best_nested_threshold.get("entry_score_threshold", 1.0)
            if best_nested_threshold is not None
            else 1.0
        )
        selected_thresholds.append((confidence_threshold, entry_score_threshold))

        model = _make_candidate_model(
            str(args.model_type),
            random_state=int(args.random_state) + int(fold.fold_index),
            min_net_edge_bps=min_net_edge_bps,
        )
        train_features = train[list(REPLAY_ALIGNED_FEATURE_COLUMNS)].astype(float)
        test_features = test[list(REPLAY_ALIGNED_FEATURE_COLUMNS)].astype(float)
        train_weights, weight_report = _edge_magnitude_sample_weights(
            train,
            min_net_edge_bps=min_net_edge_bps,
            max_weight=edge_weight_max,
            scaling_quantile=edge_weight_quantile,
        )
        _fit_replay_model(
            model,
            train_features,
            train,
            sample_weight=train_weights,
        )
        if isinstance(model, ContinuousEdgeEstimator):
            weight_report = weight_report | {
                "objective": "bounded_continuous_post_cost_edge_weighting"
            }
        positive_index = _positive_class_index(model)
        test_probabilities = np.asarray(
            model.predict_proba(test_features), dtype=float
        )[:, positive_index]
        metrics, selected = _selected_post_cost_metrics(
            test,
            test_probabilities,
            confidence_threshold=confidence_threshold,
            entry_score_threshold=entry_score_threshold,
        )
        regime_test = test.copy()
        regime_history = dataset.loc[
            pd.to_datetime(dataset["timestamp"], errors="coerce", utc=True)
            <= fold.test_end
        ].copy()
        market_regimes, regime_definition = _fold_market_regimes(regime_history, test)
        regime_test["market_regime"] = market_regimes
        if bool(getattr(args, "research_experiments", True)):
            comparison_rows.extend(_controlled_fold_comparisons(
                args=args, train=train, test=test, nested_fit=nested.fit,
                nested_selection=nested.selection, candidate_selected=selected,
                market_regimes=market_regimes, fold_index=int(fold.fold_index),
            ))
        regime_source, by_regime = _regime_post_cost_metrics(
            regime_test,
            test_probabilities,
            confidence_threshold=confidence_threshold,
            entry_score_threshold=entry_score_threshold,
        )
        fold_report = {
            "fold_index": int(fold.fold_index),
            "train_start": str(fold.train_start),
            "train_end": str(fold.train_end),
            "test_start": str(fold.test_start),
            "test_end": str(fold.test_end),
            "initial_train_rows": int(fold.initial_train_rows),
            "train_rows": int(fold.train_rows),
            "test_rows": int(fold.test_rows),
            "purged_train_rows": int(fold.purged_train_rows),
            "embargoed_train_rows": int(fold.embargoed_train_rows),
            "horizon_bars": int(fold.horizon_bars),
            "embargo_bars": int(fold.embargo_bars),
            "embargo_percent": float(fold.embargo_percent),
            "chronological_non_overlap": bool(fold.chronological_non_overlap),
            "label_purge_ok": bool(fold.label_purge_ok),
            "fit_scope": "fold_train_only",
            "threshold_scope": "nested_inner_validation_only",
            "threshold_feasible": bool(threshold_feasible),
            "abstention_diagnostics": abstention,
            "threshold_selection": {
                "scope": "nested_inner_validation_only",
                "fit_rows": int(len(nested.fit)),
                "selection_rows": int(len(nested.selection)),
                "selection_start": str(nested.selection_start),
                "purged_fit_rows": int(nested.purged_fit_rows),
                "embargoed_fit_rows": int(nested.embargoed_fit_rows),
                "embargo_bars": int(nested.embargo_bars),
                "min_support": int(
                    min(nested_min_support, max(1, len(nested.selection) // 4))
                ),
                "selected": dict(best_nested_threshold or {}),
                "infeasible_action": (
                    None if threshold_feasible else "abstain_all_outer_test_rows"
                ),
                "sample_weight": selector_weight_report,
            },
            "fit_objective": (
                "continuous_cost_adjusted_realized_edge_bps"
                if isinstance(model, ContinuousEdgeEstimator)
                else "bounded_post_cost_edge_weighted_binary"
            ),
            "sample_weight": weight_report,
            "cost_model": dict(cost_model_identity),
            "validation": _evaluate_probabilities(test["target"], test_probabilities),
            "regime_source": regime_source,
            "regime_definition": regime_definition,
            "by_market_regime": by_regime,
            "opportunity_funnel": {
                "outer_test_rows": len(test),
                "finite_scores": int(np.isfinite(test_probabilities).sum()),
                "nested_threshold_feasible": threshold_feasible,
                "selected_by_frozen_threshold": int(selected.sum()),
                "reason": "no_supported_profitable_inner_threshold" if not threshold_feasible else "frozen_threshold_applied",
                "scope": "model_selection_before_replay_execution_gates",
            },
            **metrics,
        }
        fold_reports.append(fold_report)
        oos = test.copy()
        oos["market_regime"] = market_regimes
        oos["walk_forward_probability"] = test_probabilities
        oos["walk_forward_selected"] = selected
        oos["walk_forward_fold"] = int(fold.fold_index)
        oos_frames.append(oos)
        last_bundle = (model, train, test, test_probabilities)
    if last_bundle is None or not oos_frames:
        raise RuntimeError("No usable fold-local walk-forward evaluations")

    min_trades = max(1, int(getattr(args, "walk_forward_min_trades", 250) or 250))
    min_profitable_ratio = float(
        getattr(args, "walk_forward_min_profitable_fold_ratio", 0.60) or 0.60
    )
    min_mean_edge = float(
        getattr(args, "walk_forward_min_mean_net_edge_bps", 0.0) or 0.0
    )
    min_separation = float(
        getattr(args, "walk_forward_min_ranking_separation_bps", 0.0) or 0.0
    )
    aggregate = _walk_forward_qualification(
        fold_reports,
        required_folds=evaluation_folds,
        min_trades=min_trades,
        min_mean_net_edge_bps=min_mean_edge,
        min_profitable_fold_ratio=min_profitable_ratio,
        min_ranking_separation_bps=min_separation,
    )
    oos_frame = pd.concat(oos_frames, axis=0, ignore_index=True).sort_values(
        ["timestamp", "symbol"], kind="mergesort"
    )
    oos_probabilities = pd.to_numeric(
        oos_frame["walk_forward_probability"], errors="coerce"
    ).to_numpy(dtype=float)
    aggregate_by_market_regime = _aggregate_market_regime_results(
        oos_frame,
        min_trades=min_trades,
        min_profitable_fold_ratio=min_profitable_ratio,
        min_mean_net_edge_bps=min_mean_edge,
        min_ranking_separation_bps=min_separation,
    )
    aggregate_by_symbol = _aggregate_market_regime_results(
        oos_frame,
        min_trades=min_trades,
        min_profitable_fold_ratio=min_profitable_ratio,
        min_mean_net_edge_bps=min_mean_edge,
        min_ranking_separation_bps=min_separation,
        group_column="symbol",
    )
    aggregate_by_session_regime = _aggregate_market_regime_results(
        oos_frame,
        min_trades=min_trades,
        min_profitable_fold_ratio=min_profitable_ratio,
        min_mean_net_edge_bps=min_mean_edge,
        min_ranking_separation_bps=min_separation,
        group_column="session_regime",
    )
    scoped_oos = oos_frame.copy()
    scoped_oos["symbol_market_regime"] = (
        scoped_oos["symbol"].astype(str).str.upper()
        + "::"
        + scoped_oos["market_regime"].astype(str).str.lower()
    )
    aggregate_by_symbol_market_regime = _aggregate_market_regime_results(
        scoped_oos,
        min_trades=min_trades,
        min_profitable_fold_ratio=min_profitable_ratio,
        min_mean_net_edge_bps=min_mean_edge,
        min_ranking_separation_bps=min_separation,
        group_column="symbol_market_regime",
    )
    scoped_oos["symbol_session_regime"] = (
        scoped_oos["symbol"].astype(str).str.upper()
        + "::"
        + scoped_oos["session_regime"].astype(str).str.lower()
    )
    aggregate_by_symbol_session_regime = _aggregate_market_regime_results(
        scoped_oos,
        min_trades=min_trades,
        min_profitable_fold_ratio=min_profitable_ratio,
        min_mean_net_edge_bps=min_mean_edge,
        min_ranking_separation_bps=min_separation,
        group_column="symbol_session_regime",
    )
    symbol_regime_policy = {
        scope: {
            "action": (
                "observe" if bool(metrics.get("evidence_qualified")) else "abstain"
            ),
            "evidence_qualified": bool(metrics.get("evidence_qualified")),
            "qualification_reasons": list(metrics.get("qualification_reasons") or []),
            "support": int(metrics.get("support", 0) or 0),
            "mean_post_cost_net_edge_bps": metrics.get(
                "mean_post_cost_net_edge_bps"
            ),
            "runtime_authority": False,
            "live_money_authority": False,
        }
        for scope, metrics in sorted(aggregate_by_symbol_market_regime.items())
    }
    walk_forward_report = {
        "evaluation_type": "expanding_contiguous_walk_forward",
        "market_regime_classifier": MARKET_REGIME_CLASSIFIER_ID,
        "fold_local_fitting": True,
        "time_ordered": True,
        "test_blocks_non_overlapping": True,
        "label_horizon_purged": True,
        "config": {
            "requested_folds": requested_folds,
            "evaluation_folds": evaluation_folds,
            "fixed_fold_schedule_count": len(all_splits),
            "horizon_bars": int(args.horizon_bars),
            "embargo_bars": embargo_bars,
            "embargo_percent": embargo_percent,
            "min_trades": min_trades,
            "min_profitable_fold_ratio": min_profitable_ratio,
            "min_mean_net_edge_bps": min_mean_edge,
            "min_ranking_separation_bps": min_separation,
        },
        "folds": fold_reports,
        "aggregate": aggregate,
        "by_market_regime": aggregate_by_market_regime,
        "by_symbol": aggregate_by_symbol,
        "by_session_regime": aggregate_by_session_regime,
        "by_symbol_market_regime": aggregate_by_symbol_market_regime,
        "by_symbol_session_regime": aggregate_by_symbol_session_regime,
        "symbol_regime_policy": {
            "default_action": "abstain",
            "scopes": symbol_regime_policy,
            "governance_status": "shadow",
            "promotion_authority": False,
            "runtime_authority": False,
            "live_money_authority": False,
        },
        "governance_status": "shadow",
        "promotion_authority": False,
        "live_money_authority": False,
        "offline_replay_required": True,
    }
    _, _, final_test, final_probabilities = last_bundle
    model = _make_candidate_model(
        str(args.model_type),
        random_state=int(args.random_state),
        min_net_edge_bps=min_net_edge_bps,
    )
    final_train = dataset.copy()
    final_train_features = final_train[
        list(REPLAY_ALIGNED_FEATURE_COLUMNS)
    ].astype(float)
    final_weights, final_weight_report = _edge_magnitude_sample_weights(
        final_train,
        min_net_edge_bps=min_net_edge_bps,
        max_weight=edge_weight_max,
        scaling_quantile=edge_weight_quantile,
    )
    _fit_replay_model(
        model,
        final_train_features,
        final_train,
        sample_weight=final_weights,
    )
    if isinstance(model, ContinuousEdgeEstimator):
        final_weight_report = final_weight_report | {
            "objective": "bounded_continuous_post_cost_edge_weighting"
        }
    final_confidence_threshold = float(
        np.median([value[0] for value in selected_thresholds])
        if selected_thresholds
        else (edge_global_threshold or 1.0)
    )
    final_entry_score_threshold = float(
        np.median([value[1] for value in selected_thresholds])
        if selected_thresholds
        else 1.0
    )
    walk_forward_report["selected_threshold"] = {
        "confidence_threshold": final_confidence_threshold,
        "entry_score_threshold": final_entry_score_threshold,
        "scope": "aggregate_of_nested_inner_validation_thresholds",
        "fold_threshold_count": len(selected_thresholds),
    }
    walk_forward_report["final_fit"] = {
        "scope": "development_partition_only_after_oos_evaluation",
        "rows": int(len(final_train)),
        "threshold_scope": "nested_inner_validation_only",
        "fit_objective": (
            "continuous_cost_adjusted_realized_edge_bps"
            if isinstance(model, ContinuousEdgeEstimator)
            else "bounded_post_cost_edge_weighted_binary"
        ),
        "sample_weight": final_weight_report,
        "promotion_authority": False,
        "live_money_authority": False,
    }
    walk_forward_report["controlled_comparisons"] = _summarize_controlled_comparisons(comparison_rows)
    return (
        walk_forward_report,
        model,
        final_train,
        final_test,
        final_probabilities,
        oos_frame,
        oos_probabilities,
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
    data_dir, acquisition = _resolve_training_input(args)
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = str(args.model_name or f"replay_aligned_{args.model_type}").strip()
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
    dataset, shadow_markout_evidence = _apply_shadow_markout_overrides(
        dataset,
        jsonl_path=(
            Path(getattr(args, "shadow_markout_jsonl"))
            if getattr(args, "shadow_markout_jsonl", None) is not None
            else None
        ),
        manifest_path=(
            Path(getattr(args, "shadow_markout_manifest_json"))
            if getattr(args, "shadow_markout_manifest_json", None) is not None
            else None
        ),
        horizon_bars=int(args.horizon_bars),
        label_objective=str(getattr(args, "label_objective", "net_markout")),
        min_net_edge_bps=float(args.min_net_edge_bps),
    )
    dataset_quality = dict(dataset.attrs.get("quality_report") or {})
    invalid_rate = float(dataset_quality.get("unexpected_invalid_rate", 0.0) or 0.0)
    max_invalid_rate = float(
        np.clip(
            float(getattr(args, "max_training_invalid_rate", 0.02)),
            0.0,
            1.0,
        )
    )
    dataset_quality.update(
        {
            "maximum_allowed_unexpected_invalid_rate": max_invalid_rate,
            "quality_gate_passed": invalid_rate <= max_invalid_rate,
        }
    )
    dataset_quality_path = output_dir / f"{model_name}_dataset_quality.json"
    dataset_quality_path.write_text(
        json.dumps(dataset_quality, indent=2, sort_keys=True), encoding="utf-8"
    )
    if invalid_rate > max_invalid_rate:
        raise RuntimeError(
            "Replay-aligned training data quality gate failed: "
            f"unexpected_invalid_rate={invalid_rate:.6f} "
            f"> maximum={max_invalid_rate:.6f}; "
            f"see {dataset_quality_path}"
        )
    if dataset.empty:
        raise RuntimeError("Replay-aligned training dataset is empty")
    if dataset["target"].nunique() < 2:
        raise RuntimeError("Replay-aligned target has fewer than two classes")

    edge_global_threshold = _optional_threshold(getattr(args, "edge_global_threshold", None))
    holdout_partition = chronological_development_holdout(
        dataset,
        train_fraction=float(getattr(args, "train_fraction", 0.70) or 0.70),
        embargo_bars=max(
            1, int(getattr(args, "walk_forward_embargo_bars", 1) or 1)
        ),
    )
    development = holdout_partition.development
    holdout = holdout_partition.holdout
    if development["target"].nunique() < 2:
        raise RuntimeError("Replay-aligned development target has fewer than two classes")
    live_cost_metadata = _live_cost_request_metadata(args, live_cost_model)
    cost_model_identity = {
        "version": (
            f"live_cost_model:{live_cost_metadata.get('source_sha256')}"
            if live_cost_metadata.get("source_sha256")
            else "static_fee_spread_slippage_v1"
        ),
        "fee_bps": float(args.fee_bps),
        "slippage_bps": float(args.slippage_bps),
        "live_cost_model_source_sha256": live_cost_metadata.get("source_sha256"),
    }
    (
        walk_forward_report,
        model,
        train,
        validation,
        _final_probabilities,
        oos_frame,
        oos_probabilities,
    ) = _run_fold_local_walk_forward(
        development,
        args=args,
        edge_global_threshold=edge_global_threshold,
        cost_model_identity=cost_model_identity,
    )
    selected_oos = oos_frame.loc[oos_frame["walk_forward_selected"].astype(bool)]
    scenario_costs, quote_cost_evidence = _research_cost_scenarios(args)
    cost_sensitivity = evaluate_cost_scenarios(
        gross_returns_bps=selected_oos["gross_long_bps"].to_numpy(dtype=float),
        turnover=np.full(len(selected_oos), 2.0), scenario_costs_bps=scenario_costs,
        evidence_type="historical_research_markout",
    )
    cost_sensitivity.update({
        "quote_research_evidence": quote_cost_evidence,
        "scope": "frozen_outer_fold_selections", "cost_unit": "bps_per_one_way_execution",
        "return_unit": "equal_notional_round_trip_markout_bps_not_portfolio_return",
        "assumptions": "two executions per selected trade; replaces all modeled round-trip costs; no queue or impact validation",
        "configured_mean_round_trip_cost_bps": float(selected_oos["round_trip_cost_bps"].mean()) if len(selected_oos) else None,
    })
    selected_threshold = cast(
        Mapping[str, Any], walk_forward_report.get("selected_threshold", {})
    )
    selected_confidence_threshold = float(
        selected_threshold.get(
            "confidence_threshold",
            edge_global_threshold if edge_global_threshold is not None else 1.0,
        )
    )
    selected_entry_score_threshold = float(
        selected_threshold.get("entry_score_threshold", 1.0)
    )
    observed_session_regimes = sorted(
        {
            str(value).strip().lower()
            for value in oos_frame.get("session_regime", pd.Series(dtype=str)).tolist()
            if str(value).strip()
        }
    )
    session_regime_results = cast(
        Mapping[str, Mapping[str, Any]],
        walk_forward_report.get("by_session_regime", {}),
    )
    edge_thresholds_by_regime = {
        regime: (
            selected_confidence_threshold
            if bool(session_regime_results.get(regime, {}).get("evidence_qualified"))
            else 1.0
        )
        for regime in observed_session_regimes
    }
    validation_report = _evaluate_probabilities(oos_frame["target"], oos_probabilities)
    threshold_report = _threshold_report(oos_frame, oos_probabilities)
    threshold_report_by_regime = _threshold_report_by_regime(
        oos_frame, oos_probabilities
    )
    _attach_model_metadata(
        model,
        edge_global_threshold=selected_confidence_threshold,
        edge_thresholds_by_regime=edge_thresholds_by_regime,
        horizon_bars=int(args.horizon_bars),
        label_objective=str(getattr(args, "label_objective", "net_markout")),
    )
    feature_importance = _feature_importance(model)
    generated_at = datetime.now(UTC)
    market_regime_policy = derive_market_regime_policy(
        walk_forward_report,
        generated_at=generated_at,
    )
    evaluate_holdout = bool(getattr(args, "evaluate_holdout", True))
    holdout_report: dict[str, Any] = {
        "consumed": False,
        "rows": int(len(holdout)),
        "start": str(holdout_partition.holdout_start),
        "selection_authority": False,
        "winner_selection_authority": False,
        "promotion_authority": False,
        "live_money_authority": False,
        "reason": "reserved_for_tournament_winner",
    }
    if evaluate_holdout:
        holdout_features = holdout[list(REPLAY_ALIGNED_FEATURE_COLUMNS)].astype(float)
        holdout_probabilities = np.asarray(
            model.predict_proba(holdout_features), dtype=float
        )[:, _positive_class_index(model)]
        holdout_regime_frame = holdout.copy()
        holdout_regimes, holdout_regime_definition = _fold_market_regimes(
            dataset.loc[
                pd.to_datetime(dataset["timestamp"], errors="coerce", utc=True)
                <= pd.to_datetime(holdout["timestamp"], errors="coerce", utc=True).max()
            ].copy(),
            holdout,
        )
        holdout_regime_frame["market_regime"] = holdout_regimes
        policy_allowed = np.asarray(
            [
                evaluate_market_regime_policy(
                    market_regime_policy,
                    market_regime=regime,
                    now=generated_at,
                ).allowed
                for regime in holdout_regimes
            ],
            dtype=bool,
        )
        holdout_metrics, _ = _selected_post_cost_metrics(
            holdout_regime_frame.loc[policy_allowed].copy(),
            holdout_probabilities[policy_allowed],
            confidence_threshold=selected_confidence_threshold,
            entry_score_threshold=selected_entry_score_threshold,
        )
        _, holdout_by_regime = _regime_post_cost_metrics(
            holdout_regime_frame,
            holdout_probabilities,
            confidence_threshold=selected_confidence_threshold,
            entry_score_threshold=selected_entry_score_threshold,
        )
        holdout_report = {
            "consumed": True,
            "rows": int(len(holdout)),
            "start": str(holdout_partition.holdout_start),
            "selection_authority": False,
            "winner_selection_authority": False,
            "promotion_authority": False,
            "live_money_authority": False,
            "threshold_scope": "frozen_nested_development_threshold",
            "confidence_threshold": selected_confidence_threshold,
            "entry_score_threshold": selected_entry_score_threshold,
            "metrics": holdout_metrics,
            "by_market_regime": holdout_by_regime,
            "regime_definition": holdout_regime_definition,
            "policy_source": "development_walk_forward_only",
            "policy_allowed_rows": int(np.sum(policy_allowed)),
            "policy_abstained_rows": int(np.sum(~policy_allowed)),
            "feature_autopsy": _heldout_feature_autopsy(
                model,
                holdout_features,
                holdout["target"],
                random_state=int(args.random_state),
            ),
        }

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
        max_training_invalid_rate=max_invalid_rate,
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
        walk_forward_folds=max(
            2, int(getattr(args, "walk_forward_folds", 5) or 5)
        ),
        walk_forward_embargo_bars=max(
            1, int(getattr(args, "walk_forward_embargo_bars", 1) or 1)
        ),
        walk_forward_embargo_percent=max(
            0.0,
            float(getattr(args, "walk_forward_embargo_percent", 0.0) or 0.0),
        ),
        edge_weight_max=max(
            1.0,
            float(getattr(args, "edge_weight_max", 5.0) or 5.0),
        ),
        edge_weight_quantile=float(
            np.clip(
                float(getattr(args, "edge_weight_quantile", 0.90) or 0.90),
                0.50,
                1.0,
            )
        ),
        evaluation_folds=(
            int(getattr(args, "evaluation_folds"))
            if getattr(args, "evaluation_folds", None) not in (None, 0, "")
            else None
        ),
        nested_validation_fraction=float(
            getattr(args, "nested_validation_fraction", 0.20) or 0.20
        ),
        nested_min_support=max(
            1, int(getattr(args, "nested_min_support", 25) or 25)
        ),
        evaluate_holdout=evaluate_holdout,
    )
    manifest_path = write_artifact_manifest(
        model_path=model_path,
        model_version=f"replay_aligned_{args.model_type}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
        training_data_range={
            "start": str(development["timestamp"].min()),
            "end": str(development["timestamp"].max()),
        },
        metadata={
            "strategy": "replay_aligned_markout",
            "feature_columns": list(REPLAY_ALIGNED_FEATURE_COLUMNS),
            "objective": (
                f"{config.horizon_bars}_bar_{config.label_objective}_continuous_bps"
                if isinstance(model, ContinuousEdgeEstimator)
                else f"{config.horizon_bars}_bar_{config.label_objective}_binary"
            ),
            "config": asdict(config),
            "authority": _training_authority(dataset),
            "acquisition": acquisition,
            "dataset_hash": acquisition["dataset_hash"],
            "thresholds_by_regime": edge_thresholds_by_regime,
            "threshold_scope": "nested_inner_validation_only",
            "feature_importance": feature_importance[:25],
            "heldout_feature_autopsy": holdout_report.get("feature_autopsy", {}),
            "dataset_quality": dataset_quality,
            "dataset_quality_path": str(dataset_quality_path),
            "live_cost_model": live_cost_metadata,
            "walk_forward": walk_forward_report,
            "market_regime_policy": market_regime_policy,
            "holdout_evaluation": holdout_report,
            "governance_status": "shadow",
            "promotion_authority": False,
            "live_money_authority": False,
        },
    )
    report = {
        "schema_version": "1.0.0",
        "artifact_type": "replay_aligned_training_report",
        "status": "complete",
        "generated_at": generated_at.isoformat(),
        "authority": _training_authority(dataset),
        "acquisition": acquisition,
        "model_path": str(model_path),
        "manifest_path": str(manifest_path),
        "config": asdict(config),
        "dataset": {
            "dataset_hash": acquisition["dataset_hash"],
            "load_reports": dataset.attrs.get("load_reports", {}),
            "shadow_markout_evidence": shadow_markout_evidence,
            "quality": dataset_quality,
            "quality_report_path": str(dataset_quality_path),
            "rows": int(len(dataset)),
            "train_rows": int(len(train)),
            "validation_rows": int(len(oos_frame)),
            "development_rows": int(len(development)),
            "holdout_rows": int(len(holdout)),
            "symbols": int(dataset["symbol"].nunique()),
            "positive_rate": float(dataset["target"].mean()),
            "train_positive_rate": float(train["target"].mean()),
            "validation_positive_rate": float(oos_frame["target"].mean()),
            "mean_round_trip_cost_bps": float(dataset["round_trip_cost_bps"].mean()),
            "mean_entry_slippage_bps": float(dataset["entry_slippage_bps"].mean()),
            "mean_exit_slippage_bps": float(dataset["exit_slippage_bps"].mean()),
            "mean_max_adverse_excursion_bps": float(dataset["max_adverse_excursion_bps"].mean()),
            "mean_max_favorable_excursion_bps": float(dataset["max_favorable_excursion_bps"].mean()),
            "mean_risk_adjusted_net_bps": float(dataset["risk_adjusted_net_bps"].mean()),
            "mean_label_score_bps": float(dataset["label_score_bps"].mean()),
            "mean_execution_adjusted_net_bps": float(
                dataset["execution_adjusted_net_bps"].mean()
            ),
            "passive_fill_probability_proxy": float(
                dataset["passive_fill_probability_proxy"].mean()
            ),
            "mean_opportunity_cost_bps": float(dataset["opportunity_cost_bps"].mean()),
            "split_purge": {
                "method": "per_fold_label_end_timestamp_purge_plus_embargo",
                "folds": [
                    {
                        "fold_index": fold["fold_index"],
                        "purged_train_rows": fold["purged_train_rows"],
                        "embargoed_train_rows": fold["embargoed_train_rows"],
                        "embargo_bars": fold["embargo_bars"],
                        "label_purge_ok": fold["label_purge_ok"],
                    }
                    for fold in walk_forward_report["folds"]
                ],
            },
            "development_holdout": {
                "train_fraction": float(config.train_fraction),
                "holdout_start": str(holdout_partition.holdout_start),
                "initial_development_rows": int(
                    holdout_partition.initial_development_rows
                ),
                "purged_development_rows": int(
                    holdout_partition.purged_development_rows
                ),
                "embargoed_development_rows": int(
                    holdout_partition.embargoed_development_rows
                ),
                "embargo_bars": int(holdout_partition.embargo_bars),
                "unique_timestamp_non_overlap": True,
            },
        },
        "live_cost_model": live_cost_metadata,
        "cost_sensitivity": cost_sensitivity,
        "feature_importance": feature_importance[:25],
        "heldout_feature_autopsy": holdout_report.get("feature_autopsy", {}),
        "validation": validation_report,
        "threshold_sweep": threshold_report,
        "threshold_sweep_by_regime": threshold_report_by_regime,
        "thresholds_by_regime": edge_thresholds_by_regime,
        "threshold_scope": "nested_inner_validation_only",
        "walk_forward": walk_forward_report,
        "market_regime_policy": market_regime_policy,
        "holdout_evaluation": holdout_report,
        "governance_status": "shadow",
        "promotion_authority": False,
        "live_money_authority": False,
        "recommendation": "evaluate_candidate_in_shadow_with_governed_offline_replay",
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
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--data-dir", type=Path, default=None)
    input_group.add_argument(
        "--acquisition-manifest-json",
        type=Path,
        default=None,
        help=(
            "Quality-gated JSON result from historical_training_backfill; "
            "historical evidence remains research-only."
        ),
    )
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--timestamp-col", type=str, default="timestamp")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument(
        "--model-type",
        choices=(
            "logistic",
            "meta_label",
            "random_forest",
            "hist_gradient",
            "edge_linear",
            "edge_hist_gradient",
            "edge_rank",
        ),
        default="hist_gradient",
    )
    parser.add_argument("--horizon-bars", type=int, default=1)
    parser.add_argument(
        "--label-objective",
        choices=(
            "net_markout",
            "spread_adjusted",
            "risk_adjusted",
            "mae_mfe",
            "execution_adjusted",
        ),
        default="net_markout",
        help=(
            "Training label objective. net_markout and spread_adjusted use cost-adjusted "
            "future markout; risk_adjusted and mae_mfe include adverse/favorable excursion; "
            "execution_adjusted also penalizes passive non-fill opportunity cost."
        ),
    )
    parser.add_argument(
        "--max-training-invalid-rate",
        type=float,
        default=0.02,
        help="Fail training when unexpected invalid labeled rows exceed this fraction.",
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
    parser.add_argument("--shadow-markout-jsonl", type=Path, default=None)
    parser.add_argument("--shadow-markout-manifest-json", type=Path, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--walk-forward-folds", type=int, default=5)
    parser.add_argument("--research-experiments", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--experiment-feature-removals", default="macd_signal_gap")
    parser.add_argument("--cost-scenarios-bps", default="0,3,6,10,20")
    parser.add_argument(
        "--evaluation-folds",
        type=int,
        default=None,
        help="Optional fixed-schedule fold budget used by staged research screening.",
    )
    parser.add_argument("--walk-forward-embargo-bars", type=int, default=1)
    parser.add_argument("--walk-forward-embargo-percent", type=float, default=0.0)
    parser.add_argument("--nested-validation-fraction", type=float, default=0.20)
    parser.add_argument("--nested-min-support", type=int, default=25)
    parser.add_argument(
        "--evaluate-holdout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Consume the reserved final holdout; tournaments disable this for non-winners.",
    )
    parser.add_argument("--walk-forward-min-trades", type=int, default=250)
    parser.add_argument(
        "--walk-forward-min-profitable-fold-ratio", type=float, default=0.60
    )
    parser.add_argument("--walk-forward-min-mean-net-edge-bps", type=float, default=0.0)
    parser.add_argument(
        "--walk-forward-min-ranking-separation-bps", type=float, default=0.0
    )
    parser.add_argument(
        "--edge-weight-max",
        type=float,
        default=5.0,
        help="Maximum bounded sample weight for post-cost edge magnitude.",
    )
    parser.add_argument(
        "--edge-weight-quantile",
        type=float,
        default=0.90,
        help="Fit-partition quantile used to scale post-cost edge weights.",
    )
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
