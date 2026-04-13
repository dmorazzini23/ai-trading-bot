from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd

from ai_trading.config.management import get_env, reload_env
from ai_trading.features.indicators import (
    compute_atr,
    compute_macd,
    compute_macds,
    compute_sma,
    compute_vwap,
)
from ai_trading.indicators import rsi as rsi_indicator
from ai_trading.logging import get_logger
from ai_trading.replay.event_loop import ReplayEventLoop

logger = get_logger(__name__)

_REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")


@dataclass(frozen=True)
class ReplayConfig:
    confidence_threshold: float
    entry_score_threshold: float
    allow_shorts: bool
    min_hold_bars: int
    max_hold_bars: int
    stop_loss_bps: float
    take_profit_bps: float
    trailing_stop_bps: float
    fee_bps: float
    slippage_bps: float


@dataclass(frozen=True)
class PolicyReplayProfile:
    opportunity_top_quantile: float
    opportunity_min_symbols: int
    opportunity_openings_only: bool
    expected_capture_fill_prob_floor: float
    expected_capture_floor_bps: float
    expected_capture_constraint_weight: float
    replay_quality_weight: float
    replay_quality_max_rank_uplift_abs: float
    replay_quality_max_rank_uplift_frac: float
    replay_quality_max_age_hours: float
    bandit_score_weight: float
    bandit_min_samples: int
    bandit_shadow_only: bool
    bandit_auto_promote: bool


@dataclass(frozen=True)
class ReplayModelContext:
    model: Any
    model_path: str
    feature_names: tuple[str, ...]
    positive_class_index: int
    orientation_inverse: bool
    symbol_penalties: dict[str, dict[str, float]]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _load_policy_profile_from_env() -> PolicyReplayProfile:
    return PolicyReplayProfile(
        opportunity_top_quantile=_clamp(
            float(get_env("AI_TRADING_EXEC_OPPORTUNITY_TOP_QUANTILE", 0.93, cast=float)),
            0.05,
            0.99,
        ),
        opportunity_min_symbols=max(
            1,
            int(get_env("AI_TRADING_EXEC_OPPORTUNITY_MIN_SYMBOLS", 5, cast=int)),
        ),
        opportunity_openings_only=bool(
            get_env("AI_TRADING_EXEC_OPPORTUNITY_OPENINGS_ONLY", True, cast=bool)
        ),
        expected_capture_fill_prob_floor=_clamp(
            float(
                get_env("AI_TRADING_EXEC_EXPECTED_CAPTURE_FILL_PROB_FLOOR", 0.45, cast=float)
            ),
            0.01,
            0.95,
        ),
        expected_capture_floor_bps=float(
            get_env("AI_TRADING_EXEC_EXPECTED_CAPTURE_FLOOR_BPS", 1.0, cast=float)
        ),
        expected_capture_constraint_weight=max(
            0.0,
            float(
                get_env("AI_TRADING_EXEC_EXPECTED_CAPTURE_CONSTRAINT_WEIGHT", 1.2, cast=float)
            ),
        ),
        replay_quality_weight=max(
            0.0,
            float(get_env("AI_TRADING_EXEC_REPLAY_QUALITY_WEIGHT", 0.18, cast=float)),
        ),
        replay_quality_max_rank_uplift_abs=max(
            0.0,
            float(
                get_env(
                    "AI_TRADING_EXEC_REPLAY_QUALITY_MAX_RANK_UPLIFT_ABS",
                    8.0,
                    cast=float,
                )
            ),
        ),
        replay_quality_max_rank_uplift_frac=_clamp(
            float(
                get_env(
                    "AI_TRADING_EXEC_REPLAY_QUALITY_MAX_RANK_UPLIFT_FRAC",
                    0.10,
                    cast=float,
                )
            ),
            0.0,
            1.0,
        ),
        replay_quality_max_age_hours=max(
            1.0,
            float(get_env("AI_TRADING_EXEC_REPLAY_QUALITY_MAX_AGE_HOURS", 24.0, cast=float)),
        ),
        bandit_score_weight=max(
            0.0,
            float(get_env("AI_TRADING_EXEC_BANDIT_SCORE_WEIGHT", 0.25, cast=float)),
        ),
        bandit_min_samples=max(
            1,
            int(get_env("AI_TRADING_EXEC_BANDIT_MIN_SAMPLES", 40, cast=int)),
        ),
        bandit_shadow_only=bool(
            get_env("AI_TRADING_EXEC_BANDIT_SHADOW_ONLY", True, cast=bool)
        ),
        bandit_auto_promote=bool(
            get_env("AI_TRADING_EXEC_BANDIT_AUTO_PROMOTE", False, cast=bool)
        ),
    )


def _policy_profile_payload(profile: PolicyReplayProfile) -> dict[str, Any]:
    return {
        "opportunity_top_quantile": float(profile.opportunity_top_quantile),
        "opportunity_min_symbols": int(profile.opportunity_min_symbols),
        "opportunity_openings_only": bool(profile.opportunity_openings_only),
        "expected_capture_fill_prob_floor": float(profile.expected_capture_fill_prob_floor),
        "expected_capture_floor_bps": float(profile.expected_capture_floor_bps),
        "expected_capture_constraint_weight": float(profile.expected_capture_constraint_weight),
        "replay_quality_weight": float(profile.replay_quality_weight),
        "replay_quality_max_rank_uplift_abs": float(profile.replay_quality_max_rank_uplift_abs),
        "replay_quality_max_rank_uplift_frac": float(profile.replay_quality_max_rank_uplift_frac),
        "replay_quality_max_age_hours": float(profile.replay_quality_max_age_hours),
        "bandit_score_weight": float(profile.bandit_score_weight),
        "bandit_min_samples": int(profile.bandit_min_samples),
        "bandit_shadow_only": bool(profile.bandit_shadow_only),
        "bandit_auto_promote": bool(profile.bandit_auto_promote),
    }


def _policy_fill_probability_proxy(confidence: float) -> float:
    return _clamp(0.15 + 0.85 * float(confidence), 0.01, 0.99)


def _policy_expected_capture_proxy_bps(
    *,
    score: float,
    confidence: float,
    fill_prob_proxy: float,
    ret_bps: float,
) -> float:
    alpha_component = (abs(float(score)) * 28.0) + (float(confidence) * 12.0)
    execution_component = (1.0 - float(fill_prob_proxy)) * 16.0
    volatility_drag = max(0.0, abs(float(ret_bps)) - 12.0) * 0.25
    return float(alpha_component - execution_component - volatility_drag)


def _policy_keep_count(group_size: int, top_quantile: float, min_symbols: int) -> int:
    if group_size <= 0:
        return 0
    tail_keep = int(math.ceil((1.0 - float(top_quantile)) * float(group_size)))
    return max(1, min(group_size, max(int(min_symbols), tail_keep)))


def _policy_finalize_diagnostics(
    counters: Counter[str],
    *,
    profile: PolicyReplayProfile,
) -> dict[str, Any]:
    candidates = int(counters.get("candidates", 0))
    accepted = int(counters.get("accepted", 0))
    accepted_rate = float(accepted / candidates) if candidates > 0 else 0.0
    rejected_by_reason = {
        key.replace("reject_", ""): int(value)
        for key, value in sorted(counters.items())
        if key.startswith("reject_")
    }
    rejected_total = int(sum(rejected_by_reason.values()))
    rejection_rate = float(rejected_total / candidates) if candidates > 0 else 0.0
    gate_bind_rates: dict[str, dict[str, float | int]] = {}
    gate_bind_ranked: list[dict[str, float | int | str]] = []
    for reason, count in sorted(
        rejected_by_reason.items(),
        key=lambda item: (-int(item[1]), item[0]),
    ):
        bind_rate = float(count / candidates) if candidates > 0 else 0.0
        share_of_rejections = float(count / rejected_total) if rejected_total > 0 else 0.0
        gate_bind_rates[reason] = {
            "count": int(count),
            "bind_rate_of_candidates": bind_rate,
            "share_of_rejections": share_of_rejections,
        }
        gate_bind_ranked.append(
            {
                "reason": reason,
                "count": int(count),
                "bind_rate_of_candidates": bind_rate,
                "share_of_rejections": share_of_rejections,
            }
        )
    return {
        "profile": _policy_profile_payload(profile),
        "candidates": candidates,
        "accepted": accepted,
        "accepted_rate": accepted_rate,
        "rejected_total": rejected_total,
        "rejection_rate": rejection_rate,
        "rejected_by_reason": rejected_by_reason,
        "gate_bind_rates": gate_bind_rates,
        "gate_bind_ranked": gate_bind_ranked,
        "bandit_shadow_candidates": int(counters.get("bandit_shadow_candidates", 0)),
        "bandit_applied": int(counters.get("bandit_applied", 0)),
    }


def _positive_class_probability_index(classes: Any, *, column_count: int) -> int:
    if int(column_count) <= 1:
        return 0
    try:
        labels = list(classes) if classes is not None else []
    except TypeError:
        labels = []
    if len(labels) == int(column_count):
        for idx, label in enumerate(labels):
            try:
                if int(label) == 1:
                    return int(idx)
            except (TypeError, ValueError):
                continue
    return 1


def _resolve_replay_model_path(args: argparse.Namespace) -> Path | None:
    explicit_path = getattr(args, "model_path", None)
    if explicit_path is not None:
        candidate = Path(explicit_path)
    else:
        path_raw = str(get_env("AI_TRADING_MODEL_PATH", "", cast=str) or "").strip()
        if not path_raw:
            return None
        candidate = Path(path_raw)
    return candidate.expanduser().resolve() if not candidate.is_absolute() else candidate


def _extract_symbol_penalties(raw: Any) -> dict[str, dict[str, float]]:
    if not isinstance(raw, Mapping):
        return {}
    penalties: dict[str, dict[str, float]] = {}
    for symbol_raw, payload_raw in raw.items():
        symbol = str(symbol_raw or "").strip().upper()
        if not symbol or not isinstance(payload_raw, Mapping):
            continue
        threshold_bump = float(
            np.clip(float(payload_raw.get("threshold_bump", 0.0) or 0.0), 0.0, 0.95)
        )
        confidence_scale = float(
            np.clip(float(payload_raw.get("confidence_scale", 1.0) or 1.0), 0.1, 1.0)
        )
        penalties[symbol] = {
            "threshold_bump": threshold_bump,
            "confidence_scale": confidence_scale,
            "negative_share": float(payload_raw.get("negative_share", 0.0) or 0.0),
        }
    return penalties


def _load_replay_model_context(args: argparse.Namespace) -> ReplayModelContext | None:
    if not bool(getattr(args, "use_model_score", False)):
        return None
    model_path = _resolve_replay_model_path(args)
    if model_path is None or not model_path.is_file():
        return None
    try:
        import joblib

        model = joblib.load(model_path)
    except Exception as exc:
        logger.warning(
            "OFFLINE_REPLAY_MODEL_LOAD_FAILED",
            extra={"model_path": str(model_path), "error": str(exc)},
        )
        return None
    if not hasattr(model, "predict_proba"):
        logger.warning(
            "OFFLINE_REPLAY_MODEL_SCORING_UNSUPPORTED",
            extra={"model_path": str(model_path), "reason": "missing_predict_proba"},
        )
        return None
    feature_names_raw = getattr(model, "feature_names_in_", None)
    feature_names = tuple(str(name) for name in feature_names_raw) if feature_names_raw is not None else (
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
    orientation_raw = str(getattr(model, "edge_score_orientation_", "direct") or "").strip().lower()
    classes = getattr(model, "classes_", None)
    try:
        class_count = len(list(classes)) if classes is not None else 2
    except TypeError:
        class_count = 2
    context = ReplayModelContext(
        model=model,
        model_path=str(model_path),
        feature_names=feature_names,
        positive_class_index=_positive_class_probability_index(
            classes,
            column_count=class_count,
        ),
        orientation_inverse=orientation_raw in {"inverse", "inverted", "flip"},
        symbol_penalties=_extract_symbol_penalties(
            getattr(model, "edge_negative_symbol_penalties_", None)
        ),
    )
    logger.info(
        "OFFLINE_REPLAY_MODEL_SCORING_ENABLED",
        extra={
            "model_path": context.model_path,
            "feature_count": len(context.feature_names),
            "orientation": "inverse" if context.orientation_inverse else "direct",
            "symbol_penalty_count": len(context.symbol_penalties),
        },
    )
    return context


def _safe_rsi(close_values: np.ndarray) -> np.ndarray:
    if close_values.size <= 0:
        return cast(np.ndarray, np.asarray([], dtype=float))
    try:
        out = rsi_indicator(tuple(close_values.tolist()), 14)
        arr = np.asarray(out, dtype=float)
    except Exception:
        return cast(np.ndarray, np.zeros_like(close_values, dtype=float))
    if arr.size != close_values.size:
        return cast(np.ndarray, np.zeros_like(close_values, dtype=float))
    return cast(np.ndarray, arr)


def _augment_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(frame.get("close"), errors="coerce")
    close_abs = close.abs().replace(0.0, np.nan)
    atr = pd.to_numeric(frame.get("atr"), errors="coerce")
    vwap = pd.to_numeric(frame.get("vwap"), errors="coerce").replace(0.0, np.nan)
    sma_50 = pd.to_numeric(frame.get("sma_50"), errors="coerce")
    sma_200 = pd.to_numeric(frame.get("sma_200"), errors="coerce")
    macd = pd.to_numeric(frame.get("macd"), errors="coerce")
    rsi = pd.to_numeric(frame.get("rsi"), errors="coerce")
    signal = pd.to_numeric(frame.get("signal", frame.get("macds", frame.get("macd"))), errors="coerce")
    frame["signal"] = signal
    frame["atr_pct"] = (atr / close_abs) * 100.0
    frame["vwap_distance"] = (close / vwap) - 1.0
    frame["sma_spread"] = (sma_50 - sma_200) / close_abs
    frame["macd_signal_gap"] = macd - signal
    frame["rsi_centered"] = (rsi - 50.0) / 50.0
    return frame


def _attach_policy_context(bars: list[dict[str, Any]]) -> None:
    if not bars:
        return
    latest_ts = max(
        pd.to_datetime(str(bar.get("ts", "")), errors="coerce", utc=True)
        for bar in bars
    )
    if pd.isna(latest_ts):
        latest_ts = pd.Timestamp("1970-01-01T00:00:00Z")
    ordered = sorted(
        bars,
        key=lambda row: pd.to_datetime(str(row.get("ts", "")), errors="coerce", utc=True),
    )
    prev_close_by_symbol: dict[str, float] = {}
    mean_ret_bps_by_symbol: dict[str, float] = {}
    samples_by_symbol: Counter[str] = Counter()
    for bar in ordered:
        symbol = str(bar.get("symbol", "")).strip().upper()
        close = float(bar.get("close", 0.0) or 0.0)
        score = float(bar.get("score", 0.0) or 0.0)
        confidence = float(bar.get("confidence", 0.0) or 0.0)
        ts = pd.to_datetime(str(bar.get("ts", "")), errors="coerce", utc=True)
        prev_close = prev_close_by_symbol.get(symbol)
        ret_bps = 0.0
        if prev_close is not None and prev_close > 0.0 and close > 0.0:
            ret_bps = float(((close / prev_close) - 1.0) * 10000.0)
        running_mean = float(mean_ret_bps_by_symbol.get(symbol, 0.0))
        mean_ret_bps_by_symbol[symbol] = (0.9 * running_mean) + (0.1 * ret_bps)
        samples_before = int(samples_by_symbol.get(symbol, 0))
        samples_by_symbol[symbol] = samples_before + 1
        prev_close_by_symbol[symbol] = close
        score_sign = 1.0 if score >= 0.0 else -1.0
        fill_prob_proxy = _policy_fill_probability_proxy(confidence)
        expected_capture_proxy = _policy_expected_capture_proxy_bps(
            score=score,
            confidence=confidence,
            fill_prob_proxy=fill_prob_proxy,
            ret_bps=ret_bps,
        )
        replay_quality_proxy = float(
            (0.35 * min(abs(ret_bps), 40.0) * (1.0 if (ret_bps * score_sign) > 0 else -1.0))
            - (0.10 * abs(ret_bps))
        )
        bandit_proxy = float(mean_ret_bps_by_symbol[symbol] * score_sign)
        if pd.isna(ts):
            age_hours = 0.0
        else:
            age_hours = max(0.0, float((latest_ts - ts).total_seconds() / 3600.0))
        bar["policy_fill_prob_proxy"] = fill_prob_proxy
        bar["policy_expected_capture_proxy_bps"] = expected_capture_proxy
        bar["policy_replay_quality_proxy_bps"] = replay_quality_proxy
        bar["policy_bandit_proxy_bps"] = bandit_proxy
        bar["policy_bandit_samples"] = samples_before
        bar["policy_bar_age_hours"] = age_hours
        bar["policy_opportunity_score_raw"] = expected_capture_proxy + replay_quality_proxy

    groups: dict[str, list[dict[str, Any]]] = {}
    for bar in bars:
        groups.setdefault(str(bar.get("ts", "")), []).append(bar)
    for group in groups.values():
        ranked = sorted(
            group,
            key=lambda row: float(row.get("policy_opportunity_score_raw", 0.0)),
            reverse=True,
        )
        group_size = len(ranked)
        for rank_idx, bar in enumerate(ranked):
            bar["policy_rank_index"] = int(rank_idx)
            bar["policy_group_size"] = int(group_size)
            if group_size <= 1:
                rank_quantile = 1.0
            else:
                rank_quantile = 1.0 - (float(rank_idx) / float(group_size - 1))
            bar["policy_rank_quantile"] = float(rank_quantile)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline replay using local bars to evaluate churn and hold quality."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="Path to a single OHLCV CSV file.")
    source.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing <SYMBOL>.csv files.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="",
        help="Explicit symbol name for --csv input. Defaults to CSV filename stem.",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Comma-separated symbols to include when using --data-dir.",
    )
    parser.add_argument(
        "--timestamp-col",
        type=str,
        default="timestamp",
        help="Timestamp column name. Falls back to first parseable datetime column.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.52)
    parser.add_argument("--entry-score-threshold", type=float, default=0.15)
    parser.add_argument(
        "--allow-shorts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow short entries during replay.",
    )
    parser.add_argument("--min-hold-bars", type=int, default=10)
    parser.add_argument("--max-hold-bars", type=int, default=120)
    parser.add_argument("--stop-loss-bps", type=float, default=60.0)
    parser.add_argument("--take-profit-bps", type=float, default=160.0)
    parser.add_argument("--trailing-stop-bps", type=float, default=90.0)
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument(
        "--simulation-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run parity replay using async simulated broker events.",
    )
    parser.add_argument(
        "--replay-seed",
        type=int,
        default=42,
        help="Deterministic seed used by simulation replay mode.",
    )
    parser.add_argument(
        "--max-symbol-notional",
        type=float,
        default=None,
        help="Optional replay per-symbol notional cap for invariant checks.",
    )
    parser.add_argument(
        "--max-gross-notional",
        type=float,
        default=None,
        help="Optional replay gross notional cap for invariant checks.",
    )
    parser.add_argument(
        "--persist-intents",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Persist replay intents/fills to the configured OMS intent store.",
    )
    parser.add_argument(
        "--intent-prefix",
        type=str,
        default="replay",
        help="Prefix used for persisted replay intent IDs and idempotency keys.",
    )
    parser.add_argument(
        "--policy-sensitivity-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run policy sensitivity diagnostics (baseline + per-knob ablation variants) "
            "for simulation mode."
        ),
    )
    parser.add_argument(
        "--apply-policy-controls",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply ranker policy controls (opportunity/capture/replay-quality/bandit) "
            "during normal simulation replay mode."
        ),
    )
    parser.add_argument(
        "--use-model-score",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use runtime ML model probabilities for replay scoring when a model path "
            "is configured."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional explicit model artifact path for replay model scoring.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help=(
            "Optional dotenv file to load before replay config resolution. "
            "Defaults to repository .env when omitted."
        ),
    )
    return parser


def _load_frame(csv_path: Path, timestamp_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} is empty")

    idx: pd.Index | None = None
    lower_map = {col.lower(): col for col in df.columns}
    ts_col = lower_map.get(timestamp_col.lower(), None)
    if ts_col is None:
        ts_col = lower_map.get("timestamp", None)

    if ts_col is not None:
        idx = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        df = df.drop(columns=[ts_col])
    else:
        first_col = df.columns[0]
        first_series = df[first_col]
        if pd.api.types.is_numeric_dtype(first_series):
            idx = pd.RangeIndex(start=0, stop=len(df), step=1)
        else:
            candidate = pd.to_datetime(first_series, errors="coerce", utc=True)
            parse_ratio = float(candidate.notna().mean())
            if parse_ratio >= 0.95:
                idx = candidate
                df = df.drop(columns=[first_col])
            else:
                idx = pd.RangeIndex(start=0, stop=len(df), step=1)

    rename_map = {col: col.lower() for col in df.columns}
    df = df.rename(columns=rename_map)
    missing = [col for col in _REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")
    if "volume" not in df.columns:
        df["volume"] = 0.0

    out = df[list(_REQUIRED_COLUMNS) + ["volume"]].copy()
    out.index = idx
    for col in list(_REQUIRED_COLUMNS) + ["volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=list(_REQUIRED_COLUMNS))
    if out.empty:
        raise ValueError(f"{csv_path} has no valid OHLC rows after cleanup")
    return out.sort_index()


def _compute_signal(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    close = df["close"].astype(float)
    ema_fast = close.ewm(span=8, adjust=False).mean()
    ema_slow = close.ewm(span=21, adjust=False).mean()
    trend = ((ema_fast - ema_slow) / close.replace(0.0, np.nan)).fillna(0.0)
    momentum = close.pct_change(5).fillna(0.0)
    raw = (trend * 12.0) + (momentum * 32.0)
    score = np.tanh(raw).clip(-1.0, 1.0)
    confidence = (trend.abs() * 10.0 + momentum.abs() * 30.0).clip(0.0, 1.0)
    return score.astype(float), confidence.astype(float)


def _compute_model_signal(
    df: pd.DataFrame,
    *,
    symbol: str,
    model_context: ReplayModelContext,
) -> tuple[pd.Series, pd.Series]:
    frame = df.copy()
    frame = compute_macd(frame)
    frame = compute_macds(frame)
    frame = compute_atr(frame)
    frame = compute_vwap(frame)
    frame = compute_sma(frame, windows=(50, 200))
    close_arr = pd.to_numeric(frame.get("close"), errors="coerce").to_numpy(dtype=float)
    frame["rsi"] = _safe_rsi(close_arr)
    frame = _augment_model_features(frame)

    feature_names = list(model_context.feature_names)
    for name in feature_names:
        if name not in frame.columns:
            frame[name] = np.nan
    feature_frame = frame[feature_names].apply(pd.to_numeric, errors="coerce")
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    probs: np.ndarray = np.full(len(frame), 0.5, dtype=float)
    try:
        arr = np.asarray(
            model_context.model.predict_proba(feature_frame.to_numpy(dtype=float)),
            dtype=float,
        )
        if arr.ndim == 1:
            pred_probs = np.asarray(arr, dtype=float)
        else:
            positive_index = int(model_context.positive_class_index)
            if positive_index < 0 or positive_index >= int(arr.shape[1]):
                positive_index = _positive_class_probability_index(
                    getattr(model_context.model, "classes_", None),
                    column_count=int(arr.shape[1]),
                )
            pred_probs = np.asarray(arr[:, positive_index], dtype=float)
        if pred_probs.size == probs.size:
            probs = np.clip(pred_probs, 0.0, 1.0)
        else:
            logger.warning(
                "OFFLINE_REPLAY_MODEL_SCORING_SIZE_MISMATCH",
                extra={"symbol": symbol, "expected": int(probs.size), "actual": int(pred_probs.size)},
            )
    except Exception as exc:
        logger.warning(
            "OFFLINE_REPLAY_MODEL_SCORING_FAILED",
            extra={"symbol": symbol, "error": str(exc)},
        )
        return _compute_signal(df)

    if model_context.orientation_inverse:
        probs = 1.0 - probs
    penalty = model_context.symbol_penalties.get(symbol.strip().upper(), {})
    if penalty:
        confidence_scale = float(
            np.clip(float(penalty.get("confidence_scale", 1.0) or 1.0), 0.1, 1.0)
        )
        threshold_bump = float(
            np.clip(float(penalty.get("threshold_bump", 0.0) or 0.0), 0.0, 0.95)
        )
        long_mask = probs >= 0.5
        probs = np.where(
            long_mask,
            np.clip((probs * confidence_scale) - threshold_bump, 0.0, 1.0),
            probs,
        )

    score = np.clip((2.0 * probs) - 1.0, -1.0, 1.0)
    confidence = np.maximum(probs, 1.0 - probs)
    return pd.Series(score, index=df.index, dtype=float), pd.Series(
        confidence, index=df.index, dtype=float
    )


def _entry_price(close: float, side: int, slippage_bps: float) -> float:
    slip = slippage_bps / 10000.0
    if side > 0:
        return close * (1.0 + slip)
    return close * (1.0 - slip)


def _exit_price(close: float, side: int, slippage_bps: float) -> float:
    slip = slippage_bps / 10000.0
    if side > 0:
        return close * (1.0 - slip)
    return close * (1.0 + slip)


def _profit_factor(wins: np.ndarray, losses: np.ndarray) -> float | None:
    if losses.size == 0:
        if wins.size == 0:
            return 0.0
        return None
    return float(wins.sum() / abs(losses.sum()))


def _max_drawdown_bps(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    values = np.asarray(equity_curve, dtype=float)
    peaks = np.maximum.accumulate(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = np.where(peaks > 0.0, values / peaks - 1.0, 0.0)
    return float(abs(drawdowns.min()) * 10000.0)


def _summarize_markout_fill_metrics(
    *,
    fill_events: list[dict[str, Any]],
    order_context_by_client_id: Mapping[str, Mapping[str, Any]],
    fee_bps: float,
) -> dict[str, Any]:
    """Summarize replay fill quality using one-bar markout from submit context."""

    net_edges: list[float] = []
    edge_weights: list[float] = []
    per_symbol_samples: Counter[str] = Counter()

    for event in fill_events:
        if str(event.get("event_type", "")).strip().lower() != "fill":
            continue
        client_order_id = str(event.get("client_order_id", "")).strip()
        if not client_order_id:
            continue
        context = order_context_by_client_id.get(client_order_id)
        if not isinstance(context, Mapping):
            continue
        markout_raw = context.get("markout_price")
        if markout_raw in (None, ""):
            continue
        try:
            markout_price = float(markout_raw)
            fill_price = float(event.get("fill_price", 0.0) or 0.0)
            fill_qty = float(event.get("fill_qty", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if markout_price <= 0.0 or fill_price <= 0.0 or fill_qty <= 0.0:
            continue
        side = str(event.get("side", context.get("side", "buy"))).strip().lower()
        side_sign = 1.0 if side == "buy" else -1.0
        gross_edge_bps = ((markout_price / fill_price) - 1.0) * 10000.0 * side_sign
        # One-bar markout approximates a full roundtrip by charging both legs.
        net_edge_bps = float(gross_edge_bps - (2.0 * max(0.0, fee_bps)))
        weight = float(abs(fill_qty))
        net_edges.append(net_edge_bps)
        edge_weights.append(weight)
        symbol = str(event.get("symbol", "")).strip().upper()
        if symbol:
            per_symbol_samples[symbol] += 1

    if not net_edges:
        return {
            "samples": 0,
            "win_rate": 0.0,
            "avg_win_bps": 0.0,
            "avg_loss_bps": 0.0,
            "profit_factor": 0.0,
            "expectancy_bps": 0.0,
            "net_pnl_bps": 0.0,
            "per_symbol_samples": dict(per_symbol_samples),
        }

    edges = np.asarray(net_edges, dtype=float)
    weights = np.asarray(edge_weights, dtype=float)
    total_weight = float(weights.sum())
    weighted_edge_sum = float(np.dot(edges, weights))
    wins = edges > 0.0
    losses = edges < 0.0
    win_weight = float(weights[wins].sum()) if np.any(wins) else 0.0
    weighted_win_edge = float(np.dot(edges[wins], weights[wins])) if np.any(wins) else 0.0
    weighted_loss_edge = float(np.dot(edges[losses], weights[losses])) if np.any(losses) else 0.0
    avg_win = (weighted_win_edge / win_weight) if win_weight > 0.0 else 0.0
    loss_weight = float(weights[losses].sum()) if np.any(losses) else 0.0
    avg_loss = (abs(weighted_loss_edge) / loss_weight) if loss_weight > 0.0 else 0.0
    if abs(weighted_loss_edge) <= 1e-12:
        profit_factor: float | None
        profit_factor = None if weighted_win_edge > 0.0 else 0.0
    else:
        profit_factor = float(weighted_win_edge / abs(weighted_loss_edge))
    return {
        "samples": int(edges.size),
        "win_rate": float((win_weight / total_weight) if total_weight > 0.0 else 0.0),
        "avg_win_bps": float(avg_win),
        "avg_loss_bps": float(avg_loss),
        "profit_factor": profit_factor,
        "expectancy_bps": float((weighted_edge_sum / total_weight) if total_weight > 0.0 else 0.0),
        "net_pnl_bps": float(weighted_edge_sum),
        "per_symbol_samples": dict(per_symbol_samples),
    }


def _simulate_symbol(symbol: str, df: pd.DataFrame, cfg: ReplayConfig) -> dict[str, Any]:
    score, confidence = _compute_signal(df)
    trades: list[dict[str, Any]] = []

    side = 0
    entry_price = 0.0
    entry_bar = -1
    entry_ts: str | None = None
    best_price = 0.0
    position_bars = 0
    equity = 1.0
    equity_curve: list[float] = [equity]

    for i, (ts, row) in enumerate(df.iterrows()):
        close = float(row["close"])
        if close <= 0.0:
            continue
        s = float(score.iloc[i])
        conf = float(confidence.iloc[i])

        if side != 0:
            hold_bars = i - entry_bar
            position_bars += 1
            if side > 0:
                best_price = max(best_price, close)
                adverse_from_best_bps = (close / best_price - 1.0) * 10000.0
            else:
                best_price = min(best_price, close)
                adverse_from_best_bps = (best_price / close - 1.0) * 10000.0
            pnl_bps_live = ((close / entry_price) - 1.0) * 10000.0 * side

            exit_reason: str | None = None
            if pnl_bps_live <= -cfg.stop_loss_bps:
                exit_reason = "stop_loss"
            elif hold_bars >= cfg.max_hold_bars:
                exit_reason = "max_hold"
            elif hold_bars >= cfg.min_hold_bars and pnl_bps_live >= cfg.take_profit_bps:
                exit_reason = "take_profit"
            elif (
                hold_bars >= cfg.min_hold_bars
                and pnl_bps_live > 0.0
                and adverse_from_best_bps <= -cfg.trailing_stop_bps
            ):
                exit_reason = "trailing_stop"
            elif hold_bars >= cfg.min_hold_bars:
                long_flip = side > 0 and s <= -cfg.entry_score_threshold
                short_flip = side < 0 and s >= cfg.entry_score_threshold
                if (long_flip or short_flip) and conf >= cfg.confidence_threshold:
                    exit_reason = "signal_flip"

            if exit_reason is not None:
                fill_exit = _exit_price(close, side, cfg.slippage_bps)
                pnl_bps = ((fill_exit / entry_price) - 1.0) * 10000.0 * side
                pnl_bps -= 2.0 * cfg.fee_bps
                equity *= 1.0 + (pnl_bps / 10000.0)
                trades.append(
                    {
                        "symbol": symbol,
                        "entry_ts": entry_ts,
                        "exit_ts": str(ts),
                        "side": "long" if side > 0 else "short",
                        "hold_bars": hold_bars,
                        "pnl_bps": float(pnl_bps),
                        "exit_reason": exit_reason,
                    }
                )
                side = 0
                entry_price = 0.0
                entry_bar = -1
                entry_ts = None
                best_price = 0.0

        if side == 0:
            open_long = conf >= cfg.confidence_threshold and s >= cfg.entry_score_threshold
            open_short = (
                cfg.allow_shorts
                and conf >= cfg.confidence_threshold
                and s <= -cfg.entry_score_threshold
            )
            if open_long:
                side = 1
            elif open_short:
                side = -1

            if side != 0:
                entry_price = _entry_price(close, side, cfg.slippage_bps)
                entry_bar = i
                entry_ts = str(ts)
                best_price = close

        equity_curve.append(equity)

    if side != 0 and entry_bar >= 0:
        close = float(df["close"].iloc[-1])
        fill_exit = _exit_price(close, side, cfg.slippage_bps)
        pnl_bps = ((fill_exit / entry_price) - 1.0) * 10000.0 * side
        pnl_bps -= 2.0 * cfg.fee_bps
        hold_bars = max(0, len(df) - 1 - entry_bar)
        equity *= 1.0 + (pnl_bps / 10000.0)
        trades.append(
            {
                "symbol": symbol,
                "entry_ts": entry_ts,
                "exit_ts": str(df.index[-1]),
                "side": "long" if side > 0 else "short",
                "hold_bars": hold_bars,
                "pnl_bps": float(pnl_bps),
                "exit_reason": "end_of_data",
            }
        )
        equity_curve.append(equity)

    pnl = np.asarray([float(t["pnl_bps"]) for t in trades], dtype=float)
    holds = np.asarray([float(t["hold_bars"]) for t in trades], dtype=float)
    wins = pnl[pnl > 0.0]
    losses = pnl[pnl < 0.0]
    trade_count = int(pnl.size)

    summary: dict[str, Any] = {
        "symbol": symbol,
        "bars": int(len(df)),
        "trades": trade_count,
        "win_rate": float((wins.size / trade_count) if trade_count else 0.0),
        "avg_win_bps": float(wins.mean()) if wins.size else 0.0,
        "avg_loss_bps": float(abs(losses.mean())) if losses.size else 0.0,
        "profit_factor": _profit_factor(wins, losses),
        "expectancy_bps": float(pnl.mean()) if trade_count else 0.0,
        "net_pnl_bps": float(pnl.sum()) if trade_count else 0.0,
        "median_hold_bars": float(np.median(holds)) if holds.size else 0.0,
        "churn_trades_per_100_bars": float((trade_count / len(df)) * 100.0),
        "exposure_ratio": float(position_bars / len(df)),
        "max_drawdown_bps": _max_drawdown_bps(equity_curve),
        "trades_detail": trades,
    }
    return summary


def _resolve_inputs(args: argparse.Namespace) -> dict[str, Path]:
    if args.csv is not None:
        csv_path = Path(args.csv)
        if str(csv_path).strip() in {"", "."}:
            raise ValueError("--csv must point to an OHLCV CSV file, got empty path")
        if not csv_path.exists():
            raise ValueError(f"CSV file not found: {csv_path}")
        if not csv_path.is_file():
            raise ValueError(f"--csv must be a file, got: {csv_path}")
        symbol = args.symbol.strip().upper() if args.symbol else args.csv.stem.upper()
        return {symbol: csv_path}

    assert args.data_dir is not None
    chosen = {item.strip().upper() for item in args.symbols.split(",") if item.strip()}
    paths: dict[str, Path] = {}
    for csv_path in sorted(args.data_dir.glob("*.csv")):
        symbol = csv_path.stem.upper()
        if chosen and symbol not in chosen:
            continue
        paths[symbol] = csv_path
    if not paths:
        raise ValueError("No matching CSV files found for replay")
    return paths


def _normalize_bar_ts(value: Any, fallback_index: int) -> str:
    """Return UTC ISO timestamp for replay bars."""

    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        ts = pd.Timestamp("1970-01-01T00:00:00Z") + pd.Timedelta(minutes=fallback_index)
    return str(ts.isoformat())


def _run_parity_simulation(
    *,
    args: argparse.Namespace,
    cfg: ReplayConfig,
    symbol_paths: dict[str, Path],
    model_context: ReplayModelContext | None = None,
    policy_profile: PolicyReplayProfile | None = None,
    policy_mode_label: str = "ranker_sensitivity",
) -> dict[str, Any]:
    """Run deterministic replay parity simulation with async fill events."""

    bars: list[dict[str, Any]] = []
    per_symbol: list[dict[str, Any]] = []
    order_context_by_client_id: dict[str, dict[str, Any]] = {}
    policy_counters: Counter[str] = Counter()
    synthetic_index = 0
    for symbol, csv_path in symbol_paths.items():
        frame = _load_frame(csv_path, args.timestamp_col)
        score_source = "heuristic"
        if model_context is not None:
            score, confidence = _compute_model_signal(
                frame,
                symbol=symbol,
                model_context=model_context,
            )
            score_source = "model"
        else:
            score, confidence = _compute_signal(frame)
        per_symbol.append({"symbol": symbol, "bars": int(len(frame)), "score_source": score_source})
        for pos, (ts, row) in enumerate(frame.iterrows()):
            close = float(row["close"])
            if not np.isfinite(close) or close <= 0.0:
                continue
            ts_iso = _normalize_bar_ts(ts, synthetic_index)
            next_close: float | None = None
            if pos + 1 < len(frame):
                candidate = float(frame["close"].iloc[pos + 1])
                if np.isfinite(candidate) and candidate > 0.0:
                    next_close = candidate
            bar_seq = synthetic_index
            synthetic_index += 1
            bars.append(
                {
                    "symbol": symbol,
                    "ts": ts_iso,
                    "close": close,
                    "score": float(score.iloc[pos]),
                    "confidence": float(confidence.iloc[pos]),
                    "seq": int(bar_seq),
                    "next_close": next_close,
                }
            )
    if policy_profile is not None:
        _attach_policy_context(bars)

    def strategy(bar: Mapping[str, Any]) -> Mapping[str, Any] | None:
        symbol = str(bar.get("symbol", "")).upper()
        ts_iso = str(bar.get("ts", ""))
        try:
            score = float(bar.get("score", 0.0) or 0.0)
            confidence = float(bar.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        if confidence < cfg.confidence_threshold:
            return None
        side: str | None = None
        if score >= cfg.entry_score_threshold:
            side = "buy"
        elif cfg.allow_shorts and score <= -cfg.entry_score_threshold:
            side = "sell"
        if side is None:
            return None
        price = float(bar.get("close", 0.0) or 0.0)
        if not np.isfinite(price) or price <= 0.0:
            return None

        fill_prob_proxy = float(bar.get("policy_fill_prob_proxy", confidence))
        expected_capture_proxy = float(bar.get("policy_expected_capture_proxy_bps", 0.0))
        replay_quality_proxy = float(bar.get("policy_replay_quality_proxy_bps", 0.0))
        bandit_proxy = float(bar.get("policy_bandit_proxy_bps", 0.0))
        bandit_samples = int(bar.get("policy_bandit_samples", 0) or 0)
        bar_age_hours = float(bar.get("policy_bar_age_hours", 0.0) or 0.0)
        rank_index = int(bar.get("policy_rank_index", 0) or 0)
        group_size = int(bar.get("policy_group_size", 1) or 1)
        adjusted_capture = float(expected_capture_proxy)
        replay_adjustment = 0.0
        bandit_adjustment = 0.0

        if policy_profile is not None:
            policy_counters["candidates"] += 1
            keep_count = _policy_keep_count(
                group_size=group_size,
                top_quantile=policy_profile.opportunity_top_quantile,
                min_symbols=policy_profile.opportunity_min_symbols,
            )
            if rank_index >= keep_count:
                policy_counters["reject_opportunity_quantile"] += 1
                return None
            if fill_prob_proxy < policy_profile.expected_capture_fill_prob_floor:
                policy_counters["reject_fill_prob_floor"] += 1
                return None

            if (
                policy_profile.replay_quality_weight > 0.0
                and bar_age_hours <= policy_profile.replay_quality_max_age_hours
            ):
                replay_raw = policy_profile.replay_quality_weight * replay_quality_proxy
                replay_cap_abs = max(0.0, policy_profile.replay_quality_max_rank_uplift_abs)
                replay_cap_frac = (
                    abs(expected_capture_proxy) * policy_profile.replay_quality_max_rank_uplift_frac
                )
                replay_cap = max(0.0, min(replay_cap_abs, replay_cap_frac))
                if replay_cap > 0.0:
                    replay_adjustment = float(np.clip(replay_raw, -replay_cap, replay_cap))
                    adjusted_capture += replay_adjustment

            if (
                policy_profile.bandit_score_weight > 0.0
                and bandit_samples >= policy_profile.bandit_min_samples
            ):
                bandit_adjustment = policy_profile.bandit_score_weight * bandit_proxy
                if policy_profile.bandit_shadow_only:
                    policy_counters["bandit_shadow_candidates"] += 1
                else:
                    adjusted_capture += bandit_adjustment
                    policy_counters["bandit_applied"] += 1

            adjusted_capture -= max(0.0, policy_profile.expected_capture_constraint_weight - 1.0)
            if adjusted_capture < policy_profile.expected_capture_floor_bps:
                policy_counters["reject_expected_capture_floor"] += 1
                return None
            policy_counters["accepted"] += 1

        bar_seq = int(bar.get("seq", 0) or 0)
        intent_key = f"{symbol}|{ts_iso}|{side}|{bar_seq}"
        next_close_raw = bar.get("next_close")
        next_close = None
        if next_close_raw not in (None, ""):
            try:
                parsed_next = float(next_close_raw)
            except (TypeError, ValueError):
                parsed_next = None
            if parsed_next is not None and np.isfinite(parsed_next) and parsed_next > 0.0:
                next_close = float(parsed_next)
        order_context_by_client_id[intent_key] = {
            "symbol": symbol,
            "side": side,
            "submit_ts": ts_iso,
            "submit_price": float(price),
            "markout_price": next_close,
            "policy_fill_prob_proxy": float(fill_prob_proxy),
            "policy_expected_capture_proxy_bps": float(expected_capture_proxy),
            "policy_expected_capture_adjusted_bps": float(adjusted_capture),
            "policy_replay_adjustment_bps": float(replay_adjustment),
            "policy_bandit_adjustment_bps": float(bandit_adjustment),
            "policy_rank_index": int(rank_index),
            "policy_group_size": int(group_size),
        }
        return {
            "symbol": symbol,
            "side": side,
            "qty": 1.0,
            "type": "limit",
            "price": price,
            "limit_price": price,
            "intent_key": intent_key,
            "client_order_id": intent_key,
        }

    replay = ReplayEventLoop(
        strategy=strategy,
        seed=int(args.replay_seed),
        max_symbol_notional=args.max_symbol_notional,
        max_gross_notional=args.max_gross_notional,
    ).run(bars)

    events = list(replay.get("events", []))
    fill_events = [event for event in events if event.get("event_type") == "fill"]
    fill_count_by_symbol = Counter(
        str(event.get("symbol", "")).strip().upper()
        for event in fill_events
        if str(event.get("symbol", "")).strip()
    )
    violations = list(replay.get("violations", []))
    violation_counts = Counter(
        str(item.get("code", "unknown"))
        for item in violations
    )
    markout_metrics = _summarize_markout_fill_metrics(
        fill_events=fill_events,
        order_context_by_client_id=order_context_by_client_id,
        fee_bps=cfg.fee_bps,
    )
    markout_samples_by_symbol = markout_metrics.get("per_symbol_samples", {})
    if not isinstance(markout_samples_by_symbol, dict):
        markout_samples_by_symbol = {}
    total_bars = int(len(bars))
    total_trades = int(len(fill_events))
    aggregate: dict[str, Any] = {
        "simulation_mode": True,
        "replay_seed": int(args.replay_seed),
        "symbols": len(per_symbol),
        "total_bars": total_bars,
        "total_trades": total_trades,
        "win_rate": float(markout_metrics.get("win_rate", 0.0)),
        "avg_win_bps": float(markout_metrics.get("avg_win_bps", 0.0)),
        "avg_loss_bps": float(markout_metrics.get("avg_loss_bps", 0.0)),
        "profit_factor": markout_metrics.get("profit_factor", 0.0),
        "expectancy_bps": float(markout_metrics.get("expectancy_bps", 0.0)),
        "net_pnl_bps": float(markout_metrics.get("net_pnl_bps", 0.0)),
        "median_hold_bars": 1.0 if int(markout_metrics.get("samples", 0)) > 0 else 0.0,
        "churn_trades_per_100_bars": float((total_trades / max(total_bars, 1)) * 100.0),
        "exposure_ratio": 0.0,
        "orders_submitted": int(len(replay.get("orders", []))),
        "intents_created": int(len(replay.get("intents", []))),
        "fill_events": total_trades,
        "markout_samples": int(markout_metrics.get("samples", 0)),
        "metrics_mode": "one_bar_markout",
        "markout_horizon_bars": 1,
        "violation_count": int(len(violations)),
        "violations_by_code": dict(sorted(violation_counts.items())),
        "config": {
            "confidence_threshold": cfg.confidence_threshold,
            "entry_score_threshold": cfg.entry_score_threshold,
            "allow_shorts": cfg.allow_shorts,
            "replay_seed": int(args.replay_seed),
            "max_symbol_notional": args.max_symbol_notional,
            "max_gross_notional": args.max_gross_notional,
            "use_model_score": bool(model_context is not None),
        },
    }
    if model_context is not None:
        aggregate["model_score"] = {
            "enabled": True,
            "model_path": model_context.model_path,
            "orientation": "inverse" if model_context.orientation_inverse else "direct",
            "symbol_penalty_count": len(model_context.symbol_penalties),
            "feature_count": len(model_context.feature_names),
        }
    else:
        aggregate["model_score"] = {"enabled": False}
    if policy_profile is not None:
        aggregate["policy_mode"] = str(policy_mode_label or "ranker_sensitivity")
        aggregate["policy_diagnostics"] = _policy_finalize_diagnostics(
            policy_counters,
            profile=policy_profile,
        )
    for item in per_symbol:
        symbol = str(item.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        item["fills"] = int(fill_count_by_symbol.get(symbol, 0))
        item["markout_samples"] = int(markout_samples_by_symbol.get(symbol, 0))
    return {
        "aggregate": aggregate,
        "symbols": per_symbol,
        "replay": replay,
    }


def _policy_metric_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = payload.get("aggregate", {})
    if not isinstance(aggregate, Mapping):
        return {}
    return {
        "total_trades": int(aggregate.get("total_trades", 0) or 0),
        "win_rate": float(aggregate.get("win_rate", 0.0) or 0.0),
        "profit_factor": (
            None
            if aggregate.get("profit_factor") is None
            else float(aggregate.get("profit_factor", 0.0) or 0.0)
        ),
        "expectancy_bps": float(aggregate.get("expectancy_bps", 0.0) or 0.0),
        "markout_samples": int(aggregate.get("markout_samples", 0) or 0),
    }


def _profit_factor_delta(new: Any, old: Any) -> float | None:
    if old is None or new is None:
        return None
    try:
        return float(new) - float(old)
    except (TypeError, ValueError):
        return None


def _policy_sensitivity_summary_table(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda item: float(item.get("delta_expectancy_bps", 0.0)),
        reverse=True,
    )
    table: list[dict[str, Any]] = []
    for idx, row in enumerate(ranked, start=1):
        table.append(
            {
                "rank": int(idx),
                "name": str(row.get("name", "")),
                "description": str(row.get("description", "")),
                "delta_expectancy_bps": float(row.get("delta_expectancy_bps", 0.0)),
                "delta_total_trades": int(row.get("delta_total_trades", 0)),
                "delta_profit_factor": row.get("delta_profit_factor"),
                "delta_win_rate": float(row.get("delta_win_rate", 0.0)),
                "delta_markout_samples": int(row.get("delta_markout_samples", 0)),
            }
        )
    return table


def _run_policy_sensitivity(
    *,
    args: argparse.Namespace,
    cfg: ReplayConfig,
    symbol_paths: dict[str, Path],
    model_context: ReplayModelContext | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_profile = _load_policy_profile_from_env()
    baseline_payload = _run_parity_simulation(
        args=args,
        cfg=cfg,
        symbol_paths=symbol_paths,
        model_context=model_context,
        policy_profile=baseline_profile,
        policy_mode_label="ranker_sensitivity",
    )
    baseline_metrics = _policy_metric_projection(baseline_payload)
    baseline_diag = (
        baseline_payload.get("aggregate", {}).get("policy_diagnostics", {})
        if isinstance(baseline_payload.get("aggregate"), Mapping)
        else {}
    )

    variants: list[tuple[str, str, PolicyReplayProfile]] = [
        (
            "opportunity_gate_disabled",
            "Disable opportunity top-quantile gate (keep all symbols).",
            replace(baseline_profile, opportunity_top_quantile=0.05, opportunity_min_symbols=1),
        ),
        (
            "capture_floor_disabled",
            "Disable expected-capture floors (fill prob + edge floor).",
            replace(
                baseline_profile,
                expected_capture_fill_prob_floor=0.01,
                expected_capture_floor_bps=-1_000_000.0,
            ),
        ),
        (
            "replay_quality_disabled",
            "Disable replay-quality deweight/uplift adjustment.",
            replace(baseline_profile, replay_quality_weight=0.0),
        ),
        (
            "bandit_disabled",
            "Disable bandit rank adjustment contribution.",
            replace(baseline_profile, bandit_score_weight=0.0),
        ),
        (
            "bandit_live_enabled",
            "Enable live bandit rank adjustment (disable shadow-only mode).",
            replace(baseline_profile, bandit_shadow_only=False),
        ),
        (
            "replay_quality_weight_0_10",
            "Replay-quality weight sweep variant: 0.10.",
            replace(baseline_profile, replay_quality_weight=0.10),
        ),
        (
            "replay_quality_weight_0_25",
            "Replay-quality weight sweep variant: 0.25.",
            replace(baseline_profile, replay_quality_weight=0.25),
        ),
        (
            "replay_quality_weight_0_40",
            "Replay-quality weight sweep variant: 0.40.",
            replace(baseline_profile, replay_quality_weight=0.40),
        ),
    ]

    variant_rows: list[dict[str, Any]] = []
    for name, description, profile in variants:
        variant_payload = _run_parity_simulation(
            args=args,
            cfg=cfg,
            symbol_paths=symbol_paths,
            model_context=model_context,
            policy_profile=profile,
            policy_mode_label="ranker_sensitivity",
        )
        variant_metrics = _policy_metric_projection(variant_payload)
        variant_diag = (
            variant_payload.get("aggregate", {}).get("policy_diagnostics", {})
            if isinstance(variant_payload.get("aggregate"), Mapping)
            else {}
        )
        row = {
            "name": name,
            "description": description,
            "profile": _policy_profile_payload(profile),
            "metrics": variant_metrics,
            "policy_diagnostics": variant_diag,
            "delta_vs_baseline": {
                "total_trades": int(variant_metrics["total_trades"]) - int(baseline_metrics["total_trades"]),
                "win_rate": float(variant_metrics["win_rate"]) - float(baseline_metrics["win_rate"]),
                "profit_factor": _profit_factor_delta(
                    variant_metrics.get("profit_factor"),
                    baseline_metrics.get("profit_factor"),
                ),
                "expectancy_bps": float(variant_metrics["expectancy_bps"])
                - float(baseline_metrics["expectancy_bps"]),
                "markout_samples": int(variant_metrics["markout_samples"])
                - int(baseline_metrics["markout_samples"]),
            },
        }
        variant_rows.append(row)

    contributions = sorted(
        [
            {
                "name": row["name"],
                "description": row["description"],
                "delta_expectancy_bps": float(row["delta_vs_baseline"]["expectancy_bps"]),
                "delta_total_trades": int(row["delta_vs_baseline"]["total_trades"]),
                "delta_win_rate": float(row["delta_vs_baseline"]["win_rate"]),
                "delta_profit_factor": row["delta_vs_baseline"]["profit_factor"],
                "delta_markout_samples": int(row["delta_vs_baseline"]["markout_samples"]),
            }
            for row in variant_rows
        ],
        key=lambda item: abs(float(item["delta_expectancy_bps"])),
        reverse=True,
    )

    report: dict[str, Any] = {
        "enabled": True,
        "baseline": {
            "profile": _policy_profile_payload(baseline_profile),
            "metrics": baseline_metrics,
            "policy_diagnostics": baseline_diag,
        },
        "variants": variant_rows,
        "per_knob_contribution": contributions,
        "summary_table": _policy_sensitivity_summary_table(contributions),
    }
    logger.info(
        "OFFLINE_REPLAY_POLICY_SENSITIVITY_COMPLETE",
        extra={
            "variant_count": len(variant_rows),
            "baseline_trades": baseline_metrics.get("total_trades", 0),
            "baseline_expectancy_bps": baseline_metrics.get("expectancy_bps", 0.0),
        },
    )
    return baseline_payload, report


def _persist_replay_to_oms(
    *,
    replay: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    """Persist replay intents and fill events to durable OMS storage."""

    from ai_trading.oms.intent_store import IntentStore
    from ai_trading.oms.statuses import is_terminal_intent_status

    now_iso = datetime.now(UTC).isoformat()
    started = perf_counter()
    intent_items = replay.get("intents")
    order_items = replay.get("orders")
    event_items = replay.get("events")
    intents = intent_items if isinstance(intent_items, list) else []
    orders = order_items if isinstance(order_items, list) else []
    events = event_items if isinstance(event_items, list) else []
    logger.info(
        "OFFLINE_REPLAY_OMS_PERSIST_START",
        extra={
            "intents": len(intents),
            "orders": len(orders),
            "events": len(events),
            "prefix": prefix,
        },
    )

    store = IntentStore()
    order_by_client_id: dict[str, dict[str, Any]] = {}
    for order in orders:
        if not isinstance(order, dict):
            continue
        client_order_id = str(order.get("client_order_id", "")).strip()
        if client_order_id:
            order_by_client_id[client_order_id] = order

    intent_id_by_client_id: dict[str, str] = {}
    terminal_existing_ids: set[str] = set()
    created_count = 0
    existing_count = 0
    existing_terminal_skipped = 0
    skipped_count = 0
    for intent in intents:
        if not isinstance(intent, dict):
            skipped_count += 1
            continue
        client_order_id = str(intent.get("intent_key", "")).strip()
        symbol = str(intent.get("symbol", "")).strip().upper()
        side = str(intent.get("side", "buy")).strip().lower()
        try:
            qty = float(intent.get("qty", 0.0) or 0.0)
        except (TypeError, ValueError):
            qty = 0.0
        decision_ts = str(intent.get("ts", "")).strip() or now_iso
        if not client_order_id or not symbol or qty <= 0.0:
            skipped_count += 1
            continue
        digest = hashlib.sha256(
            f"{prefix}|{client_order_id}".encode("utf-8")
        ).hexdigest()[:24]
        intent_id = f"{prefix}-{digest}"
        idempotency_key = f"{prefix}|{client_order_id}"
        existing_record = store.get_intent_by_key(idempotency_key)
        if existing_record is not None:
            record = existing_record
            existing_count += 1
            if is_terminal_intent_status(existing_record.status):
                existing_terminal_skipped += 1
                terminal_existing_ids.add(existing_record.intent_id)
        else:
            record, created = store.create_intent(
                intent_id=intent_id,
                idempotency_key=idempotency_key,
                symbol=symbol,
                side=side,
                quantity=qty,
                decision_ts=decision_ts,
                metadata={
                    "source": "offline_replay",
                    "client_order_id": client_order_id,
                    "persisted_at": now_iso,
                },
                status="PENDING_SUBMIT",
            )
            if created:
                created_count += 1
            else:
                existing_count += 1
        intent_id_by_client_id[client_order_id] = record.intent_id
        if existing_record is None:
            order = order_by_client_id.get(client_order_id, {})
            broker_order_id = str(order.get("id", "")).strip() if isinstance(order, dict) else ""
            if not broker_order_id:
                broker_order_id = f"{prefix}-no-order-{digest}"
            store.mark_submitted(record.intent_id, broker_order_id)

    fill_events = 0
    partially_filled = 0
    filled_terminal = 0
    fill_intent_ids: set[str] = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        if str(event.get("event_type", "")).strip().lower() != "fill":
            continue
        client_order_id = str(event.get("client_order_id", "")).strip()
        intent_id = intent_id_by_client_id.get(client_order_id)
        if not intent_id:
            continue
        if intent_id in terminal_existing_ids:
            continue
        fill_events += 1
        try:
            fill_qty = float(event.get("fill_qty", 0.0) or 0.0)
        except (TypeError, ValueError):
            fill_qty = 0.0
        try:
            fill_price = float(event.get("fill_price")) if event.get("fill_price") is not None else None
        except (TypeError, ValueError):
            fill_price = None
        fill_ts = str(event.get("ts", "")).strip() or None
        if fill_qty > 0.0:
            store.record_fill(
                intent_id,
                fill_qty=fill_qty,
                fill_price=fill_price,
                fill_ts=fill_ts,
            )
            fill_intent_ids.add(intent_id)
        status = str(event.get("status", "")).strip().lower()
        if status == "filled":
            store.close_intent(intent_id, final_status="FILLED")
            filled_terminal += 1
        elif status in {"partially_filled", "partial_fill"}:
            partially_filled += 1

    closed_without_fill = 0
    for client_order_id, intent_id in intent_id_by_client_id.items():
        if intent_id in fill_intent_ids:
            continue
        if intent_id in terminal_existing_ids:
            continue
        order = order_by_client_id.get(client_order_id, {})
        order_status = str(order.get("status", "")).strip().lower() if isinstance(order, dict) else ""
        final_status = "CANCELLED" if order_status in {"accepted", "new", "open"} else "CLOSED"
        store.close_intent(
            intent_id,
            final_status=final_status,
            last_error="offline_replay_unfilled",
        )
        closed_without_fill += 1

    summary: dict[str, Any] = {
        "persisted": True,
        "database_url_configured": bool(getattr(store, "database_url", "")),
        "created_intents": created_count,
        "existing_intents": existing_count,
        "existing_terminal_intents_skipped": existing_terminal_skipped,
        "skipped_intents": skipped_count,
        "fill_events": fill_events,
        "partially_filled_events": partially_filled,
        "filled_terminal_events": filled_terminal,
        "closed_without_fill": closed_without_fill,
    }
    summary["elapsed_seconds"] = round(perf_counter() - started, 3)
    logger.info("OFFLINE_REPLAY_OMS_PERSIST_COMPLETE", extra=summary)
    return summary


def _run_replay(args: argparse.Namespace) -> dict[str, Any]:
    cfg = ReplayConfig(
        confidence_threshold=float(args.confidence_threshold),
        entry_score_threshold=float(args.entry_score_threshold),
        allow_shorts=bool(args.allow_shorts),
        min_hold_bars=max(1, int(args.min_hold_bars)),
        max_hold_bars=max(2, int(args.max_hold_bars)),
        stop_loss_bps=max(1.0, float(args.stop_loss_bps)),
        take_profit_bps=max(1.0, float(args.take_profit_bps)),
        trailing_stop_bps=max(1.0, float(args.trailing_stop_bps)),
        fee_bps=max(0.0, float(args.fee_bps)),
        slippage_bps=max(0.0, float(args.slippage_bps)),
    )
    if cfg.max_hold_bars < cfg.min_hold_bars:
        raise ValueError("max-hold-bars must be >= min-hold-bars")

    symbol_paths = _resolve_inputs(args)
    if bool(args.simulation_mode):
        model_context = _load_replay_model_context(args)
        if bool(args.policy_sensitivity_mode) and bool(args.persist_intents):
            raise ValueError("--policy-sensitivity-mode cannot be combined with --persist-intents")
        if bool(args.policy_sensitivity_mode):
            payload, sensitivity_report = _run_policy_sensitivity(
                args=args,
                cfg=cfg,
                symbol_paths=symbol_paths,
                model_context=model_context,
            )
            payload["policy_sensitivity"] = sensitivity_report
            payload["aggregate"]["policy_sensitivity_mode"] = True
            payload["aggregate"]["policy_sensitivity_variant_count"] = int(
                len(sensitivity_report.get("variants", []))
            )
        else:
            policy_profile: PolicyReplayProfile | None = None
            if bool(args.apply_policy_controls):
                policy_profile = _load_policy_profile_from_env()
            payload = _run_parity_simulation(
                args=args,
                cfg=cfg,
                symbol_paths=symbol_paths,
                model_context=model_context,
                policy_profile=policy_profile,
                policy_mode_label="ranker_controls",
            )
            payload["aggregate"]["policy_controls_applied"] = bool(policy_profile is not None)
        if bool(args.persist_intents):
            persist_summary = _persist_replay_to_oms(
                replay=payload.get("replay", {}),
                prefix=str(args.intent_prefix or "replay").strip() or "replay",
            )
            payload["aggregate"]["oms_persist_summary"] = persist_summary
        return payload
    if bool(args.persist_intents):
        raise ValueError("--persist-intents requires --simulation-mode")
    if bool(args.policy_sensitivity_mode):
        raise ValueError("--policy-sensitivity-mode requires --simulation-mode")
    if bool(args.apply_policy_controls):
        raise ValueError("--apply-policy-controls requires --simulation-mode")

    per_symbol: list[dict[str, Any]] = []
    for symbol, csv_path in symbol_paths.items():
        frame = _load_frame(csv_path, args.timestamp_col)
        per_symbol.append(_simulate_symbol(symbol, frame, cfg))

    all_trades: list[dict[str, Any]] = []
    total_bars = 0
    total_position_bars = 0.0
    for item in per_symbol:
        all_trades.extend(item["trades_detail"])
        total_bars += int(item["bars"])
        total_position_bars += float(item["exposure_ratio"]) * float(item["bars"])

    pnl = np.asarray([float(t["pnl_bps"]) for t in all_trades], dtype=float)
    holds = np.asarray([float(t["hold_bars"]) for t in all_trades], dtype=float)
    wins = pnl[pnl > 0.0]
    losses = pnl[pnl < 0.0]
    trade_count = int(pnl.size)
    aggregate: dict[str, Any] = {
        "symbols": len(per_symbol),
        "total_bars": total_bars,
        "total_trades": trade_count,
        "win_rate": float((wins.size / trade_count) if trade_count else 0.0),
        "avg_win_bps": float(wins.mean()) if wins.size else 0.0,
        "avg_loss_bps": float(abs(losses.mean())) if losses.size else 0.0,
        "profit_factor": _profit_factor(wins, losses),
        "expectancy_bps": float(pnl.mean()) if trade_count else 0.0,
        "net_pnl_bps": float(pnl.sum()) if trade_count else 0.0,
        "median_hold_bars": float(np.median(holds)) if holds.size else 0.0,
        "churn_trades_per_100_bars": float((trade_count / max(total_bars, 1)) * 100.0),
        "exposure_ratio": float(total_position_bars / max(total_bars, 1)),
        "config": {
            "confidence_threshold": cfg.confidence_threshold,
            "entry_score_threshold": cfg.entry_score_threshold,
            "allow_shorts": cfg.allow_shorts,
            "min_hold_bars": cfg.min_hold_bars,
            "max_hold_bars": cfg.max_hold_bars,
            "stop_loss_bps": cfg.stop_loss_bps,
            "take_profit_bps": cfg.take_profit_bps,
            "trailing_stop_bps": cfg.trailing_stop_bps,
            "fee_bps": cfg.fee_bps,
            "slippage_bps": cfg.slippage_bps,
        },
    }
    return {"aggregate": aggregate, "symbols": per_symbol}


def _load_runtime_env_for_replay(args: argparse.Namespace) -> None:
    env_file = args.env_file
    if env_file is None:
        # Preserve explicit process env overrides for replay experiments while still
        # loading defaults from .env when present.
        reload_env(path=None, override=False)
        return
    reload_env(path=env_file, override=False)


def run_replay(argv: list[str] | None = None) -> dict[str, Any]:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _load_runtime_env_for_replay(args)
    return _run_replay(args)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _load_runtime_env_for_replay(args)

    try:
        payload = _run_replay(args)
    except Exception as exc:
        logger.error("OFFLINE_REPLAY_FAILED", extra={"error": str(exc)}, exc_info=True)
        return 1

    aggregate = payload["aggregate"]
    logger.info(
        "OFFLINE_REPLAY_COMPLETE",
        extra={
            "symbols": aggregate["symbols"],
            "bars": aggregate["total_bars"],
            "trades": aggregate["total_trades"],
            "win_rate": aggregate["win_rate"],
            "profit_factor": aggregate["profit_factor"],
            "expectancy_bps": aggregate["expectancy_bps"],
            "median_hold_bars": aggregate["median_hold_bars"],
            "churn_100_bars": aggregate["churn_trades_per_100_bars"],
        },
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        logger.info("OFFLINE_REPLAY_JSON_WRITTEN", extra={"path": str(args.output_json)})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
