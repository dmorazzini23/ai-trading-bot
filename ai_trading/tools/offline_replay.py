from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

import argparse
import csv
from collections import Counter, deque
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, cast
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ai_trading.config.management import get_env, reload_env
from ai_trading.data.historical_bars import HistoricalBarLoadReport, load_historical_bars
from ai_trading.features.indicators import (
    compute_atr,
    compute_macd,
    compute_macds,
    compute_sma,
    compute_vwap,
)
from ai_trading.indicators import rsi as rsi_indicator
from ai_trading.logging import get_logger
from ai_trading.models.artifacts import load_verified_joblib_artifact
from ai_trading.replay.event_loop import ReplayEventLoop
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path

logger = get_logger(__name__)

OFFLINE_REPLAY_SCHEMA_VERSION = "1.0.0"


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
    live_cost_model: LiveCostReplayModel | None = None
    regime_thresholds: RegimeThresholdModel | None = None
    sizing_policy: str = "flat"
    sizing_min_scale: float = 1.0
    sizing_max_scale: float = 1.0
    sizing_cost_penalty_bps: float = 25.0


@dataclass(frozen=True)
class LiveCostReplayBucket:
    symbol: str
    side: str
    session_regime: str
    slippage_bps: float
    sample_count: int
    source_metric: str


@dataclass(frozen=True)
class LiveCostReplayModel:
    path: str
    generated_at: str | None
    status: str
    bucket_count: int
    buckets: dict[tuple[str, str, str], LiveCostReplayBucket]


@dataclass(frozen=True)
class RegimeThreshold:
    regime: str
    confidence_threshold: float
    entry_score_threshold: float
    source: str
    sample_count: int = 0


@dataclass(frozen=True)
class RegimeThresholdModel:
    path: str
    generated_at: str | None
    status: str
    regimes: dict[str, RegimeThreshold]


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
    supports_short_scores: bool = False


@dataclass(frozen=True)
class MarkoutVetoConfig:
    mode: str
    lookback: int
    min_samples: int
    min_mean_bps: float
    max_wrong_way_rate: float


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
        "opportunity_openings_only_skipped": int(
            counters.get("opportunity_openings_only_skipped", 0)
        ),
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
        model = load_verified_joblib_artifact(model_path)
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
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
    label_sides_raw = getattr(model, "replay_label_sides_", ())
    try:
        label_sides = {str(item).strip().lower() for item in label_sides_raw}
    except TypeError:
        label_sides = set()
    semantics = str(getattr(model, "edge_score_semantics_", "") or "").strip().lower()
    supports_short_scores = bool(getattr(model, "supports_short_scores_", False)) or bool(
        label_sides.intersection({"sell", "short", "sell_short"})
    ) or semantics in {"directional", "long_short_probability", "signed_edge"}
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
        supports_short_scores=supports_short_scores,
    )
    logger.info(
        "OFFLINE_REPLAY_MODEL_SCORING_ENABLED",
        extra={
            "model_path": context.model_path,
            "feature_count": len(context.feature_names),
            "orientation": "inverse" if context.orientation_inverse else "direct",
            "symbol_penalty_count": len(context.symbol_penalties),
            "supports_short_scores": bool(context.supports_short_scores),
        },
    )
    return context


def _safe_rsi(close_values: np.ndarray) -> np.ndarray:
    if close_values.size <= 0:
        return cast(np.ndarray, np.asarray([], dtype=float))
    try:
        out = rsi_indicator(tuple(close_values.tolist()), 14)
        arr = np.asarray(out, dtype=float)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
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


def _sanitize_model_feature_index(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    """Normalize index labels before indicator computation to avoid duplicate-label failures."""

    frame = frame.sort_index(kind="stable")
    duplicate_labels = int(frame.index.duplicated(keep=False).sum())
    null_labels = int(pd.isna(frame.index).sum())
    if duplicate_labels > 0 or null_labels > 0:
        logger.info(
            "OFFLINE_REPLAY_MODEL_FEATURE_INDEX_SANITIZED",
            extra={
                "symbol": symbol,
                "rows": int(len(frame)),
                "duplicate_labels": duplicate_labels,
                "null_labels": null_labels,
            },
        )
        frame = frame.reset_index(drop=True)
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
        "--sizing-policy",
        choices=("flat", "confidence", "confidence-cost"),
        default="flat",
        help="Replay position sizing policy. Defaults to flat sizing.",
    )
    parser.add_argument("--sizing-min-scale", type=float, default=1.0)
    parser.add_argument("--sizing-max-scale", type=float, default=1.0)
    parser.add_argument(
        "--sizing-cost-penalty-bps",
        type=float,
        default=25.0,
        help="Cost level where confidence-cost sizing reaches minimum scale.",
    )
    parser.add_argument(
        "--live-cost-model-json",
        type=Path,
        default=None,
        help=(
            "Optional live cost model artifact to use for non-simulation replay "
            "slippage assumptions."
        ),
    )
    parser.add_argument(
        "--use-live-cost-model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use AI_TRADING_LIVE_COST_MODEL_PATH for non-simulation replay when "
            "--live-cost-model-json is omitted."
        ),
    )
    parser.add_argument(
        "--regime-thresholds-json",
        type=Path,
        default=None,
        help="Optional regime threshold artifact for replay entry gates.",
    )
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
    parser.add_argument(
        "--export-accepted-candidates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write accepted replay candidate component rows as CSV and JSONL.",
    )
    parser.add_argument(
        "--accepted-candidates-dir",
        type=Path,
        default=None,
        help="Directory for accepted_candidates.csv/jsonl; defaults beside --output-json.",
    )
    parser.add_argument(
        "--markout-veto-mode",
        choices=("off", "shadow", "enforce"),
        default="off",
        help="Replay-only markout veto mode. Shadow records reasons; enforce rejects candidates.",
    )
    parser.add_argument("--markout-veto-lookback", type=int, default=20)
    parser.add_argument("--markout-veto-min-samples", type=int, default=5)
    parser.add_argument("--markout-veto-min-mean-bps", type=float, default=0.0)
    parser.add_argument("--markout-veto-max-wrong-way-rate", type=float, default=0.60)
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


def _input_report_payload(
    reports: Mapping[str, HistoricalBarLoadReport],
) -> dict[str, dict[str, Any]]:
    return {
        symbol: report.as_dict()
        for symbol, report in sorted(reports.items())
    }


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
    frame = _sanitize_model_feature_index(df.copy(), symbol=symbol)
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
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    probs: np.ndarray = np.full(len(frame), 0.5, dtype=float)
    try:
        arr = np.asarray(
            model_context.model.predict_proba(feature_frame.astype(float)),
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
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
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


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _parse_utc_datetime_text(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
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


def _replay_session_regime(ts: Any) -> str:
    if isinstance(ts, pd.Timestamp):
        dt = ts.to_pydatetime()
    elif isinstance(ts, datetime):
        dt = ts
    else:
        dt = _parse_utc_datetime_text(str(ts))
        if dt is None:
            return "unknown"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    ts_et = dt.astimezone(ZoneInfo("America/New_York"))
    minute = (ts_et.hour * 60) + ts_et.minute
    if ts_et.weekday() >= 5 or minute < (9 * 60 + 30) or minute >= (16 * 60):
        return "offhours"
    if minute < (10 * 60 + 15):
        return "opening"
    if minute >= (15 * 60 + 15):
        return "closing"
    return "midday"


def _normalize_live_cost_side(side: str) -> str:
    token = str(side or "").strip().lower()
    if token in {"long", "buy", "cover", "buy_to_cover"}:
        return "buy"
    if token in {"short", "sell_short", "sellshort"}:
        return "sell_short"
    if token in {"sell", "sell_long"}:
        return "sell"
    return token or "unknown"


def _live_cost_slippage_from_row(row: Mapping[str, Any]) -> tuple[float | None, str | None]:
    for key in ("p90_adverse_slippage_bps", "mean_slippage_bps"):
        value = _finite_float(row.get(key))
        if value is not None and value >= 0.0:
            return float(value), key
    return None, None


def _load_live_cost_replay_model(args: argparse.Namespace) -> LiveCostReplayModel | None:
    explicit_path = getattr(args, "live_cost_model_json", None)
    use_live_cost_model = getattr(args, "use_live_cost_model", None)
    if explicit_path is None and use_live_cost_model is None:
        use_live_cost_model = bool(
            get_env("AI_TRADING_OFFLINE_REPLAY_USE_LIVE_COST_MODEL", False, cast=bool)
        )
    if explicit_path is None and not bool(use_live_cost_model):
        return None
    path = (
        Path(explicit_path).expanduser()
        if explicit_path is not None
        else resolve_runtime_artifact_path(
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
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping) or payload.get("artifact_type") != "live_cost_model":
        return None
    status = payload.get("status")
    if not isinstance(status, Mapping) or not bool(status.get("available")):
        return None
    if str(status.get("status") or "").lower() != "ready":
        return None
    rows = payload.get("by_symbol_side_session")
    if not isinstance(rows, list):
        return None
    buckets: dict[tuple[str, str, str], LiveCostReplayBucket] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not bool(row.get("sufficient_samples")):
            continue
        slippage_bps, source_metric = _live_cost_slippage_from_row(row)
        if slippage_bps is None or source_metric is None:
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        side = _normalize_live_cost_side(str(row.get("side") or ""))
        session_regime = str(row.get("session_regime") or "").strip().lower()
        if not symbol or not side or not session_regime:
            continue
        sample_count = int(_finite_float(row.get("sample_count")) or 0.0)
        buckets[(symbol, side, session_regime)] = LiveCostReplayBucket(
            symbol=symbol,
            side=side,
            session_regime=session_regime,
            slippage_bps=float(slippage_bps),
            sample_count=sample_count,
            source_metric=source_metric,
        )
    if not buckets:
        return None
    return LiveCostReplayModel(
        path=str(path),
        generated_at=(
            str(payload.get("generated_at")) if payload.get("generated_at") else None
        ),
        status=str(status.get("status") or "ready"),
        bucket_count=len(buckets),
        buckets=buckets,
    )


def _threshold_rows_from_payload(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = payload.get("regimes")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, Mapping)]
    mapping = payload.get("thresholds_by_regime")
    if isinstance(mapping, Mapping):
        out: list[Mapping[str, Any]] = []
        for regime, row in mapping.items():
            if isinstance(row, Mapping):
                item = dict(row)
                item.setdefault("regime", regime)
                out.append(item)
        return out
    return []


def _load_regime_threshold_model(args: argparse.Namespace) -> RegimeThresholdModel | None:
    explicit_path = str(getattr(args, "regime_thresholds_json", "") or "").strip()
    if not explicit_path:
        return None
    path = Path(explicit_path).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    status_raw = payload.get("status")
    status = status_raw if isinstance(status_raw, Mapping) else {}
    if status and not bool(status.get("available", True)):
        return None
    regimes: dict[str, RegimeThreshold] = {}
    for row in _threshold_rows_from_payload(payload):
        regime = str(
            row.get("regime") or row.get("session_regime") or row.get("bucket") or ""
        ).strip().lower()
        if not regime:
            continue
        confidence_threshold = _finite_float(row.get("confidence_threshold"))
        entry_score_threshold = _finite_float(row.get("entry_score_threshold"))
        if confidence_threshold is None or entry_score_threshold is None:
            continue
        regimes[regime] = RegimeThreshold(
            regime=regime,
            confidence_threshold=float(_clamp(confidence_threshold, 0.0, 0.999999)),
            entry_score_threshold=float(_clamp(entry_score_threshold, 0.0, 0.999999)),
            source=str(row.get("source") or f"regime_threshold:{regime}"),
            sample_count=int(_finite_float(row.get("sample_count")) or 0.0),
        )
    if not regimes:
        return None
    return RegimeThresholdModel(
        path=str(path),
        generated_at=str(payload.get("generated_at")) if payload.get("generated_at") else None,
        status=str(status.get("status") or payload.get("status") or "ready"),
        regimes=regimes,
    )


def _thresholds_for_regime(
    cfg: ReplayConfig,
    *,
    ts: Any,
) -> tuple[float, float, str, str]:
    regime = _replay_session_regime(ts)
    model = cfg.regime_thresholds
    if model is not None:
        threshold = model.regimes.get(regime) or model.regimes.get("default")
        if threshold is not None:
            return (
                float(threshold.confidence_threshold),
                float(threshold.entry_score_threshold),
                regime,
                threshold.source,
            )
    return (
        float(cfg.confidence_threshold),
        float(cfg.entry_score_threshold),
        regime,
        "global",
    )


def _replay_slippage_bps(
    cfg: ReplayConfig,
    *,
    symbol: str,
    side: str,
    ts: Any,
) -> float:
    model = cfg.live_cost_model
    if model is None:
        return float(cfg.slippage_bps)
    symbol_token = str(symbol or "").strip().upper()
    side_token = _normalize_live_cost_side(side)
    session_regime = _replay_session_regime(ts)
    candidates = [
        (symbol_token, side_token, session_regime),
    ]
    if side_token == "sell_short":
        candidates.append((symbol_token, "sell", session_regime))
    elif side_token == "sell":
        candidates.append((symbol_token, "sell_short", session_regime))
    for key in candidates:
        bucket = model.buckets.get(key)
        if bucket is not None:
            return float(bucket.slippage_bps)
    return float(cfg.slippage_bps)


def _sizing_multiplier(
    cfg: ReplayConfig,
    *,
    confidence: float,
    score: float,
    slippage_bps: float,
) -> tuple[float, dict[str, Any]]:
    policy = str(cfg.sizing_policy or "flat").strip().lower()
    min_scale = _clamp(float(cfg.sizing_min_scale), 0.0, 10.0)
    max_scale = _clamp(float(cfg.sizing_max_scale), 0.0, 10.0)
    if max_scale < min_scale:
        max_scale = min_scale
    context: dict[str, Any] = {
        "policy": policy,
        "confidence": float(confidence),
        "score": float(score),
        "min_scale": float(min_scale),
        "max_scale": float(max_scale),
    }
    if policy == "flat":
        context["reason"] = "flat"
        return 1.0, context

    threshold = _clamp(float(cfg.confidence_threshold), 0.0, 0.999999)
    confidence_edge = _clamp(
        (float(confidence) - threshold) / max(1.0 - threshold, 1e-9),
        0.0,
        1.0,
    )
    score_edge = _clamp(abs(float(score)), 0.0, 1.0)
    edge_scale = max(confidence_edge, score_edge)
    multiplier = min_scale + ((max_scale - min_scale) * edge_scale)
    context["confidence_edge"] = float(confidence_edge)
    context["score_edge"] = float(score_edge)

    if policy == "confidence-cost":
        penalty_bps = max(0.0, float(cfg.sizing_cost_penalty_bps))
        cost_ratio = (
            _clamp(float(slippage_bps) / penalty_bps, 0.0, 1.0)
            if penalty_bps > 0.0
            else 0.0
        )
        cost_multiplier = 1.0 - cost_ratio
        multiplier = min_scale + ((multiplier - min_scale) * cost_multiplier)
        context["slippage_bps"] = float(slippage_bps)
        context["cost_penalty_bps"] = float(penalty_bps)
        context["cost_multiplier"] = float(cost_multiplier)

    bounded = _clamp(multiplier, min_scale, max_scale)
    context["multiplier"] = float(bounded)
    return float(bounded), context


def _live_cost_model_config_payload(model: LiveCostReplayModel | None) -> dict[str, Any]:
    if model is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "path": model.path,
        "generated_at": model.generated_at,
        "status": model.status,
        "bucket_count": model.bucket_count,
    }


def _regime_threshold_config_payload(model: RegimeThresholdModel | None) -> dict[str, Any]:
    if model is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "path": model.path,
        "generated_at": model.generated_at,
        "status": model.status,
        "regime_count": int(len(model.regimes)),
        "regimes": {
            key: {
                "confidence_threshold": value.confidence_threshold,
                "entry_score_threshold": value.entry_score_threshold,
                "source": value.source,
                "sample_count": value.sample_count,
            }
            for key, value in sorted(model.regimes.items())
        },
    }


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


def _summarize_trades_by_regime(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    for trade in trades:
        regime = str(trade.get("session_regime") or "unknown").strip().lower() or "unknown"
        pnl_value = _finite_float(trade.get("pnl_bps"))
        if pnl_value is None:
            continue
        grouped.setdefault(regime, []).append(float(pnl_value))
    rows: list[dict[str, Any]] = []
    for regime, values in sorted(grouped.items()):
        pnl = np.asarray(values, dtype=float)
        wins = pnl[pnl > 0.0]
        losses = pnl[pnl < 0.0]
        rows.append(
            {
                "session_regime": regime,
                "trades": int(pnl.size),
                "win_rate": float((wins.size / pnl.size) if pnl.size else 0.0),
                "expectancy_bps": float(pnl.mean()) if pnl.size else 0.0,
                "net_pnl_bps": float(pnl.sum()) if pnl.size else 0.0,
                "profit_factor": _profit_factor(wins, losses),
            }
        )
    rows.sort(key=lambda row: float(row.get("expectancy_bps") or 0.0), reverse=True)
    return rows


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
    entry_size_multiplier = 1.0
    entry_sizing_context: dict[str, Any] = {"policy": "flat", "reason": "flat"}
    entry_session_regime = "unknown"
    entry_threshold_source = "global"
    entry_confidence_threshold = float(cfg.confidence_threshold)
    entry_entry_score_threshold = float(cfg.entry_score_threshold)
    position_bars = 0
    equity = 1.0
    equity_curve: list[float] = [equity]

    for i, (ts, row) in enumerate(df.iterrows()):
        close = float(row["close"])
        if close <= 0.0:
            continue
        s = float(score.iloc[i])
        conf = float(confidence.iloc[i])
        effective_confidence_threshold, effective_entry_threshold, session_regime, threshold_source = (
            _thresholds_for_regime(cfg, ts=ts)
        )

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
                long_flip = side > 0 and s <= -effective_entry_threshold
                short_flip = side < 0 and s >= effective_entry_threshold
                if (long_flip or short_flip) and conf >= effective_confidence_threshold:
                    exit_reason = "signal_flip"

            if exit_reason is not None:
                exit_side = "sell" if side > 0 else "buy"
                exit_slippage_bps = _replay_slippage_bps(
                    cfg,
                    symbol=symbol,
                    side=exit_side,
                    ts=ts,
                )
                fill_exit = _exit_price(close, side, exit_slippage_bps)
                raw_pnl_bps = ((fill_exit / entry_price) - 1.0) * 10000.0 * side
                raw_pnl_bps -= 2.0 * cfg.fee_bps
                pnl_bps = raw_pnl_bps * float(entry_size_multiplier)
                equity *= 1.0 + (pnl_bps / 10000.0)
                trades.append(
                    {
                        "symbol": symbol,
                        "entry_ts": entry_ts,
                        "exit_ts": str(ts),
                        "side": "long" if side > 0 else "short",
                        "hold_bars": hold_bars,
                        "pnl_bps": float(pnl_bps),
                        "raw_pnl_bps": float(raw_pnl_bps),
                        "session_regime": str(entry_session_regime),
                        "threshold_source": str(entry_threshold_source),
                        "confidence_threshold": float(entry_confidence_threshold),
                        "entry_score_threshold": float(entry_entry_score_threshold),
                        "size_multiplier": float(entry_size_multiplier),
                        "sizing": dict(entry_sizing_context),
                        "exit_reason": exit_reason,
                    }
                )
                side = 0
                entry_price = 0.0
                entry_bar = -1
                entry_ts = None
                best_price = 0.0
                entry_size_multiplier = 1.0
                entry_sizing_context = {"policy": "flat", "reason": "flat"}
                entry_session_regime = "unknown"
                entry_threshold_source = "global"
                entry_confidence_threshold = float(cfg.confidence_threshold)
                entry_entry_score_threshold = float(cfg.entry_score_threshold)

        if side == 0:
            open_long = conf >= effective_confidence_threshold and s >= effective_entry_threshold
            open_short = (
                cfg.allow_shorts
                and conf >= effective_confidence_threshold
                and s <= -effective_entry_threshold
            )
            if open_long:
                side = 1
            elif open_short:
                side = -1

            if side != 0:
                entry_side = "buy" if side > 0 else "sell_short"
                entry_slippage_bps = _replay_slippage_bps(
                    cfg,
                    symbol=symbol,
                    side=entry_side,
                    ts=ts,
                )
                entry_size_multiplier, entry_sizing_context = _sizing_multiplier(
                    cfg,
                    confidence=conf,
                    score=s,
                    slippage_bps=entry_slippage_bps,
                )
                entry_price = _entry_price(close, side, entry_slippage_bps)
                entry_bar = i
                entry_ts = str(ts)
                best_price = close
                entry_session_regime = session_regime
                entry_threshold_source = threshold_source
                entry_confidence_threshold = effective_confidence_threshold
                entry_entry_score_threshold = effective_entry_threshold

        equity_curve.append(equity)

    if side != 0 and entry_bar >= 0:
        close = float(df["close"].iloc[-1])
        exit_side = "sell" if side > 0 else "buy"
        fill_exit = _exit_price(
            close,
            side,
            _replay_slippage_bps(
                cfg,
                symbol=symbol,
                side=exit_side,
                ts=df.index[-1],
            ),
        )
        raw_pnl_bps = ((fill_exit / entry_price) - 1.0) * 10000.0 * side
        raw_pnl_bps -= 2.0 * cfg.fee_bps
        pnl_bps = raw_pnl_bps * float(entry_size_multiplier)
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
                "raw_pnl_bps": float(raw_pnl_bps),
                "session_regime": str(entry_session_regime),
                "threshold_source": str(entry_threshold_source),
                "confidence_threshold": float(entry_confidence_threshold),
                "entry_score_threshold": float(entry_entry_score_threshold),
                "size_multiplier": float(entry_size_multiplier),
                "sizing": dict(entry_sizing_context),
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


def _resolve_markout_veto_config(args: argparse.Namespace) -> MarkoutVetoConfig | None:
    mode = str(getattr(args, "markout_veto_mode", "off") or "off").strip().lower()
    if mode in {"", "off", "none", "disabled"}:
        return None
    if mode not in {"shadow", "enforce"}:
        raise ValueError("--markout-veto-mode must be off, shadow, or enforce")
    lookback = max(1, int(getattr(args, "markout_veto_lookback", 20) or 20))
    min_samples = max(1, int(getattr(args, "markout_veto_min_samples", 5) or 5))
    min_mean_raw = getattr(args, "markout_veto_min_mean_bps", 0.0)
    wrong_way_raw = getattr(args, "markout_veto_max_wrong_way_rate", 0.60)
    min_mean_bps = 0.0 if min_mean_raw is None else float(min_mean_raw)
    max_wrong_way_rate = 0.60 if wrong_way_raw is None else float(wrong_way_raw)
    return MarkoutVetoConfig(
        mode=mode,
        lookback=lookback,
        min_samples=min(min_samples, lookback),
        min_mean_bps=min_mean_bps,
        max_wrong_way_rate=_clamp(
            max_wrong_way_rate,
            0.0,
            1.0,
        ),
    )


def _candidate_markout_bps(
    *,
    side: str,
    submit_price: float,
    markout_price: float | None,
) -> float | None:
    if markout_price is None or submit_price <= 0.0 or markout_price <= 0.0:
        return None
    side_sign = 1.0 if str(side).strip().lower() == "buy" else -1.0
    return float(((float(markout_price) / float(submit_price)) - 1.0) * 10000.0 * side_sign)


def _markout_veto_reason(
    history: deque[float],
    *,
    config: MarkoutVetoConfig,
) -> str | None:
    samples = [float(value) for value in history if math.isfinite(float(value))]
    if len(samples) < int(config.min_samples):
        return None
    mean_bps = float(sum(samples) / len(samples))
    wrong_way_rate = float(sum(1 for value in samples if value <= 0.0) / len(samples))
    reasons: list[str] = []
    if mean_bps < float(config.min_mean_bps):
        reasons.append("mean_markout_below_floor")
    if wrong_way_rate > float(config.max_wrong_way_rate):
        reasons.append("wrong_way_rate_above_limit")
    return ";".join(reasons) if reasons else None


def _accepted_candidate_row(
    *,
    context: Mapping[str, Any],
    fill_event: Mapping[str, Any] | None,
    cfg: ReplayConfig,
) -> dict[str, Any]:
    side = str(context.get("side", "")).strip().lower()
    submit_price = float(context.get("submit_price", 0.0) or 0.0)
    markout_raw = context.get("markout_price")
    markout_price: float | None = None
    if markout_raw not in (None, ""):
        try:
            parsed = float(markout_raw)
        except (TypeError, ValueError):
            parsed = 0.0
        if parsed > 0.0 and math.isfinite(parsed):
            markout_price = float(parsed)
    gross_markout_bps = _candidate_markout_bps(
        side=side,
        submit_price=submit_price,
        markout_price=markout_price,
    )
    net_markout_bps = (
        float(gross_markout_bps) - (2.0 * max(0.0, float(cfg.fee_bps)))
        if gross_markout_bps is not None
        else None
    )
    row = {
        "ts": context.get("submit_ts"),
        "symbol": context.get("symbol"),
        "side": side,
        "score": context.get("score"),
        "confidence": context.get("confidence"),
        "session_regime": context.get("session_regime"),
        "threshold_source": context.get("threshold_source"),
        "entry_score_threshold": context.get("entry_score_threshold"),
        "confidence_threshold": context.get("confidence_threshold"),
        "score_source": context.get("score_source"),
        "expected_capture_bps": context.get("policy_expected_capture_proxy_bps"),
        "expected_capture_adjusted_bps": context.get("policy_expected_capture_adjusted_bps"),
        "fill_probability": context.get("policy_fill_prob_proxy"),
        "replay_adjustment_bps": context.get("policy_replay_adjustment_bps"),
        "bandit_adjustment_bps": context.get("policy_bandit_adjustment_bps"),
        "rank_score_baseline": context.get("rank_score_baseline"),
        "rank_score_post_adjustments": context.get("rank_score_post_adjustments"),
        "rank_index": context.get("policy_rank_index"),
        "group_size": context.get("policy_group_size"),
        "submit_price": submit_price,
        "markout_price": markout_price,
        "markout_bps": gross_markout_bps,
        "net_markout_bps": net_markout_bps,
        "direction_correct": (
            None if gross_markout_bps is None else bool(gross_markout_bps > 0.0)
        ),
        "accepted_reason": context.get("accepted_reason", "accepted"),
        "veto_shadow_reason": context.get("veto_shadow_reason"),
        "veto_bucket": context.get("veto_bucket"),
        "filled": bool(fill_event is not None),
        "fill_price": None,
        "fill_qty": None,
        "client_order_id": context.get("client_order_id"),
        "size_multiplier": context.get("size_multiplier"),
        "sizing_policy": (
            context.get("sizing", {}).get("policy")
            if isinstance(context.get("sizing"), Mapping)
            else None
        ),
    }
    if fill_event is not None:
        row["fill_price"] = fill_event.get("fill_price")
        row["fill_qty"] = fill_event.get("fill_qty")
    return row


def _candidate_quality_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    markouts: list[float] = []
    net_markouts: list[float] = []
    direction_correct: list[bool] = []
    filled_count = 0
    for row in rows:
        if bool(row.get("filled")):
            filled_count += 1
        raw_markout = row.get("markout_bps")
        raw_net_markout = row.get("net_markout_bps")
        try:
            markout = float(raw_markout)
        except (TypeError, ValueError):
            markout = math.nan
        try:
            net_markout = float(raw_net_markout)
        except (TypeError, ValueError):
            net_markout = math.nan
        if math.isfinite(markout):
            markouts.append(markout)
        if math.isfinite(net_markout):
            net_markouts.append(net_markout)
        raw_direction = row.get("direction_correct")
        if isinstance(raw_direction, bool):
            direction_correct.append(raw_direction)
    sample_count = len(markouts)
    positive_count = sum(1 for value in markouts if value > 0.0)
    wrong_way_count = sum(1 for value in direction_correct if not value)
    return {
        "candidates": int(len(rows)),
        "filled": int(filled_count),
        "markout_samples": int(sample_count),
        "total_net_markout_bps": (
            float(sum(net_markouts)) if net_markouts else None
        ),
        "mean_markout_bps": (
            float(sum(markouts) / sample_count) if sample_count else None
        ),
        "mean_net_markout_bps": (
            float(sum(net_markouts) / len(net_markouts)) if net_markouts else None
        ),
        "positive_rate": float(positive_count / sample_count) if sample_count else None,
        "wrong_way_rate": (
            float(wrong_way_count / len(direction_correct))
            if direction_correct
            else None
        ),
    }


def _finite_row_float(row: Mapping[str, Any], key: str) -> float | None:
    try:
        value = float(row.get(key))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _quantile_edges(values: list[float], buckets: int = 4) -> list[float]:
    if not values:
        return []
    ordered = sorted(values)
    edges: list[float] = []
    for index in range(1, int(buckets)):
        pos = int(round((len(ordered) - 1) * (index / float(buckets))))
        edges.append(float(ordered[pos]))
    return sorted(set(edges))


def _bucket_label(value: float | None, edges: list[float]) -> str:
    if value is None:
        return "missing"
    lower: float | None = None
    for edge in edges:
        if value <= edge:
            return f"({lower if lower is not None else '-inf'}, {edge}]"
        lower = edge
    return f"({lower if lower is not None else '-inf'}, inf)"


def _rank_bucket(row: Mapping[str, Any]) -> str:
    rank = _finite_row_float(row, "rank_index")
    if rank is None:
        return "missing"
    rank_int = int(rank)
    if rank_int <= 0:
        return "0"
    if rank_int <= 2:
        return "1-2"
    if rank_int <= 5:
        return "3-5"
    if rank_int <= 10:
        return "6-10"
    return "11+"


def _candidate_hour(row: Mapping[str, Any]) -> int | None:
    raw_ts = row.get("ts")
    if raw_ts in (None, ""):
        return None
    try:
        parsed = pd.to_datetime(raw_ts, errors="coerce", utc=True)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return int(parsed.hour)


def _session_segment(row: Mapping[str, Any]) -> str:
    hour = _candidate_hour(row)
    if hour is None:
        return "missing"
    if hour < 13:
        return "pre_regular"
    if hour < 15:
        return "open"
    if hour < 18:
        return "midday"
    if hour < 20:
        return "late"
    return "post_regular"


def _group_quality(
    rows: list[dict[str, Any]],
    *,
    key_name: str,
    key_func: Any,
    min_candidates: int = 1,
    sort_by: str = "mean_net_markout_bps",
    reverse: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(key_func(row))
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, group_rows in grouped.items():
        if len(group_rows) < int(min_candidates):
            continue
        summary = _candidate_quality_summary(group_rows)
        summary[key_name] = key
        out.append(summary)

    def sort_value(item: Mapping[str, Any]) -> tuple[bool, float]:
        raw = item.get(sort_by)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return (True, 0.0)
        if not math.isfinite(value):
            return (True, 0.0)
        return (False, value)

    out.sort(key=sort_value, reverse=reverse)
    return out[:limit] if limit is not None else out


def _summarize_cap_adjustments(
    cap_adjustments: list[Mapping[str, Any]],
    *,
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_count_by_symbol = Counter(
        str(row.get("symbol") or "").strip().upper()
        for row in candidate_rows
        if str(row.get("symbol") or "").strip()
    )
    by_symbol: Counter[str] = Counter()
    by_side: Counter[str] = Counter()
    by_cap_kind: Counter[str] = Counter()
    for adjustment in cap_adjustments:
        symbol = str(adjustment.get("symbol") or "").strip().upper()
        side = str(adjustment.get("side") or "").strip().lower()
        cap_kind = str(adjustment.get("cap_kind") or "unknown").strip().lower()
        if symbol:
            by_symbol[symbol] += 1
        if side:
            by_side[side] += 1
        by_cap_kind[cap_kind or "unknown"] += 1
    by_symbol_rows: list[dict[str, Any]] = []
    for symbol, count in by_symbol.most_common():
        candidates = int(candidate_count_by_symbol.get(symbol, 0))
        by_symbol_rows.append(
            {
                "symbol": symbol,
                "adjustments": int(count),
                "candidate_count": candidates,
                "adjustments_per_candidate": (
                    float(count / candidates) if candidates else None
                ),
            }
        )
    return {
        "total": int(len(cap_adjustments)),
        "by_cap_kind": dict(sorted(by_cap_kind.items())),
        "by_side": dict(sorted(by_side.items())),
        "top_symbols": by_symbol_rows[:20],
    }


def _summarize_candidate_quality(
    rows: list[dict[str, Any]],
    *,
    cap_adjustments: list[Mapping[str, Any]],
) -> dict[str, Any]:
    score_edges = _quantile_edges(
        [
            value
            for row in rows
            if (value := _finite_row_float(row, "score")) is not None
        ]
    )
    confidence_edges = _quantile_edges(
        [
            value
            for row in rows
            if (value := _finite_row_float(row, "confidence")) is not None
        ]
    )
    return {
        "overall": _candidate_quality_summary(rows),
        "by_side": _group_quality(
            rows,
            key_name="side",
            key_func=lambda row: str(row.get("side") or "missing").lower(),
        ),
        "worst_symbols": _group_quality(
            rows,
            key_name="symbol",
            key_func=lambda row: str(row.get("symbol") or "missing").upper(),
            min_candidates=20,
            limit=20,
        ),
        "best_symbols": _group_quality(
            rows,
            key_name="symbol",
            key_func=lambda row: str(row.get("symbol") or "missing").upper(),
            min_candidates=20,
            reverse=True,
            limit=20,
        ),
        "by_score_bucket": _group_quality(
            rows,
            key_name="score_bucket",
            key_func=lambda row: _bucket_label(
                _finite_row_float(row, "score"),
                score_edges,
            ),
        ),
        "by_confidence_bucket": _group_quality(
            rows,
            key_name="confidence_bucket",
            key_func=lambda row: _bucket_label(
                _finite_row_float(row, "confidence"),
                confidence_edges,
            ),
        ),
        "by_rank_bucket": _group_quality(
            rows,
            key_name="rank_bucket",
            key_func=_rank_bucket,
        ),
        "by_utc_hour": _group_quality(
            rows,
            key_name="utc_hour",
            key_func=lambda row: (
                str(hour) if (hour := _candidate_hour(row)) is not None else "missing"
            ),
        ),
        "by_session_segment": _group_quality(
            rows,
            key_name="session_segment",
            key_func=_session_segment,
        ),
        "by_session_regime": _group_quality(
            rows,
            key_name="session_regime",
            key_func=lambda row: str(row.get("session_regime") or "missing").lower(),
        ),
        "by_threshold_source": _group_quality(
            rows,
            key_name="threshold_source",
            key_func=lambda row: str(row.get("threshold_source") or "global").lower(),
        ),
        "cap_adjustments": _summarize_cap_adjustments(
            cap_adjustments,
            candidate_rows=rows,
        ),
    }


def _summarize_markout_veto_shadow_quality(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    flagged = [
        row for row in rows if str(row.get("veto_shadow_reason") or "").strip()
    ]
    unflagged = [
        row for row in rows if not str(row.get("veto_shadow_reason") or "").strip()
    ]
    flagged_summary = _candidate_quality_summary(flagged)
    unflagged_summary = _candidate_quality_summary(unflagged)
    flagged_net = flagged_summary.get("mean_net_markout_bps")
    unflagged_net = unflagged_summary.get("mean_net_markout_bps")
    net_edge_vs_unflagged: float | None = None
    enforcement_supported = False
    if flagged_net is not None and unflagged_net is not None:
        net_edge_vs_unflagged = float(flagged_net) - float(unflagged_net)
        enforcement_supported = bool(net_edge_vs_unflagged < 0.0)
    return {
        "flagged": flagged_summary,
        "unflagged": unflagged_summary,
        "flagged_net_edge_bps_vs_unflagged": net_edge_vs_unflagged,
        "enforcement_supported": bool(enforcement_supported),
        "recommendation": "enforce_candidate" if enforcement_supported else "shadow_only",
    }


def _write_accepted_candidate_artifacts(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
) -> dict[str, str]:
    base_dir_raw = getattr(args, "accepted_candidates_dir", None)
    if base_dir_raw is not None:
        base_dir = Path(base_dir_raw)
    elif getattr(args, "output_json", None) is not None:
        base_dir = Path(args.output_json).parent
    else:
        base_dir = Path("artifacts") / "offline_replay_accepted_candidates"
    base_dir.mkdir(parents=True, exist_ok=True)
    csv_path = base_dir / "accepted_candidates.csv"
    jsonl_path = base_dir / "accepted_candidates.jsonl"
    fieldnames = [
        "ts",
        "symbol",
        "side",
        "score",
        "confidence",
        "session_regime",
        "threshold_source",
        "entry_score_threshold",
        "confidence_threshold",
        "score_source",
        "expected_capture_bps",
        "expected_capture_adjusted_bps",
        "fill_probability",
        "replay_adjustment_bps",
        "bandit_adjustment_bps",
        "rank_score_baseline",
        "rank_score_post_adjustments",
        "rank_index",
        "group_size",
        "submit_price",
        "markout_price",
        "markout_bps",
        "net_markout_bps",
        "direction_correct",
        "accepted_reason",
        "veto_shadow_reason",
        "veto_bucket",
        "filled",
        "fill_price",
        "fill_qty",
        "client_order_id",
        "size_multiplier",
        "sizing_policy",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "accepted_candidates_csv": str(csv_path),
        "accepted_candidates_jsonl": str(jsonl_path),
    }


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
    load_reports: dict[str, HistoricalBarLoadReport] = {}
    order_context_by_client_id: dict[str, dict[str, Any]] = {}
    policy_counters: Counter[str] = Counter()
    markout_veto_counters: Counter[str] = Counter()
    markout_veto_config = _resolve_markout_veto_config(args)
    markout_veto_history: dict[str, deque[float]] = {}
    opportunity_opened_symbols: set[str] = set()
    synthetic_index = 0
    for symbol, csv_path in symbol_paths.items():
        frame, load_report = load_historical_bars(csv_path, timestamp_col=args.timestamp_col)
        load_reports[symbol] = load_report
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
        confidence_threshold, entry_score_threshold, session_regime, threshold_source = (
            _thresholds_for_regime(cfg, ts=ts_iso)
        )
        if confidence < confidence_threshold:
            return None
        side: str | None = None
        if score >= entry_score_threshold:
            side = "buy"
        elif (
            cfg.allow_shorts
            and (model_context is None or model_context.supports_short_scores)
            and score <= -entry_score_threshold
        ):
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
        opportunity_gate_skipped = False

        next_close_raw = bar.get("next_close")
        next_close = None
        if next_close_raw not in (None, ""):
            try:
                parsed_next = float(next_close_raw)
            except (TypeError, ValueError):
                parsed_next = None
            if parsed_next is not None and np.isfinite(parsed_next) and parsed_next > 0.0:
                next_close = float(parsed_next)

        if policy_profile is not None:
            policy_counters["candidates"] += 1
            opportunity_gate_applied = not (
                policy_profile.opportunity_openings_only
                and symbol in opportunity_opened_symbols
            )
            if opportunity_gate_applied:
                keep_count = _policy_keep_count(
                    group_size=group_size,
                    top_quantile=policy_profile.opportunity_top_quantile,
                    min_symbols=policy_profile.opportunity_min_symbols,
                )
                if rank_index >= keep_count:
                    policy_counters["reject_opportunity_quantile"] += 1
                    return None
            else:
                policy_counters["opportunity_openings_only_skipped"] += 1
                opportunity_gate_skipped = True
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

        veto_bucket = f"{symbol}:{side}"
        veto_shadow_reason = None
        if markout_veto_config is not None:
            markout_veto_counters["candidates"] += 1
            history = markout_veto_history.setdefault(
                veto_bucket,
                deque(maxlen=int(markout_veto_config.lookback)),
            )
            veto_shadow_reason = _markout_veto_reason(
                history,
                config=markout_veto_config,
            )
            if veto_shadow_reason:
                markout_veto_counters["shadow_flagged"] += 1
                if markout_veto_config.mode == "enforce":
                    markout_veto_counters["rejected"] += 1
                    markout_veto_counters[f"reject_{veto_shadow_reason}"] += 1
                    return None
        if policy_profile is not None:
            policy_counters["accepted"] += 1

        entry_side = side if side == "buy" else "sell_short"
        entry_slippage_bps = _replay_slippage_bps(
            cfg,
            symbol=symbol,
            side=entry_side,
            ts=ts_iso,
        )
        size_multiplier, sizing_context = _sizing_multiplier(
            cfg,
            confidence=confidence,
            score=score,
            slippage_bps=entry_slippage_bps,
        )
        bar_seq = int(bar.get("seq", 0) or 0)
        intent_key = f"{symbol}|{ts_iso}|{side}|{bar_seq}"
        candidate_markout = _candidate_markout_bps(
            side=side,
            submit_price=float(price),
            markout_price=next_close,
        )
        if markout_veto_config is not None and candidate_markout is not None:
            markout_veto_history.setdefault(
                veto_bucket,
                deque(maxlen=int(markout_veto_config.lookback)),
            ).append(float(candidate_markout))
        order_context_by_client_id[intent_key] = {
            "client_order_id": intent_key,
            "symbol": symbol,
            "side": side,
            "submit_ts": ts_iso,
            "submit_price": float(price),
            "markout_price": next_close,
            "score": float(score),
            "confidence": float(confidence),
            "score_source": score_source,
            "entry_score_threshold": float(entry_score_threshold),
            "confidence_threshold": float(confidence_threshold),
            "session_regime": session_regime,
            "threshold_source": threshold_source,
            "policy_fill_prob_proxy": float(fill_prob_proxy),
            "policy_expected_capture_proxy_bps": float(expected_capture_proxy),
            "policy_expected_capture_adjusted_bps": float(adjusted_capture),
            "policy_replay_adjustment_bps": float(replay_adjustment),
            "policy_bandit_adjustment_bps": float(bandit_adjustment),
            "policy_rank_index": int(rank_index),
            "policy_group_size": int(group_size),
            "rank_score_baseline": float(expected_capture_proxy),
            "rank_score_post_adjustments": float(adjusted_capture),
            "accepted_reason": "policy_controls" if policy_profile is not None else "thresholds",
            "opportunity_openings_only_skip": bool(opportunity_gate_skipped),
            "veto_shadow_reason": veto_shadow_reason,
            "veto_bucket": veto_bucket,
            "size_multiplier": float(size_multiplier),
            "sizing": dict(sizing_context),
        }
        return {
            "symbol": symbol,
            "side": side,
            "qty": float(size_multiplier),
            "type": "limit",
            "price": price,
            "limit_price": price,
            "intent_key": intent_key,
            "client_order_id": intent_key,
        }

    def on_replay_events(replay_events: list[dict[str, Any]]) -> None:
        for event in replay_events:
            if str(event.get("event_type", "")).strip().lower() != "fill":
                continue
            try:
                fill_qty = float(event.get("fill_qty", 0.0) or 0.0)
            except (TypeError, ValueError):
                fill_qty = 0.0
            if fill_qty <= 0.0:
                continue
            client_order_id = str(event.get("client_order_id", "")).strip()
            context = order_context_by_client_id.get(client_order_id)
            if not context:
                continue
            symbol_value = str(context.get("symbol") or event.get("symbol") or "").strip().upper()
            if symbol_value:
                opportunity_opened_symbols.add(symbol_value)

    replay = ReplayEventLoop(
        strategy=strategy,
        seed=int(args.replay_seed),
        max_symbol_notional=args.max_symbol_notional,
        max_gross_notional=args.max_gross_notional,
        event_callback=on_replay_events if policy_profile is not None else None,
    ).run(bars)

    events = list(replay.get("events", []))
    fill_events = [event for event in events if event.get("event_type") == "fill"]
    fill_event_by_client_id = {
        str(event.get("client_order_id", "")).strip(): event
        for event in fill_events
        if str(event.get("client_order_id", "")).strip()
    }
    fill_count_by_symbol = Counter(
        str(event.get("symbol", "")).strip().upper()
        for event in fill_events
        if str(event.get("symbol", "")).strip()
    )
    violations = list(replay.get("violations", []))
    cap_adjustments = list(replay.get("cap_adjustments", []))
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
        "cap_adjustments_count": int(len(cap_adjustments)),
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
            "regime_thresholds": _regime_threshold_config_payload(cfg.regime_thresholds),
        },
    }
    if cfg.live_cost_model is not None:
        aggregate["config"]["live_cost_model"] = _live_cost_model_config_payload(
            cfg.live_cost_model
        )
    if model_context is not None:
        aggregate["model_score"] = {
            "enabled": True,
            "model_path": model_context.model_path,
            "orientation": "inverse" if model_context.orientation_inverse else "direct",
            "symbol_penalty_count": len(model_context.symbol_penalties),
            "feature_count": len(model_context.feature_names),
            "supports_short_scores": bool(model_context.supports_short_scores),
        }
    else:
        aggregate["model_score"] = {"enabled": False}
    if policy_profile is not None:
        aggregate["policy_mode"] = str(policy_mode_label or "ranker_sensitivity")
        aggregate["policy_diagnostics"] = _policy_finalize_diagnostics(
            policy_counters,
            profile=policy_profile,
        )
    if markout_veto_config is not None:
        rejected_by_reason = {
            key.replace("reject_", ""): int(value)
            for key, value in sorted(markout_veto_counters.items())
            if key.startswith("reject_")
        }
        aggregate["markout_veto"] = {
            "enabled": True,
            "mode": markout_veto_config.mode,
            "lookback": int(markout_veto_config.lookback),
            "min_samples": int(markout_veto_config.min_samples),
            "min_mean_bps": float(markout_veto_config.min_mean_bps),
            "max_wrong_way_rate": float(markout_veto_config.max_wrong_way_rate),
            "candidates": int(markout_veto_counters.get("candidates", 0)),
            "shadow_flagged": int(markout_veto_counters.get("shadow_flagged", 0)),
            "rejected": int(markout_veto_counters.get("rejected", 0)),
            "rejected_by_reason": rejected_by_reason,
        }
    else:
        aggregate["markout_veto"] = {"enabled": False, "mode": "off"}

    accepted_candidate_rows = [
        _accepted_candidate_row(
            context=context,
            fill_event=fill_event_by_client_id.get(str(client_order_id)),
            cfg=cfg,
        )
        for client_order_id, context in sorted(order_context_by_client_id.items())
    ]
    aggregate["accepted_candidate_count"] = int(len(accepted_candidate_rows))
    aggregate["candidate_quality"] = _summarize_candidate_quality(
        accepted_candidate_rows,
        cap_adjustments=cast(list[Mapping[str, Any]], cap_adjustments),
    )
    if markout_veto_config is not None:
        aggregate["markout_veto"]["shadow_quality"] = (
            _summarize_markout_veto_shadow_quality(accepted_candidate_rows)
        )
    artifacts: dict[str, str] = {}
    if bool(getattr(args, "export_accepted_candidates", False)):
        artifacts.update(
            _write_accepted_candidate_artifacts(
                args=args,
                rows=accepted_candidate_rows,
            )
        )
    for item in per_symbol:
        symbol = str(item.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        item["fills"] = int(fill_count_by_symbol.get(symbol, 0))
        item["markout_samples"] = int(markout_samples_by_symbol.get(symbol, 0))
    return {
        "schema_version": OFFLINE_REPLAY_SCHEMA_VERSION,
        "artifact_type": "offline_replay_summary",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "aggregate": aggregate,
        "inputs": {"symbols": _input_report_payload(load_reports)},
        "symbols": per_symbol,
        "replay": replay,
        "artifacts": artifacts,
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
    replay_quality_sweep_max_age_hours = max(
        float(baseline_profile.replay_quality_max_age_hours),
        24.0 * 30.0,
    )
    bandit_live_description = (
        "Enable live bandit rank adjustment with permissive sample threshold."
        if baseline_profile.bandit_shadow_only
        else "Increase live bandit coverage with permissive sample threshold."
    )
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
            bandit_live_description,
            replace(
                baseline_profile,
                bandit_shadow_only=False,
                bandit_min_samples=1,
            ),
        ),
        (
            "replay_quality_weight_0_10",
            "Replay-quality weight sweep variant: 0.10 (extended age horizon).",
            replace(
                baseline_profile,
                replay_quality_weight=0.10,
                replay_quality_max_age_hours=replay_quality_sweep_max_age_hours,
            ),
        ),
        (
            "replay_quality_weight_0_25",
            "Replay-quality weight sweep variant: 0.25 (extended age horizon).",
            replace(
                baseline_profile,
                replay_quality_weight=0.25,
                replay_quality_max_age_hours=replay_quality_sweep_max_age_hours,
            ),
        ),
        (
            "replay_quality_weight_0_40",
            "Replay-quality weight sweep variant: 0.40 (extended age horizon).",
            replace(
                baseline_profile,
                replay_quality_weight=0.40,
                replay_quality_max_age_hours=replay_quality_sweep_max_age_hours,
            ),
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
    live_cost_model = _load_live_cost_replay_model(args)
    regime_thresholds = _load_regime_threshold_model(args)
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
        live_cost_model=live_cost_model,
        regime_thresholds=regime_thresholds,
        sizing_policy=str(args.sizing_policy or "flat").strip().lower() or "flat",
        sizing_min_scale=max(0.0, float(args.sizing_min_scale)),
        sizing_max_scale=max(0.0, float(args.sizing_max_scale)),
        sizing_cost_penalty_bps=max(0.0, float(args.sizing_cost_penalty_bps)),
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
    load_reports: dict[str, HistoricalBarLoadReport] = {}
    for symbol, csv_path in symbol_paths.items():
        frame, load_report = load_historical_bars(csv_path, timestamp_col=args.timestamp_col)
        load_reports[symbol] = load_report
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
    size_multipliers = np.asarray(
        [float(t.get("size_multiplier", 1.0) or 1.0) for t in all_trades],
        dtype=float,
    )
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
    if cfg.live_cost_model is not None:
        aggregate["config"]["live_cost_model"] = _live_cost_model_config_payload(
            cfg.live_cost_model
        )
    if cfg.regime_thresholds is not None:
        aggregate["config"]["regime_thresholds"] = _regime_threshold_config_payload(
            cfg.regime_thresholds
        )
        aggregate["by_session_regime"] = _summarize_trades_by_regime(all_trades)
    if str(cfg.sizing_policy or "flat").strip().lower() != "flat":
        aggregate["avg_size_multiplier"] = (
            float(size_multipliers.mean()) if size_multipliers.size else 1.0
        )
        aggregate["max_size_multiplier"] = (
            float(size_multipliers.max()) if size_multipliers.size else 1.0
        )
        aggregate["config"]["sizing_policy"] = cfg.sizing_policy
        aggregate["config"]["sizing_min_scale"] = cfg.sizing_min_scale
        aggregate["config"]["sizing_max_scale"] = cfg.sizing_max_scale
        aggregate["config"]["sizing_cost_penalty_bps"] = cfg.sizing_cost_penalty_bps
    return {
        "schema_version": OFFLINE_REPLAY_SCHEMA_VERSION,
        "artifact_type": "offline_replay_summary",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "aggregate": aggregate,
        "inputs": {"symbols": _input_report_payload(load_reports)},
        "symbols": per_symbol,
    }


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
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
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
        payload.setdefault("artifacts", {})
        artifacts = payload.get("artifacts")
        if isinstance(artifacts, dict):
            artifacts["output_json"] = str(args.output_json)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        logger.info("OFFLINE_REPLAY_JSON_WRITTEN", extra={"path": str(args.output_json)})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
