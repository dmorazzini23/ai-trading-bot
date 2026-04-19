"""Prelude helpers for netting-cycle ranking state."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping, Sequence


@dataclass(slots=True)
class ReplayQualityState:
    by_symbol: dict[str, dict[str, float]]
    by_symbol_session: dict[str, dict[str, float]]
    by_symbol_session_regime: dict[str, dict[str, float]]
    context: dict[str, Any]
    effective_weight: float


@dataclass(slots=True)
class PolicyRuntimeOverrideState:
    bandit_enabled: bool
    counterfactual_enabled: bool
    geometric_tiebreak_enabled: bool
    portfolio_log_growth_rank_enabled: bool
    disabled_gate_roots: set[str]
    disabled_sleeves: set[str]
    payload: dict[str, Any]


def _normalize_symbol_quality_map(
    raw: Any,
    *,
    safe_float: Callable[[Any], float | None],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if not isinstance(raw, Mapping):
        return out
    for raw_key, raw_metrics in raw.items():
        symbol = str(raw_key or "").strip().upper()
        if not symbol or not isinstance(raw_metrics, Mapping):
            continue
        sample_count = int(safe_float(raw_metrics.get("sample_count")) or 0.0)
        net_edge_bps = safe_float(raw_metrics.get("net_edge_bps"))
        if sample_count <= 0 or net_edge_bps is None:
            continue
        out[symbol] = {
            "sample_count": float(sample_count),
            "net_edge_bps": float(net_edge_bps),
            "win_rate": float(safe_float(raw_metrics.get("win_rate")) or 0.0),
            "profit_factor": float(safe_float(raw_metrics.get("profit_factor")) or 0.0),
        }
    return out


def _normalize_bucket_quality_map(
    raw: Any,
    *,
    with_regime: bool,
    safe_float: Callable[[Any], float | None],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if not isinstance(raw, Mapping):
        return out
    expected_parts = 3 if with_regime else 2
    for raw_key, raw_metrics in raw.items():
        key = str(raw_key or "").strip()
        if not key or not isinstance(raw_metrics, Mapping):
            continue
        parts = key.split(":")
        if len(parts) < expected_parts:
            continue
        symbol = str(parts[0] or "").strip().upper()
        session_token = str(parts[1] or "").strip().lower() or "offhours"
        if with_regime:
            regime_token = str(parts[2] or "").strip().lower() or "unknown"
            normalized_key = f"{symbol}:{session_token}:{regime_token}"
        else:
            normalized_key = f"{symbol}:{session_token}"
        sample_count = int(safe_float(raw_metrics.get("sample_count")) or 0.0)
        net_edge_bps = safe_float(raw_metrics.get("net_edge_bps"))
        if sample_count <= 0 or net_edge_bps is None:
            continue
        out[normalized_key] = {
            "sample_count": float(sample_count),
            "net_edge_bps": float(net_edge_bps),
            "win_rate": float(safe_float(raw_metrics.get("win_rate")) or 0.0),
            "profit_factor": float(safe_float(raw_metrics.get("profit_factor")) or 0.0),
        }
    return out


def load_replay_quality_state(
    *,
    state: Any,
    now: datetime,
    enabled: bool,
    weight: float,
    max_age_hours: float,
    auto_disable_if_stale: bool,
    get_env: Callable[..., Any],
    safe_float: Callable[[Any], float | None],
    parse_iso_timestamp: Callable[[Any], datetime | None],
    resolve_runtime_artifact_path_func: Callable[..., Any],
    load_latest_replay_quality_summaries_func: Callable[..., tuple[Any, Any, Any, dict[str, Any]]],
) -> ReplayQualityState:
    by_symbol: dict[str, dict[str, float]] = {}
    by_symbol_session: dict[str, dict[str, float]] = {}
    by_symbol_session_regime: dict[str, dict[str, float]] = {}
    context: dict[str, Any] = {"source": "none"}
    effective_weight = float(weight)
    if enabled:
        state_replay_raw = getattr(state, "replay_symbol_summary", None)
        state_replay_buckets_raw = getattr(state, "replay_bucket_summary", None)
        state_replay_ts = parse_iso_timestamp(
            getattr(state, "replay_symbol_summary_updated_at", None)
        )
        if isinstance(state_replay_raw, Mapping) or isinstance(state_replay_buckets_raw, Mapping):
            age_seconds: float | None = None
            if state_replay_ts is not None:
                age_seconds = max(0.0, (now - state_replay_ts).total_seconds())
            if age_seconds is None or age_seconds <= float(max_age_hours) * 3600.0:
                by_symbol = _normalize_symbol_quality_map(
                    state_replay_raw,
                    safe_float=safe_float,
                )
                buckets = (
                    dict(state_replay_buckets_raw)
                    if isinstance(state_replay_buckets_raw, Mapping)
                    else {}
                )
                by_symbol_session = _normalize_bucket_quality_map(
                    buckets.get("by_symbol_session"),
                    with_regime=False,
                    safe_float=safe_float,
                )
                by_symbol_session_regime = _normalize_bucket_quality_map(
                    buckets.get("by_symbol_session_regime"),
                    with_regime=True,
                    safe_float=safe_float,
                )
                if by_symbol or by_symbol_session or by_symbol_session_regime:
                    context = {
                        "source": "state",
                        "updated_at": (
                            state_replay_ts.isoformat() if state_replay_ts is not None else None
                        ),
                        "symbols": int(len(by_symbol)),
                        "symbol_sessions": int(len(by_symbol_session)),
                        "symbol_session_regimes": int(len(by_symbol_session_regime)),
                    }
        if not by_symbol and not by_symbol_session and not by_symbol_session_regime:
            replay_output_dir = resolve_runtime_artifact_path_func(
                str(
                    get_env(
                        "AI_TRADING_REPLAY_OUTPUT_DIR",
                        "runtime/replay_outputs",
                        cast=str,
                    )
                    or ""
                ).strip()
                or "runtime/replay_outputs",
                default_relative="runtime/replay_outputs",
            )
            (
                loaded_symbol,
                loaded_session,
                loaded_session_regime,
                context,
            ) = load_latest_replay_quality_summaries_func(
                replay_output_dir,
                max_age_hours=float(max_age_hours),
            )
            by_symbol = _normalize_symbol_quality_map(
                loaded_symbol,
                safe_float=safe_float,
            )
            by_symbol_session = _normalize_bucket_quality_map(
                loaded_session,
                with_regime=False,
                safe_float=safe_float,
            )
            by_symbol_session_regime = _normalize_bucket_quality_map(
                loaded_session_regime,
                with_regime=True,
                safe_float=safe_float,
            )
        if auto_disable_if_stale and not by_symbol and not by_symbol_session and not by_symbol_session_regime:
            effective_weight = 0.0
            context = {
                **dict(context),
                "source": "none",
                "auto_disabled": True,
                "auto_disabled_reason": "stale_or_missing_data",
            }
    return ReplayQualityState(
        by_symbol=by_symbol,
        by_symbol_session=by_symbol_session,
        by_symbol_session_regime=by_symbol_session_regime,
        context=context,
        effective_weight=effective_weight,
    )


def apply_policy_runtime_overrides(
    *,
    load_policy_runtime_toggles_func: Callable[[], Mapping[str, Any]],
    bandit_enabled: bool,
    counterfactual_enabled: bool,
    geometric_tiebreak_enabled: bool,
    portfolio_log_growth_rank_enabled: bool,
) -> PolicyRuntimeOverrideState:
    payload = dict(load_policy_runtime_toggles_func())
    disabled_slices = {
        str(item).strip().upper()
        for item in payload.get("disabled_slices", [])
        if str(item).strip()
    }
    toggles_raw = payload.get("toggles")
    toggles = dict(toggles_raw) if isinstance(toggles_raw, Mapping) else {}
    ranker_toggles_raw = toggles.get("rankers")
    ranker_toggles = dict(ranker_toggles_raw) if isinstance(ranker_toggles_raw, Mapping) else {}

    if "RANKER:BANDIT" in disabled_slices or not bool(ranker_toggles.get("bandit_enabled", True)):
        bandit_enabled = False
    if "RANKER:COUNTERFACTUAL" in disabled_slices or not bool(
        ranker_toggles.get("counterfactual_enabled", True)
    ):
        counterfactual_enabled = False
    if "RANKER:GEOMETRIC" in disabled_slices or not bool(ranker_toggles.get("geometric_enabled", True)):
        geometric_tiebreak_enabled = False
    if "RANKER:PORTFOLIO_LOG_GROWTH" in disabled_slices or not bool(
        ranker_toggles.get("portfolio_log_growth_enabled", True)
    ):
        portfolio_log_growth_rank_enabled = False

    disabled_gate_roots_raw = toggles.get("disabled_gate_roots")
    if isinstance(disabled_gate_roots_raw, Sequence) and not isinstance(
        disabled_gate_roots_raw,
        (str, bytes, bytearray),
    ):
        disabled_gate_roots = {
            str(item).strip().upper()
            for item in disabled_gate_roots_raw
            if str(item).strip()
        }
    else:
        disabled_gate_roots = {
            str(item).split(":", 1)[1].strip().upper()
            for item in disabled_slices
            if str(item).startswith("GATE:") and ":" in str(item)
        }

    disabled_sleeves_raw = toggles.get("disabled_sleeves")
    if isinstance(disabled_sleeves_raw, Sequence) and not isinstance(
        disabled_sleeves_raw,
        (str, bytes, bytearray),
    ):
        disabled_sleeves = {
            str(item).strip().lower()
            for item in disabled_sleeves_raw
            if str(item).strip()
        }
    else:
        disabled_sleeves = {
            str(item).split(":", 1)[1].strip().lower()
            for item in disabled_slices
            if str(item).startswith("SLEEVE:") and ":" in str(item)
        }

    return PolicyRuntimeOverrideState(
        bandit_enabled=bandit_enabled,
        counterfactual_enabled=counterfactual_enabled,
        geometric_tiebreak_enabled=geometric_tiebreak_enabled,
        portfolio_log_growth_rank_enabled=portfolio_log_growth_rank_enabled,
        disabled_gate_roots=disabled_gate_roots,
        disabled_sleeves=disabled_sleeves,
        payload=payload,
    )


__all__ = [
    "PolicyRuntimeOverrideState",
    "ReplayQualityState",
    "apply_policy_runtime_overrides",
    "load_replay_quality_state",
]
