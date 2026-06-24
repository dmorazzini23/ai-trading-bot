"""Build rolling symbol-quality scorecards for universe pruning decisions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from ai_trading.config.management import get_env
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _to_int(value: Any) -> int:
    parsed = _to_float(value)
    if parsed is None:
        return 0
    return max(0, int(parsed))


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _symbol_set(raw_value: Any) -> set[str]:
    return {
        token.strip().upper()
        for token in str(raw_value or "").replace(";", ",").split(",")
        if token and token.strip()
    }


def _default_path(env_key: str, default_relative: str) -> Path:
    configured = str(
        get_env(env_key, default_relative, cast=str, resolve_aliases=False)
        or default_relative
    ).strip()
    return resolve_runtime_artifact_path(configured, default_relative=default_relative)


def _path_arg(raw: str, *, env_key: str, default_relative: str) -> Path:
    if str(raw or "").strip():
        return Path(raw).expanduser()
    return _default_path(env_key, default_relative)


def _weighted_mean(values: Iterable[tuple[float | None, int]]) -> float | None:
    total = 0.0
    weight = 0
    for value, count in values:
        if value is None or count <= 0:
            continue
        total += float(value) * int(count)
        weight += int(count)
    if weight <= 0:
        return None
    return float(total / weight)


def _max_metric(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return max(clean) if clean else None


def _symbol_row_map(rows: Any) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if symbol:
            out[symbol] = row
    return out


def _live_cost_by_symbol(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    buckets = payload.get("by_symbol_side_session")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    if isinstance(buckets, list):
        for row in buckets:
            if not isinstance(row, Mapping):
                continue
            symbol = str(row.get("symbol") or "").strip().upper()
            if symbol:
                grouped[symbol].append(row)
    out: dict[str, dict[str, Any]] = {}
    for symbol, rows in grouped.items():
        samples = sum(_to_int(row.get("sample_count")) for row in rows)
        out[symbol] = {
            "sample_count": int(samples),
            "mean_total_cost_bps": _weighted_mean(
                (_to_float(row.get("mean_total_cost_bps")), _to_int(row.get("sample_count")))
                for row in rows
            ),
            "p90_total_cost_bps": _max_metric(
                _to_float(row.get("p90_total_cost_bps")) for row in rows
            ),
            "mean_spread_bps": _weighted_mean(
                (_to_float(row.get("mean_spread_bps")), _to_int(row.get("sample_count")))
                for row in rows
            ),
            "p90_spread_bps": _max_metric(_to_float(row.get("p90_spread_bps")) for row in rows),
            "mean_quote_age_ms": _weighted_mean(
                (_to_float(row.get("mean_quote_age_ms")), _to_int(row.get("sample_count")))
                for row in rows
            ),
            "p90_quote_age_ms": _max_metric(
                _to_float(row.get("p90_quote_age_ms")) for row in rows
            ),
        }
    return out


def _shadow_markout_by_symbol(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    summary = payload.get("markout_summary")
    if not isinstance(summary, Mapping):
        summaries = payload.get("markout_summaries")
        if isinstance(summaries, Mapping) and summaries:
            first_key = sorted(str(key) for key in summaries)[0]
            candidate = summaries.get(first_key)
            summary = candidate if isinstance(candidate, Mapping) else {}
    if not isinstance(summary, Mapping):
        return {}
    rows: list[Any] = []
    rows.extend(summary.get("best_symbols") if isinstance(summary.get("best_symbols"), list) else [])
    rows.extend(summary.get("worst_symbols") if isinstance(summary.get("worst_symbols"), list) else [])
    return _symbol_row_map(rows)


def _execution_quality_by_symbol(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return _symbol_row_map(payload.get("by_symbol"))


def _replay_summary_by_symbol(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = payload.get("replay_symbol_summary")
    if not isinstance(rows, Mapping):
        return {}
    out: dict[str, Mapping[str, Any]] = {}
    for symbol, row in rows.items():
        symbol_key = str(symbol or "").strip().upper()
        if symbol_key and isinstance(row, Mapping):
            out[symbol_key] = row
    return out


def _previous_persistence(payload: Mapping[str, Any]) -> dict[str, tuple[str, int]]:
    out: dict[str, tuple[str, int]] = {}
    rows = payload.get("symbols")
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        mode = str(row.get("recommended_mode") or "allow").strip().lower()
        if symbol:
            out[symbol] = (mode, _to_int(row.get("persistence_count")))
    return out


def _samples_from_markout(row: Mapping[str, Any] | None) -> int:
    if not isinstance(row, Mapping):
        return 0
    return max(
        _to_int(row.get("samples")),
        _to_int(row.get("sample_count")),
        _to_int(row.get("champion_samples")),
        _to_int(row.get("challenger_samples")),
    )


def _quality_score(
    *,
    mean_markout_bps: float | None,
    positive_rate: float | None,
    p90_total_cost_bps: float | None,
    p90_spread_bps: float | None,
) -> float | None:
    components: list[float] = []
    if mean_markout_bps is not None:
        components.append(float(mean_markout_bps))
    if positive_rate is not None:
        components.append((float(positive_rate) - 0.5) * 20.0)
    if p90_total_cost_bps is not None:
        components.append(-float(p90_total_cost_bps) * 0.5)
    if p90_spread_bps is not None:
        components.append(-float(p90_spread_bps) * 0.2)
    if not components:
        return None
    return float(sum(components))


def _mode_for_symbol(
    *,
    sample_count: int,
    mean_markout_bps: float | None,
    p90_total_cost_bps: float | None,
    p90_spread_bps: float | None,
    min_samples: int,
    shadow_total_cost_bps: float,
    disable_total_cost_bps: float,
    shadow_markout_bps: float,
    disable_markout_bps: float,
    shadow_spread_bps: float,
    disable_spread_bps: float,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if sample_count < min_samples:
        return "allow", ["insufficient_samples"]
    disable = False
    shadow = False
    if p90_total_cost_bps is not None:
        if p90_total_cost_bps >= disable_total_cost_bps:
            disable = True
            reasons.append("p90_total_cost_bps_disable")
        elif p90_total_cost_bps >= shadow_total_cost_bps:
            shadow = True
            reasons.append("p90_total_cost_bps_shadow")
    if mean_markout_bps is not None:
        if mean_markout_bps <= disable_markout_bps:
            disable = True
            reasons.append("mean_markout_bps_disable")
        elif mean_markout_bps <= shadow_markout_bps:
            shadow = True
            reasons.append("mean_markout_bps_shadow")
    if p90_spread_bps is not None:
        if p90_spread_bps >= disable_spread_bps:
            disable = True
            reasons.append("p90_spread_bps_disable")
        elif p90_spread_bps >= shadow_spread_bps:
            shadow = True
            reasons.append("p90_spread_bps_shadow")
    if disable:
        return "disabled", reasons
    if shadow:
        return "shadow_only", reasons
    return "allow", reasons or ["healthy"]


def _shadow_promotion_suggestions(
    rows: list[dict[str, Any]],
    *,
    executable_symbols: set[str],
    shadow_symbols: set[str],
    min_score_delta: float,
    min_samples: int,
) -> dict[str, Any]:
    by_symbol = {
        str(row.get("symbol") or "").strip().upper(): row
        for row in rows
        if str(row.get("symbol") or "").strip()
    }
    executable_scores = [
        _to_float(by_symbol[symbol].get("quality_score"))
        for symbol in executable_symbols
        if symbol in by_symbol
    ]
    executable_scores = [score for score in executable_scores if score is not None]
    baseline_score = min(executable_scores) if executable_scores else None
    suggestions: list[dict[str, Any]] = []
    for symbol in sorted(shadow_symbols):
        row = by_symbol.get(symbol)
        if row is None:
            continue
        if str(row.get("effective_mode") or "").lower() != "allow":
            continue
        sample_count = _to_int(row.get("sample_count"))
        if sample_count < max(1, int(min_samples)):
            continue
        quality_score = _to_float(row.get("quality_score"))
        if quality_score is None:
            continue
        score_delta = None if baseline_score is None else float(quality_score - baseline_score)
        if score_delta is not None and score_delta < float(min_score_delta):
            continue
        suggestions.append(
            {
                "symbol": symbol,
                "recommended_action": "consider_promote_shadow_to_canary",
                "quality_score": float(quality_score),
                "baseline_executable_score": baseline_score,
                "score_delta": score_delta,
                "sample_count": int(sample_count),
                "current_shadow_symbols": sorted(shadow_symbols),
                "current_executable_symbols": sorted(executable_symbols),
                "reason": "shadow_symbol_scores_better_than_executable_baseline",
            }
        )
    return {
        "available": bool(suggestions),
        "suggestions": suggestions,
        "thresholds": {
            "min_score_delta": float(min_score_delta),
            "min_samples": int(max(1, min_samples)),
        },
    }


def _universe_diagnostics(
    rows: list[dict[str, Any]],
    *,
    executable_symbols: set[str],
    shadow_symbols: set[str],
    starvation_threshold: float,
) -> dict[str, Any]:
    evidence_symbols = {
        str(row.get("symbol") or "").strip().upper()
        for row in rows
        if str(row.get("symbol") or "").strip()
    }
    configured_symbols = set(executable_symbols) | set(shadow_symbols)
    row_samples = {
        str(row.get("symbol") or "").strip().upper(): _to_int(row.get("sample_count"))
        for row in rows
        if str(row.get("symbol") or "").strip()
    }
    total_samples = sum(row_samples.values())
    dominant_symbol = None
    dominant_share = 0.0
    if total_samples > 0 and row_samples:
        dominant_symbol, dominant_samples = max(row_samples.items(), key=lambda item: item[1])
        dominant_share = float(dominant_samples / total_samples)
    executable_without_evidence = sorted(executable_symbols - evidence_symbols)
    configured_without_evidence = sorted(configured_symbols - evidence_symbols)
    diagnostics = {
        "universe_mismatch": bool(configured_without_evidence),
        "configured_symbols": sorted(configured_symbols),
        "evidence_symbols": sorted(evidence_symbols),
        "executable_symbols": sorted(executable_symbols),
        "shadow_symbols": sorted(shadow_symbols),
        "configured_without_evidence": configured_without_evidence,
        "executable_without_evidence": executable_without_evidence,
        "symbol_starvation": bool(
            dominant_symbol is not None
            and len(configured_symbols) > 1
            and dominant_share >= float(starvation_threshold)
        ),
        "dominant_symbol": dominant_symbol,
        "dominant_sample_share": dominant_share,
        "starvation_threshold": float(starvation_threshold),
    }
    return diagnostics


def _synthetic_exploration_row(symbol: str) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "sample_count": 0,
        "live_cost_sample_count": 0,
        "shadow_markout_samples": 0,
        "replay_samples": 0,
        "execution_quality_events": 0,
        "mean_net_markout_bps": None,
        "positive_rate": None,
        "profit_factor": None,
        "mean_total_cost_bps": None,
        "p90_total_cost_bps": None,
        "mean_spread_bps": None,
        "p90_spread_bps": None,
        "mean_quote_age_ms": None,
        "p90_quote_age_ms": None,
        "quality_score": None,
        "recommended_mode": "allow",
        "effective_mode": "allow",
        "persistence_count": 1,
        "reasons": ["configured_without_evidence", "paper_only_exploration_candidate"],
        "runtime_authority": False,
        "promotion_authority": False,
        "live_money_authority": False,
    }


def build_symbol_universe_scorecard(
    *,
    live_cost_model: Mapping[str, Any] | None = None,
    shadow_report: Mapping[str, Any] | None = None,
    execution_quality_governor: Mapping[str, Any] | None = None,
    replay_report: Mapping[str, Any] | None = None,
    previous_scorecard: Mapping[str, Any] | None = None,
    min_samples: int = 25,
    min_persistence: int = 2,
    shadow_total_cost_bps: float = 20.0,
    disable_total_cost_bps: float = 35.0,
    shadow_markout_bps: float = -10.0,
    disable_markout_bps: float = -25.0,
    shadow_spread_bps: float = 35.0,
    disable_spread_bps: float = 60.0,
    executable_symbols: Iterable[str] | None = None,
    shadow_symbols: Iterable[str] | None = None,
    shadow_promotion_min_score_delta: float = 0.5,
    shadow_promotion_min_samples: int = 10,
    starvation_threshold: float = 0.95,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return a symbol universe scorecard from recent runtime artifacts."""

    generated_at = now.astimezone(UTC) if now is not None else datetime.now(UTC)
    live_by_symbol = _live_cost_by_symbol(live_cost_model or {})
    markout_by_symbol = _shadow_markout_by_symbol(shadow_report or {})
    replay_by_symbol = _replay_summary_by_symbol(replay_report or {})
    quality_by_symbol = _execution_quality_by_symbol(execution_quality_governor or {})
    previous = _previous_persistence(previous_scorecard or {})
    symbols = sorted(
        set(live_by_symbol) | set(markout_by_symbol) | set(replay_by_symbol) | set(quality_by_symbol)
    )
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        live = live_by_symbol.get(symbol, {})
        markout = markout_by_symbol.get(symbol)
        replay = replay_by_symbol.get(symbol)
        quality = quality_by_symbol.get(symbol)
        live_samples = _to_int(live.get("sample_count"))
        markout_samples = _samples_from_markout(markout)
        replay_samples = _to_int(replay.get("sample_count")) if isinstance(replay, Mapping) else 0
        quality_samples = _to_int(quality.get("events")) if isinstance(quality, Mapping) else 0
        sample_count = max(live_samples, markout_samples, replay_samples, quality_samples)
        mean_markout_bps = _to_float(markout.get("mean_net_markout_bps")) if isinstance(markout, Mapping) else None
        if mean_markout_bps is None and isinstance(replay, Mapping):
            mean_markout_bps = _to_float(replay.get("net_edge_bps"))
        positive_rate = _to_float(markout.get("positive_rate")) if isinstance(markout, Mapping) else None
        if positive_rate is None and isinstance(replay, Mapping):
            positive_rate = _to_float(replay.get("win_rate"))
        p90_total_cost_bps = _to_float(live.get("p90_total_cost_bps"))
        p90_spread_bps = _to_float(live.get("p90_spread_bps"))
        if p90_spread_bps is None and isinstance(quality, Mapping):
            p90_spread_bps = _to_float(quality.get("p90_spread_bps"))
        recommended_mode, reasons = _mode_for_symbol(
            sample_count=sample_count,
            mean_markout_bps=mean_markout_bps,
            p90_total_cost_bps=p90_total_cost_bps,
            p90_spread_bps=p90_spread_bps,
            min_samples=max(1, int(min_samples)),
            shadow_total_cost_bps=float(shadow_total_cost_bps),
            disable_total_cost_bps=float(disable_total_cost_bps),
            shadow_markout_bps=float(shadow_markout_bps),
            disable_markout_bps=float(disable_markout_bps),
            shadow_spread_bps=float(shadow_spread_bps),
            disable_spread_bps=float(disable_spread_bps),
        )
        prev_mode, prev_count = previous.get(symbol, ("", 0))
        persistence_count = prev_count + 1 if prev_mode == recommended_mode else 1
        effective_mode = recommended_mode
        if recommended_mode != "allow" and persistence_count < max(1, int(min_persistence)):
            effective_mode = "allow"
            reasons = [*reasons, "awaiting_persistence"]
        rows.append(
            {
                "symbol": symbol,
                "sample_count": int(sample_count),
                "live_cost_sample_count": int(live_samples),
                "shadow_markout_samples": int(markout_samples),
                "replay_samples": int(replay_samples),
                "execution_quality_events": int(quality_samples),
                "mean_net_markout_bps": mean_markout_bps,
                "positive_rate": positive_rate,
                "profit_factor": (
                    _to_float(replay.get("profit_factor")) if isinstance(replay, Mapping) else None
                ),
                "mean_total_cost_bps": _to_float(live.get("mean_total_cost_bps")),
                "p90_total_cost_bps": p90_total_cost_bps,
                "mean_spread_bps": _to_float(live.get("mean_spread_bps")),
                "p90_spread_bps": p90_spread_bps,
                "mean_quote_age_ms": _to_float(live.get("mean_quote_age_ms")),
                "p90_quote_age_ms": _to_float(live.get("p90_quote_age_ms")),
                "quality_score": _quality_score(
                    mean_markout_bps=mean_markout_bps,
                    positive_rate=positive_rate,
                    p90_total_cost_bps=p90_total_cost_bps,
                    p90_spread_bps=p90_spread_bps,
                ),
                "recommended_mode": recommended_mode,
                "effective_mode": effective_mode,
                "persistence_count": int(persistence_count),
                "reasons": reasons,
            }
        )
    rows.sort(
        key=lambda row: (
            str(row.get("effective_mode")) != "disabled",
            str(row.get("effective_mode")) != "shadow_only",
            -(row.get("sample_count") or 0),
            str(row.get("symbol") or ""),
        )
    )
    executable_set = {
        str(symbol).strip().upper()
        for symbol in (executable_symbols or [])
        if str(symbol).strip()
    }
    shadow_set = {
        str(symbol).strip().upper()
        for symbol in (shadow_symbols or [])
        if str(symbol).strip()
    }
    existing_symbols = {
        str(row.get("symbol") or "").strip().upper()
        for row in rows
        if str(row.get("symbol") or "").strip()
    }
    paper_sampling_symbols = _symbol_set(
        get_env(
            "AI_TRADING_PAPER_SAMPLING_ALLOWED_SYMBOLS",
            "",
            cast=str,
            resolve_aliases=False,
        )
    )
    include_zero_evidence = bool(
        get_env(
            "AI_TRADING_SYMBOL_UNIVERSE_INCLUDE_CONFIGURED_WITHOUT_EVIDENCE",
            True,
            cast=bool,
        )
    )
    zero_evidence_exploration_symbols: list[str] = []
    if include_zero_evidence and paper_sampling_symbols:
        configured_for_exploration = sorted((executable_set | shadow_set) & paper_sampling_symbols)
        for symbol in configured_for_exploration:
            if symbol not in existing_symbols:
                rows.append(_synthetic_exploration_row(symbol))
                existing_symbols.add(symbol)
                zero_evidence_exploration_symbols.append(symbol)
        rows.sort(
            key=lambda row: (
                str(row.get("effective_mode")) != "disabled",
                str(row.get("effective_mode")) != "shadow_only",
                -(row.get("sample_count") or 0),
                str(row.get("symbol") or ""),
            )
        )
    disabled = [str(row["symbol"]) for row in rows if row.get("effective_mode") == "disabled"]
    shadow_only = [
        str(row["symbol"]) for row in rows if row.get("effective_mode") == "shadow_only"
    ]
    status = "ready" if rows else "unavailable"
    return {
        "schema_version": "1.0.0",
        "artifact_type": "symbol_universe_scorecard",
        "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
        "source": "runtime_symbol_quality_artifacts",
        "authority": {
            "runtime_authority": False,
            "promotion_authority": False,
            "live_money_authority": False,
        },
        "status": {
            "available": bool(rows),
            "status": status,
            "mode": "observe",
            "reason": "ok" if rows else "no_symbol_samples",
        },
        "thresholds": {
            "min_samples": int(max(1, min_samples)),
            "min_persistence": int(max(1, min_persistence)),
            "shadow_total_cost_bps": float(shadow_total_cost_bps),
            "disable_total_cost_bps": float(disable_total_cost_bps),
            "shadow_markout_bps": float(shadow_markout_bps),
            "disable_markout_bps": float(disable_markout_bps),
            "shadow_spread_bps": float(shadow_spread_bps),
            "disable_spread_bps": float(disable_spread_bps),
        },
        "summary": {
            "symbol_count": int(len(rows)),
            "disabled_count": int(len(disabled)),
            "shadow_only_count": int(len(shadow_only)),
            "allow_count": int(
                sum(1 for row in rows if row.get("effective_mode") == "allow")
            ),
        },
        "policy": {
            "disabled_symbols": disabled,
            "shadow_only_symbols": shadow_only,
            "allowed_symbols": [
                str(row["symbol"]) for row in rows if row.get("effective_mode") == "allow"
            ],
        },
        "shadow_promotion": _shadow_promotion_suggestions(
            rows,
            executable_symbols=executable_set,
            shadow_symbols=shadow_set,
            min_score_delta=float(shadow_promotion_min_score_delta),
            min_samples=int(shadow_promotion_min_samples),
        ),
        "diagnostics": _universe_diagnostics(
            rows,
            executable_symbols=executable_set,
            shadow_symbols=shadow_set,
            starvation_threshold=float(starvation_threshold),
        )
        | {
            "zero_evidence_exploration_enabled": bool(include_zero_evidence),
            "zero_evidence_exploration_symbols": zero_evidence_exploration_symbols,
            "paper_sampling_symbols": sorted(paper_sampling_symbols),
        },
        "symbols": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-cost-model-json", default="")
    parser.add_argument("--shadow-report-json", default="")
    parser.add_argument("--execution-quality-governor-json", default="")
    parser.add_argument("--replay-report-json", default="")
    parser.add_argument("--previous-scorecard-json", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--min-samples", type=int, default=25)
    parser.add_argument("--min-persistence", type=int, default=2)
    parser.add_argument("--shadow-total-cost-bps", type=float, default=20.0)
    parser.add_argument("--disable-total-cost-bps", type=float, default=35.0)
    parser.add_argument("--shadow-markout-bps", type=float, default=-10.0)
    parser.add_argument("--disable-markout-bps", type=float, default=-25.0)
    parser.add_argument("--shadow-spread-bps", type=float, default=35.0)
    parser.add_argument("--disable-spread-bps", type=float, default=60.0)
    parser.add_argument("--executable-symbols", default="")
    parser.add_argument("--shadow-symbols", default="")
    parser.add_argument("--shadow-promotion-min-score-delta", type=float, default=0.5)
    parser.add_argument("--shadow-promotion-min-samples", type=int, default=10)
    parser.add_argument("--starvation-threshold", type=float, default=0.95)
    args = parser.parse_args(argv)

    live_cost_path = _path_arg(
        args.live_cost_model_json,
        env_key="AI_TRADING_LIVE_COST_MODEL_PATH",
        default_relative="runtime/live_cost_model_latest.json",
    )
    shadow_path = Path(args.shadow_report_json).expanduser() if str(args.shadow_report_json or "").strip() else None
    replay_path = Path(args.replay_report_json).expanduser() if str(args.replay_report_json or "").strip() else None
    governor_path = _path_arg(
        args.execution_quality_governor_json,
        env_key="AI_TRADING_EXECUTION_QUALITY_GOVERNOR_REPORT_PATH",
        default_relative="runtime/execution_quality_governor_latest.json",
    )
    output_path = _path_arg(
        args.output_json,
        env_key="AI_TRADING_SYMBOL_UNIVERSE_SCORECARD_PATH",
        default_relative="runtime/symbol_universe_scorecard_latest.json",
    )
    previous_path = (
        Path(args.previous_scorecard_json).expanduser()
        if str(args.previous_scorecard_json or "").strip()
        else output_path
    )
    report = build_symbol_universe_scorecard(
        live_cost_model=_read_json(live_cost_path),
        shadow_report=_read_json(shadow_path),
        execution_quality_governor=_read_json(governor_path),
        replay_report=_read_json(replay_path),
        previous_scorecard=_read_json(previous_path),
        min_samples=max(1, int(args.min_samples)),
        min_persistence=max(1, int(args.min_persistence)),
        shadow_total_cost_bps=float(args.shadow_total_cost_bps),
        disable_total_cost_bps=float(args.disable_total_cost_bps),
        shadow_markout_bps=float(args.shadow_markout_bps),
        disable_markout_bps=float(args.disable_markout_bps),
        shadow_spread_bps=float(args.shadow_spread_bps),
        disable_spread_bps=float(args.disable_spread_bps),
        executable_symbols=_symbol_set(
            args.executable_symbols
            or get_env("AI_TRADING_CANARY_SYMBOLS", "", cast=str, resolve_aliases=False)
        ),
        shadow_symbols=_symbol_set(
            args.shadow_symbols
            or get_env("AI_TRADING_ML_SHADOW_EXTRA_SYMBOLS", "", cast=str, resolve_aliases=False)
        ),
        shadow_promotion_min_score_delta=float(args.shadow_promotion_min_score_delta),
        shadow_promotion_min_samples=max(1, int(args.shadow_promotion_min_samples)),
        starvation_threshold=float(args.starvation_threshold),
    )
    report["paths"] = {
        "live_cost_model": str(live_cost_path),
        "shadow_report": str(shadow_path) if shadow_path is not None else None,
        "replay_report": str(replay_path) if replay_path is not None else None,
        "execution_quality_governor": str(governor_path),
        "previous_scorecard": str(previous_path),
        "report": str(output_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(
        json.dumps(
            {"path": str(output_path), "status": report["status"], "summary": report["summary"]},
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
