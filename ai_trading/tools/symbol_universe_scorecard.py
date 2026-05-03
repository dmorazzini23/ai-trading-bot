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
