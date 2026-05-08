"""Build an advisory symbol-promotion comparison artifact.

This tool reads existing research/runtime artifacts only. It does not mutate
runtime symbol gates, launch profiles, or promotion authority.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from ai_trading.runtime.artifacts import resolve_runtime_artifact_path

_DEFAULT_SYMBOLS = ("AMZN", "AAPL", "MSFT")
_RECOMMENDATIONS = {
    "keep_allow",
    "keep_canary",
    "move_to_shadow_only",
    "consider_promotion",
    "disable",
    "collect_more_evidence",
}


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


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
    clean = [float(value) for value in values if value is not None]
    return max(clean) if clean else None


def _symbol_set(raw: str | Iterable[str] | None) -> set[str]:
    if raw is None:
        return set()
    if isinstance(raw, str):
        tokens = raw.replace(";", ",").split(",")
    else:
        tokens = [str(item) for item in raw]
    return {token.strip().upper() for token in tokens if token and token.strip()}


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
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    if isinstance(buckets, list):
        for row in buckets:
            if not isinstance(row, Mapping):
                continue
            symbol = str(row.get("symbol") or "").strip().upper()
            if symbol:
                grouped.setdefault(symbol, []).append(row)
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
        }
    return out


def _markout_rows(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
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


def _shadow_cost_rows(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    cost_breakdowns = payload.get("cost_breakdowns")
    if not isinstance(cost_breakdowns, Mapping):
        return {}
    rows = cost_breakdowns.get("by_symbol")
    out: dict[str, Mapping[str, Any]] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        symbol = str(row.get("symbol") or row.get("key") or "").strip().upper()
        if symbol:
            out[symbol] = row
    return out


def _replay_by_symbol(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = payload.get("replay_symbol_summary")
    if isinstance(rows, Mapping):
        return {
            str(symbol).strip().upper(): row
            for symbol, row in rows.items()
            if str(symbol).strip() and isinstance(row, Mapping)
        }
    for key in ("symbols", "by_symbol"):
        mapped = _symbol_row_map(payload.get(key))
        if mapped:
            return mapped
    return {}


def _scorecard_by_symbol(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return _symbol_row_map(payload.get("symbols"))


def _trading_day_by_symbol(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    flow = payload.get("symbol_trade_flow")
    out: dict[str, dict[str, Any]] = {}
    if isinstance(flow, Mapping):
        for symbol, row in flow.items():
            symbol_key = str(symbol or "").strip().upper()
            if symbol_key and isinstance(row, Mapping):
                out[symbol_key] = {
                    "desired": _to_int(row.get("desired")),
                    "submitted": _to_int(row.get("submitted")),
                    "rejected": _to_int(row.get("rejected")),
                    "fills": _to_int(row.get("fills")),
                }
    for symbol, pnl in (payload.get("symbol_contribution") or {}).items():
        symbol_key = str(symbol or "").strip().upper()
        if symbol_key:
            out.setdefault(symbol_key, {})["pnl"] = _to_float(pnl)
    for source_key, target_key in (
        ("symbol_slippage_bps", "slippage_bps"),
        ("symbol_realized_edge_bps", "realized_edge_bps"),
        ("symbol_expected_edge_bps", "expected_edge_bps"),
    ):
        rows = payload.get(source_key)
        if not isinstance(rows, Mapping):
            continue
        for symbol, value in rows.items():
            symbol_key = str(symbol or "").strip().upper()
            if symbol_key:
                out.setdefault(symbol_key, {})[target_key] = _to_float(value)
    return out


def _current_mode(
    symbol: str,
    *,
    scorecard_row: Mapping[str, Any] | None,
    canary_symbols: set[str],
    allow_symbols: set[str],
    shadow_symbols: set[str],
) -> str:
    if symbol in canary_symbols:
        return "canary"
    if symbol in shadow_symbols:
        return "shadow_only"
    if symbol in allow_symbols:
        return "allow"
    if isinstance(scorecard_row, Mapping):
        mode = str(
            scorecard_row.get("effective_mode") or scorecard_row.get("recommended_mode") or ""
        ).strip().lower()
        if mode == "disabled":
            return "disabled"
        if mode == "shadow_only":
            return "shadow_only"
        if mode == "allow":
            return "allow"
    return "unknown"


def _metric_count(
    *,
    live: Mapping[str, Any],
    shadow: Mapping[str, Any],
    replay: Mapping[str, Any],
    trading_day: Mapping[str, Any],
) -> int:
    keys = [
        live.get("p90_total_cost_bps"),
        live.get("p90_spread_bps"),
        shadow.get("mean_net_markout_bps"),
        shadow.get("positive_rate"),
        replay.get("net_edge_bps"),
        replay.get("win_rate"),
        replay.get("profit_factor"),
        trading_day.get("fills"),
        trading_day.get("rejected"),
        trading_day.get("slippage_bps"),
        trading_day.get("pnl"),
    ]
    return sum(1 for value in keys if value is not None)


def _recommendation(
    *,
    current_mode: str,
    sample_count: int,
    metric_count: int,
    min_samples: int,
    live: Mapping[str, Any],
    shadow: Mapping[str, Any],
    replay: Mapping[str, Any],
    trading_day: Mapping[str, Any],
    scorecard_row: Mapping[str, Any] | None,
) -> tuple[str, list[str], str, dict[str, Any]]:
    reasons: list[str] = []
    sufficient = sample_count >= max(1, int(min_samples))
    if not sufficient:
        reasons.append("insufficient_samples")
    if metric_count <= 1:
        reasons.append("limited_metric_coverage")

    negative = False
    severe = False
    positive = False

    p90_total_cost = _to_float(live.get("p90_total_cost_bps"))
    p90_spread = _to_float(live.get("p90_spread_bps"))
    mean_markout = _to_float(shadow.get("mean_net_markout_bps"))
    if mean_markout is None:
        mean_markout = _to_float(replay.get("net_edge_bps"))
    win_rate = _to_float(replay.get("win_rate"))
    if win_rate is None:
        win_rate = _to_float(shadow.get("positive_rate"))
    profit_factor = _to_float(replay.get("profit_factor"))
    rejected = _to_int(trading_day.get("rejected"))
    fills = _to_int(trading_day.get("fills"))
    slippage = _to_float(trading_day.get("slippage_bps"))
    pnl = _to_float(trading_day.get("pnl"))

    if p90_total_cost is not None:
        if p90_total_cost >= 35.0:
            severe = True
            reasons.append("p90_total_cost_bps_disable")
        elif p90_total_cost >= 20.0:
            negative = True
            reasons.append("p90_total_cost_bps_high")
    if p90_spread is not None:
        if p90_spread >= 60.0:
            severe = True
            reasons.append("p90_spread_bps_disable")
        elif p90_spread >= 35.0:
            negative = True
            reasons.append("p90_spread_bps_high")
    if mean_markout is not None:
        if mean_markout <= -25.0:
            severe = True
            reasons.append("markout_disable")
        elif mean_markout <= -10.0:
            negative = True
            reasons.append("markout_negative")
        elif mean_markout >= 2.0:
            positive = True
            reasons.append("markout_positive")
    if win_rate is not None:
        if win_rate < 0.40:
            negative = True
            reasons.append("win_rate_low")
        elif win_rate >= 0.55:
            positive = True
            reasons.append("win_rate_positive")
    if profit_factor is not None:
        if profit_factor < 0.80:
            negative = True
            reasons.append("profit_factor_low")
        elif profit_factor >= 1.20:
            positive = True
            reasons.append("profit_factor_positive")
    if fills > 0 and rejected > fills * 2:
        negative = True
        reasons.append("rejections_exceed_fills")
    if slippage is not None and slippage >= 15.0:
        negative = True
        reasons.append("slippage_high")
    if pnl is not None:
        if pnl < 0.0:
            negative = True
            reasons.append("realized_pnl_negative")
        elif pnl > 0.0:
            positive = True
            reasons.append("realized_pnl_positive")
    if isinstance(scorecard_row, Mapping):
        effective_mode = str(scorecard_row.get("effective_mode") or "").lower()
        if effective_mode == "disabled":
            severe = True
            reasons.append("scorecard_disabled")
        elif effective_mode == "shadow_only":
            negative = True
            reasons.append("scorecard_shadow_only")

    if not sufficient or metric_count <= 1:
        recommendation = "collect_more_evidence"
    elif severe:
        recommendation = "disable"
    elif negative:
        recommendation = "move_to_shadow_only"
    elif positive and current_mode == "shadow_only":
        recommendation = "consider_promotion"
    elif current_mode == "canary":
        recommendation = "keep_canary"
    elif current_mode == "allow":
        recommendation = "keep_allow"
    else:
        recommendation = "collect_more_evidence"

    if recommendation not in _RECOMMENDATIONS:
        recommendation = "collect_more_evidence"
    if not reasons:
        reasons.append("evidence_healthy")
    confidence = "low"
    if sufficient and metric_count >= 4:
        confidence = "high" if recommendation != "collect_more_evidence" else "medium"
    elif sufficient and metric_count >= 2:
        confidence = "medium"
    sufficiency = {
        "sufficient": bool(sufficient),
        "sample_count": int(sample_count),
        "required_min_samples": int(max(1, min_samples)),
        "metric_count": int(metric_count),
    }
    return recommendation, reasons, confidence, sufficiency


def build_symbol_promotion_comparison(
    *,
    report_date: str,
    symbols: Iterable[str] = _DEFAULT_SYMBOLS,
    live_cost_model: Mapping[str, Any] | None = None,
    replay_report: Mapping[str, Any] | None = None,
    shadow_report: Mapping[str, Any] | None = None,
    trading_day_report: Mapping[str, Any] | None = None,
    symbol_scorecard: Mapping[str, Any] | None = None,
    canary_symbols: Iterable[str] | None = None,
    allow_symbols: Iterable[str] | None = None,
    shadow_symbols: Iterable[str] | None = None,
    min_samples: int = 25,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return a read-only advisory comparison for the requested symbols."""

    generated_at = now.astimezone(UTC) if now is not None else datetime.now(UTC)
    target_symbols = sorted(_symbol_set(symbols))
    live_by_symbol = _live_cost_by_symbol(live_cost_model or {})
    replay_by_symbol = _replay_by_symbol(replay_report or {})
    markout_by_symbol = _markout_rows(shadow_report or {})
    shadow_cost_by_symbol = _shadow_cost_rows(shadow_report or {})
    trading_by_symbol = _trading_day_by_symbol(trading_day_report or {})
    scorecard_by_symbol = _scorecard_by_symbol(symbol_scorecard or {})
    scorecard_policy = (
        symbol_scorecard.get("policy") if isinstance(symbol_scorecard, Mapping) else {}
    )
    policy_allow = (
        _symbol_set(scorecard_policy.get("allowed_symbols"))
        if isinstance(scorecard_policy, Mapping)
        else set()
    )
    policy_shadow = (
        _symbol_set(scorecard_policy.get("shadow_only_symbols"))
        if isinstance(scorecard_policy, Mapping)
        else set()
    )
    policy_disabled = (
        _symbol_set(scorecard_policy.get("disabled_symbols"))
        if isinstance(scorecard_policy, Mapping)
        else set()
    )
    canary_set = _symbol_set(canary_symbols)
    allow_set = _symbol_set(allow_symbols) | policy_allow
    shadow_set = _symbol_set(shadow_symbols) | policy_shadow
    rows: list[dict[str, Any]] = []
    for symbol in target_symbols:
        live = live_by_symbol.get(symbol, {})
        replay = replay_by_symbol.get(symbol, {})
        markout = markout_by_symbol.get(symbol, {})
        shadow_cost = shadow_cost_by_symbol.get(symbol, {})
        trading_day = trading_by_symbol.get(symbol, {})
        scorecard_row = scorecard_by_symbol.get(symbol)
        current_mode = (
            "disabled"
            if symbol in policy_disabled
            else _current_mode(
                symbol,
                scorecard_row=scorecard_row,
                canary_symbols=canary_set,
                allow_symbols=allow_set,
                shadow_symbols=shadow_set,
            )
        )
        replay_samples = max(
            _to_int(replay.get("sample_count")),
            _to_int(replay.get("samples")),
            _to_int(replay.get("closed_trades")),
        )
        shadow_samples = max(
            _to_int(markout.get("samples")),
            _to_int(markout.get("sample_count")),
            _to_int(shadow_cost.get("rows")),
        )
        live_samples = _to_int(live.get("sample_count"))
        trading_samples = max(_to_int(trading_day.get("fills")), _to_int(trading_day.get("desired")))
        scorecard_samples = _to_int(scorecard_row.get("sample_count")) if isinstance(scorecard_row, Mapping) else 0
        sample_count = max(
            live_samples,
            replay_samples,
            shadow_samples,
            trading_samples,
            scorecard_samples,
        )
        live_metrics = {
            "sample_count": int(live_samples),
            "mean_total_cost_bps": _to_float(live.get("mean_total_cost_bps")),
            "p90_total_cost_bps": _to_float(live.get("p90_total_cost_bps")),
            "mean_spread_bps": _to_float(live.get("mean_spread_bps")),
            "p90_spread_bps": _to_float(live.get("p90_spread_bps")),
            "mean_quote_age_ms": _to_float(live.get("mean_quote_age_ms")),
        }
        replay_metrics = {
            "sample_count": int(replay_samples),
            "net_edge_bps": _to_float(replay.get("net_edge_bps")),
            "win_rate": _to_float(replay.get("win_rate")),
            "profit_factor": _to_float(replay.get("profit_factor")),
        }
        shadow_metrics = {
            "samples": int(shadow_samples),
            "mean_net_markout_bps": _to_float(markout.get("mean_net_markout_bps")),
            "positive_rate": _to_float(markout.get("positive_rate")),
            "mean_spread_bps": _to_float(shadow_cost.get("mean_spread_bps")),
            "p90_spread_bps": _to_float(shadow_cost.get("p90_spread_bps")),
            "mean_quote_age_ms": _to_float(shadow_cost.get("mean_quote_age_ms")),
        }
        trading_metrics = {
            "desired": _to_int(trading_day.get("desired")),
            "submitted": _to_int(trading_day.get("submitted")),
            "rejected": _to_int(trading_day.get("rejected")),
            "fills": _to_int(trading_day.get("fills")),
            "slippage_bps": _to_float(trading_day.get("slippage_bps")),
            "realized_edge_bps": _to_float(trading_day.get("realized_edge_bps")),
            "expected_edge_bps": _to_float(trading_day.get("expected_edge_bps")),
            "pnl": _to_float(trading_day.get("pnl")),
        }
        metric_count = _metric_count(
            live=live_metrics,
            shadow=shadow_metrics,
            replay=replay_metrics,
            trading_day=trading_metrics,
        )
        recommendation, reasons, confidence, sufficiency = _recommendation(
            current_mode=current_mode,
            sample_count=sample_count,
            metric_count=metric_count,
            min_samples=max(1, int(min_samples)),
            live=live_metrics,
            shadow=shadow_metrics,
            replay=replay_metrics,
            trading_day=trading_metrics,
            scorecard_row=scorecard_row,
        )
        rows.append(
            {
                "symbol": symbol,
                "current_mode": current_mode,
                "recommendation": recommendation,
                "confidence": confidence,
                "sample_sufficiency": sufficiency,
                "reasons": reasons,
                "metrics": {
                    "live_cost": live_metrics,
                    "replay": replay_metrics,
                    "shadow": shadow_metrics,
                    "trading_day": trading_metrics,
                },
            }
        )
    counts = Counter(str(row.get("recommendation") or "unknown") for row in rows)
    return {
        "schema_version": "1.0.0",
        "artifact_type": "symbol_promotion_comparison",
        "report_date": report_date,
        "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
        "source": "generated_artifacts_only",
        "promotion_authority": False,
        "runtime_symbol_gating_changed": False,
        "status": "ready" if rows else "unavailable",
        "symbols_requested": target_symbols,
        "thresholds": {"min_samples": int(max(1, min_samples))},
        "summary": {
            "symbol_count": int(len(rows)),
            "recommendations": dict(sorted(counts.items())),
            "sample_sufficient_count": int(
                sum(1 for row in rows if row.get("sample_sufficiency", {}).get("sufficient"))
            ),
        },
        "symbols": rows,
    }


def symbol_promotion_digest(payload: Mapping[str, Any], *, limit: int = 5) -> str:
    rows = payload.get("symbols")
    if not isinstance(rows, list) or not rows:
        return "none"
    parts: list[str] = []
    for row in rows[: max(1, int(limit))]:
        if not isinstance(row, Mapping):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        recommendation = str(row.get("recommendation") or "n/a")
        confidence = str(row.get("confidence") or "n/a")
        if symbol:
            parts.append(f"{symbol}:{recommendation}/{confidence}")
    return ", ".join(parts) if parts else "none"


def _default_report_root() -> Path:
    return resolve_runtime_artifact_path(
        "runtime/research_reports",
        default_relative="runtime/research_reports",
        for_write=True,
    )


def _default_outputs(report_date: str) -> tuple[Path, Path]:
    root = _default_report_root()
    compact = str(report_date).replace("-", "")
    return (
        root / "symbol_promotion" / f"symbol_promotion_{compact}.json",
        root / "latest" / "symbol_promotion_latest.json",
    )


def _default_path(relative: str) -> Path:
    return resolve_runtime_artifact_path(relative, default_relative=relative)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--symbols", default=",".join(_DEFAULT_SYMBOLS))
    parser.add_argument("--live-cost-model-json", type=Path, default=None)
    parser.add_argument("--replay-report-json", type=Path, default=None)
    parser.add_argument("--shadow-report-json", type=Path, default=None)
    parser.add_argument("--trading-day-json", type=Path, default=None)
    parser.add_argument("--symbol-scorecard-json", type=Path, default=None)
    parser.add_argument("--canary-symbols", default="")
    parser.add_argument("--allow-symbols", default="")
    parser.add_argument("--shadow-symbols", default="")
    parser.add_argument("--min-samples", type=int, default=25)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--latest-json", type=Path, default=None)
    args = parser.parse_args(argv)

    output_json, latest_json = _default_outputs(str(args.report_date))
    output_json = args.output_json or output_json
    latest_json = args.latest_json or latest_json
    report = build_symbol_promotion_comparison(
        report_date=str(args.report_date),
        symbols=_symbol_set(str(args.symbols)),
        live_cost_model=_read_json(
            args.live_cost_model_json or _default_path("runtime/live_cost_model_latest.json")
        ),
        replay_report=_read_json(
            args.replay_report_json or _default_path("runtime/replay_governance_refresh_latest.json")
        ),
        shadow_report=_read_json(
            args.shadow_report_json or _default_path("runtime/ml_shadow_report_latest.json")
        ),
        trading_day_report=_read_json(
            args.trading_day_json
            or _default_path("runtime/research_reports/latest/trading_day_latest.json")
        ),
        symbol_scorecard=_read_json(
            args.symbol_scorecard_json or _default_path("runtime/symbol_universe_scorecard_latest.json")
        ),
        canary_symbols=_symbol_set(str(args.canary_symbols)),
        allow_symbols=_symbol_set(str(args.allow_symbols)),
        shadow_symbols=_symbol_set(str(args.shadow_symbols)),
        min_samples=max(1, int(args.min_samples)),
    )
    report["paths"] = {"report": str(output_json), "latest": str(latest_json)}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    latest_json.parent.mkdir(parents=True, exist_ok=True)
    latest_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(
        json.dumps(
            {
                "path": str(output_json),
                "latest": str(latest_json),
                "status": report["status"],
                "summary": report["summary"],
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
