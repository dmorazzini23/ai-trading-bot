"""Produce manual-only symbol lifecycle recommendations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from ai_trading.runtime.artifacts import resolve_runtime_artifact_path
from ai_trading.tools.symbol_promotion_comparison import build_symbol_promotion_comparison


_AUTHORITY_ORDER = {"disabled": 0, "shadow_only": 1, "canary": 2, "restricted": 3, "allow": 4}


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _symbol_set(raw: str | Iterable[str] | None) -> set[str]:
    if raw is None:
        return set()
    tokens = raw.replace(";", ",").split(",") if isinstance(raw, str) else [str(item) for item in raw]
    return {token.strip().upper() for token in tokens if token and token.strip()}


def _lifecycle_recommendation(row: Mapping[str, Any]) -> tuple[str, str]:
    current = str(row.get("current_mode") or "shadow_only").lower()
    advisory = str(row.get("recommendation") or "collect_more_evidence")
    if advisory == "consider_promotion":
        target = "canary" if current in {"disabled", "shadow_only", "unknown"} else "allow"
        return "consider_canary" if target == "canary" else "consider_allow", target
    if advisory == "disable":
        return "disable", "disabled"
    if advisory == "move_to_shadow_only":
        return "move_to_shadow_only", "shadow_only"
    if advisory == "keep_canary":
        return "keep_canary", "canary"
    if advisory == "keep_allow":
        return "keep_allow", "allow"
    if current == "disabled":
        return "keep_disabled", "disabled"
    if current == "restricted":
        return "restrict", "restricted"
    return "collect_more_evidence", current if current in _AUTHORITY_ORDER else "shadow_only"


def build_symbol_lifecycle_report(
    *,
    report_date: str,
    symbols: Iterable[str],
    live_cost_model: Mapping[str, Any] | None = None,
    replay_report: Mapping[str, Any] | None = None,
    shadow_report: Mapping[str, Any] | None = None,
    trading_day_report: Mapping[str, Any] | None = None,
    symbol_scorecard: Mapping[str, Any] | None = None,
    canary_symbols: Iterable[str] | None = None,
    allow_symbols: Iterable[str] | None = None,
    shadow_symbols: Iterable[str] | None = None,
    min_samples: int = 25,
) -> dict[str, Any]:
    comparison = build_symbol_promotion_comparison(
        report_date=report_date,
        symbols=symbols,
        live_cost_model=live_cost_model,
        replay_report=replay_report,
        shadow_report=shadow_report,
        trading_day_report=trading_day_report,
        symbol_scorecard=symbol_scorecard,
        canary_symbols=canary_symbols,
        allow_symbols=allow_symbols,
        shadow_symbols=shadow_symbols,
        min_samples=min_samples,
    )
    rows: list[dict[str, Any]] = []
    for raw in comparison.get("symbols", []):
        if not isinstance(raw, Mapping):
            continue
        recommendation, target_mode = _lifecycle_recommendation(raw)
        current_mode = str(raw.get("current_mode") or "shadow_only").lower()
        requires_manual_approval = _AUTHORITY_ORDER.get(target_mode, 1) > _AUTHORITY_ORDER.get(current_mode, 1)
        rows.append(
            {
                "symbol": raw.get("symbol"),
                "current_mode": current_mode,
                "recommended_mode": target_mode,
                "recommendation": recommendation,
                "confidence": raw.get("confidence"),
                "reasons": list(raw.get("reasons", [])) if isinstance(raw.get("reasons"), list) else [],
                "sample_sufficiency": dict(raw.get("sample_sufficiency", {}))
                if isinstance(raw.get("sample_sufficiency"), Mapping)
                else {},
                "manual_approval_required": bool(requires_manual_approval),
                "authority_increase": bool(requires_manual_approval),
                "metrics": dict(raw.get("metrics", {})) if isinstance(raw.get("metrics"), Mapping) else {},
            }
        )
    counts = Counter(str(row["recommendation"]) for row in rows)
    return {
        "schema_version": "1.0.0",
        "artifact_type": "symbol_lifecycle_report",
        "report_date": report_date,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "ready" if rows else "unavailable",
        "runtime_symbol_gating_changed": False,
        "promotion_authority": False,
        "manual_approval_required_for_authority_increase": True,
        "summary": {"symbol_count": len(rows), "recommendations": dict(sorted(counts.items()))},
        "symbols": rows,
    }


def _default_outputs(report_date: str) -> tuple[Path, Path]:
    root = resolve_runtime_artifact_path("runtime/research_reports", default_relative="runtime/research_reports", for_write=True)
    compact = report_date.replace("-", "")
    return root / "symbol_lifecycle" / f"symbol_lifecycle_{compact}.json", root / "latest" / "symbol_lifecycle_latest.json"


def _default_path(relative: str) -> Path:
    return resolve_runtime_artifact_path(relative, default_relative=relative)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--symbols", default="AAPL,AMZN,MSFT")
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
    output, latest = _default_outputs(str(args.report_date))
    output = args.output_json or output
    latest = args.latest_json or latest
    report = build_symbol_lifecycle_report(
        report_date=str(args.report_date),
        symbols=_symbol_set(str(args.symbols)),
        live_cost_model=_read_json(args.live_cost_model_json or _default_path("runtime/live_cost_model_latest.json")),
        replay_report=_read_json(args.replay_report_json or _default_path("runtime/replay_governance_refresh_latest.json")),
        shadow_report=_read_json(args.shadow_report_json or _default_path("runtime/ml_shadow_report_latest.json")),
        trading_day_report=_read_json(args.trading_day_json or _default_path("runtime/reports/trading_day_latest.json")),
        symbol_scorecard=_read_json(args.symbol_scorecard_json or _default_path("runtime/symbol_universe_scorecard_latest.json")),
        canary_symbols=_symbol_set(str(args.canary_symbols)),
        allow_symbols=_symbol_set(str(args.allow_symbols)),
        shadow_symbols=_symbol_set(str(args.shadow_symbols)),
        min_samples=int(args.min_samples),
    )
    for path in (output, latest):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps({"path": str(output), "status": report["status"]}, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
