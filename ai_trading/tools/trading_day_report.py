"""Build a human-readable trading-day attribution report."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _read_jsonl(
    path: Path | None,
    *,
    report_date: str | None = None,
    max_rows: int = 200_000,
) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                if report_date and not _date_match(parsed, report_date):
                    continue
                rows.append(parsed)
                if len(rows) >= max(1, int(max_rows)):
                    rows.pop(0)
    return rows


def _date_match(row: Mapping[str, Any], report_date: str) -> bool:
    ts = str(row.get("ts") or row.get("timestamp") or row.get("decision_ts") or "")
    return ts.startswith(report_date)


def build_trading_day_report(
    *,
    report_date: str,
    order_intents: Sequence[Mapping[str, Any]],
    fills: Sequence[Mapping[str, Any]],
    shadow_rows: Sequence[Mapping[str, Any]],
    gate_rows: Sequence[Mapping[str, Any]],
    live_cost_model: Mapping[str, Any],
    symbol_scorecard: Mapping[str, Any],
) -> dict[str, Any]:
    intents = [row for row in order_intents if _date_match(row, report_date)]
    fill_rows = [row for row in fills if _date_match(row, report_date)]
    shadows = [row for row in shadow_rows if _date_match(row, report_date)]
    gates = [row for row in gate_rows if _date_match(row, report_date)]
    rejected = [row for row in gates if str(row.get("action") or row.get("status") or "").lower() in {"reject", "rejected", "blocked"}]
    reject_reasons = Counter(str(row.get("reason") or row.get("gate") or "unknown") for row in rejected)
    symbol_pnl: dict[str, float] = defaultdict(float)
    symbol_realized_edge_bps: dict[str, float] = defaultdict(float)
    symbol_expected_edge_bps: dict[str, float] = defaultdict(float)
    symbol_slippage_bps: dict[str, float] = defaultdict(float)
    for row in fill_rows:
        symbol = str(row.get("symbol") or "").upper() or "UNKNOWN"
        pnl = row.get("pnl") if row.get("pnl") is not None else row.get("realized_pnl")
        try:
            symbol_pnl[symbol] += float(pnl or 0.0)
        except (TypeError, ValueError):
            pass
        for source_key, target in (
            ("realized_net_edge_bps", symbol_realized_edge_bps),
            ("expected_net_edge_bps", symbol_expected_edge_bps),
            ("slippage_bps", symbol_slippage_bps),
        ):
            try:
                target[symbol] += float(row.get(source_key) or 0.0)
            except (TypeError, ValueError):
                pass
    return {
        "schema_version": "1.0.0",
        "artifact_type": "trading_day_report",
        "report_date": report_date,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "desired_trades": {"count": len(intents)},
        "submitted_trades": {"count": len([row for row in intents if str(row.get("status") or "").upper() in {"SUBMITTED", "FILLED"}])},
        "rejected_trades": {
            "count": len(rejected),
            "reasons": dict(reject_reasons),
        },
        "realized_fills": {"count": len(fill_rows)},
        "slippage_spread_cost": {
            "live_cost_status": live_cost_model.get("status", {}),
            "summary": live_cost_model.get("summary", {}),
        },
        "estimated_edge_vs_realized": {
            "shadow_rows": len(shadows),
            "fill_rows": len(fill_rows),
        },
        "symbol_contribution": dict(sorted(symbol_pnl.items())),
        "symbol_realized_edge_bps": dict(sorted(symbol_realized_edge_bps.items())),
        "symbol_expected_edge_bps": dict(sorted(symbol_expected_edge_bps.items())),
        "symbol_slippage_bps": dict(sorted(symbol_slippage_bps.items())),
        "gate_effectiveness": {"rejected_by_gate": dict(reject_reasons)},
        "missed_opportunities": {
            "shadow_only_count": sum(1 for row in shadows if bool(row.get("challenger_would_trade")) and not bool(row.get("champion_would_trade"))),
        },
        "symbol_scorecard": {
            "summary": symbol_scorecard.get("summary", {}),
            "symbols": symbol_scorecard.get("symbols", []),
        },
        "next_session_recommendation": "review_live_capital_readiness_before_live_trading",
    }


def _default_report_paths(report_date: str) -> tuple[Path, Path, Path]:
    root = resolve_runtime_artifact_path(
        "runtime/reports",
        default_relative="runtime/reports",
        for_write=True,
    )
    compact = report_date.replace("-", "")
    return (
        root / f"trading_day_{compact}.json",
        root / "trading_day_latest.json",
        root / "trading_day_latest.md",
    )


def _markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"# Trading Day {report.get('report_date')}",
            "",
            f"- Desired trades: `{report.get('desired_trades', {}).get('count', 0)}`",
            f"- Submitted trades: `{report.get('submitted_trades', {}).get('count', 0)}`",
            f"- Rejected trades: `{report.get('rejected_trades', {}).get('count', 0)}`",
            f"- Realized fills: `{report.get('realized_fills', {}).get('count', 0)}`",
            f"- Next session: `{report.get('next_session_recommendation')}`",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--order-intents-jsonl", type=Path, default=None)
    parser.add_argument("--fills-jsonl", type=Path, default=None)
    parser.add_argument("--shadow-jsonl", type=Path, default=None)
    parser.add_argument("--gate-jsonl", type=Path, default=None)
    parser.add_argument("--live-cost-model-json", type=Path, default=None)
    parser.add_argument("--symbol-scorecard-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--latest-json", type=Path, default=None)
    parser.add_argument("--latest-md", type=Path, default=None)
    args = parser.parse_args(argv)
    output_json, latest_json, latest_md = _default_report_paths(str(args.report_date))
    output_json = args.output_json or output_json
    latest_json = args.latest_json or latest_json
    latest_md = args.latest_md or latest_md
    report = build_trading_day_report(
        report_date=str(args.report_date),
        order_intents=_read_jsonl(args.order_intents_jsonl, report_date=str(args.report_date)),
        fills=_read_jsonl(args.fills_jsonl, report_date=str(args.report_date)),
        shadow_rows=_read_jsonl(args.shadow_jsonl, report_date=str(args.report_date)),
        gate_rows=_read_jsonl(args.gate_jsonl, report_date=str(args.report_date)),
        live_cost_model=_read_json(args.live_cost_model_json),
        symbol_scorecard=_read_json(args.symbol_scorecard_json),
    )
    for path, content in (
        (output_json, json.dumps(report, indent=2, sort_keys=True) + "\n"),
        (latest_json, json.dumps(report, indent=2, sort_keys=True) + "\n"),
        (latest_md, _markdown(report)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    sys.stdout.write(json.dumps({"path": str(output_json), "status": "written"}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
