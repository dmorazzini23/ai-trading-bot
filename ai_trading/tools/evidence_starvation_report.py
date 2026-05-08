"""Report when paper trading is too throttled to collect useful evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


def _read_jsonl(path: Path | None, *, report_date: str | None = None) -> list[dict[str, Any]]:
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
            if isinstance(parsed, dict) and (not report_date or _date_match(parsed, report_date)):
                rows.append(parsed)
    return rows


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _date_match(row: Mapping[str, Any], report_date: str) -> bool:
    ts = str(row.get("ts") or row.get("timestamp") or row.get("decision_ts") or "")
    return ts.startswith(report_date)


def _symbols(raw: str | Sequence[str] | None) -> list[str]:
    if isinstance(raw, str):
        values = raw.split(",")
    elif raw is None:
        values = []
    else:
        values = [str(value) for value in raw]
    return sorted({value.strip().upper() for value in values if value and value.strip()})


def _symbol(row: Mapping[str, Any]) -> str:
    return str(row.get("symbol") or row.get("ticker") or "UNKNOWN").strip().upper() or "UNKNOWN"


def _submitted(row: Mapping[str, Any]) -> bool:
    status = str(row.get("status") or row.get("action") or "").strip().lower()
    return status in {"submitted", "filled", "accepted", "pending_new", "new"}


def _rejected(row: Mapping[str, Any]) -> bool:
    status = str(row.get("status") or row.get("action") or "").strip().lower()
    return status in {"blocked", "reject", "rejected", "skip", "skipped"}


def build_evidence_starvation_report(
    *,
    report_date: str,
    executable_symbols: Sequence[str] = (),
    shadow_symbols: Sequence[str] = (),
    candidates: Sequence[Mapping[str, Any]] = (),
    order_intents: Sequence[Mapping[str, Any]] = (),
    fills: Sequence[Mapping[str, Any]] = (),
    gate_rows: Sequence[Mapping[str, Any]] = (),
    runtime_gonogo: Mapping[str, Any] | None = None,
    sample_target: int = 150,
    min_daily_fills: int = 3,
) -> dict[str, Any]:
    configured = _symbols(executable_symbols)
    shadow = _symbols(shadow_symbols)
    candidate_symbols = Counter(_symbol(row) for row in candidates if _symbol(row) != "UNKNOWN")
    accepted_decisions = [
        row
        for row in list(candidates) + list(order_intents)
        if _submitted(row)
    ]
    rejected_decisions = [
        row
        for row in list(candidates) + list(gate_rows)
        if _rejected(row)
    ]
    submitted_orders = [row for row in order_intents if _submitted(row)]
    fill_count = len(fills)
    target = max(1, int(sample_target))
    remaining = max(0, target - fill_count)
    daily_rate = max(float(fill_count), 0.0)
    estimated_days = math.ceil(remaining / daily_rate) if daily_rate > 0.0 else None
    hard_blockers: list[str] = []
    gonogo_payload = runtime_gonogo or {}
    if gonogo_payload.get("gate_passed") is False:
        hard_blockers.extend(str(item) for item in gonogo_payload.get("failed_checks", []) or [])
    active_candidate_symbols = sorted(candidate_symbols)
    no_candidate_symbols = sorted(set(configured).difference(active_candidate_symbols))
    dominant_share = 0.0
    dominant_symbol = None
    total_candidates = sum(candidate_symbols.values())
    if total_candidates:
        dominant_symbol, dominant_count = candidate_symbols.most_common(1)[0]
        dominant_share = float(dominant_count / total_candidates)
    if hard_blockers:
        recommendation = "stay_observe_due_to_hard_safety"
        status = "blocked_by_safety"
    elif fill_count >= target:
        recommendation = "keep_sampling"
        status = "sufficient"
    elif fill_count < int(min_daily_fills):
        recommendation = "widen_paper_diagnostic_sampling"
        status = "starved"
    elif no_candidate_symbols:
        recommendation = "add_shadow_symbols"
        status = "symbol_starved"
    elif dominant_share >= 0.95 and len(configured) > 1:
        recommendation = "widen_paper_diagnostic_sampling"
        status = "symbol_concentrated"
    else:
        recommendation = "keep_sampling"
        status = "collecting"
    return {
        "schema_version": "1.0.0",
        "artifact_type": "evidence_starvation_report",
        "report_date": report_date,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": status,
        "recommendation": recommendation,
        "configured_executable_symbols": configured,
        "shadow_symbols": shadow,
        "actual_candidate_symbols": active_candidate_symbols,
        "symbols_without_candidates": no_candidate_symbols,
        "dominant_symbol": dominant_symbol,
        "dominant_sample_share": dominant_share,
        "counts": {
            "candidate_decisions": total_candidates,
            "accepted_decisions": len(accepted_decisions),
            "rejected_decisions": len(rejected_decisions),
            "submitted_orders": len(submitted_orders),
            "fills": fill_count,
            "sample_target": target,
        },
        "estimated_days_until_sample_sufficiency": estimated_days,
        "hard_safety_blockers": hard_blockers,
        "promotion_authority": False,
        "live_money_authority": False,
    }


def _default_report_paths(report_date: str) -> tuple[Path, Path]:
    root = resolve_runtime_artifact_path(
        "runtime/reports",
        default_relative="runtime/reports",
        for_write=True,
    )
    compact = report_date.replace("-", "")
    return root / f"evidence_starvation_{compact}.json", root / "evidence_starvation_latest.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--executable-symbols", default="")
    parser.add_argument("--shadow-symbols", default="")
    parser.add_argument("--candidates-jsonl", type=Path, default=None)
    parser.add_argument("--order-intents-jsonl", type=Path, default=None)
    parser.add_argument("--fills-jsonl", type=Path, default=None)
    parser.add_argument("--gate-jsonl", type=Path, default=None)
    parser.add_argument("--runtime-gonogo-json", type=Path, default=None)
    parser.add_argument("--sample-target", type=int, default=150)
    parser.add_argument("--min-daily-fills", type=int, default=3)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--latest-json", type=Path, default=None)
    args = parser.parse_args(argv)
    output_json, latest_json = _default_report_paths(str(args.report_date))
    output_json = args.output_json or output_json
    latest_json = args.latest_json or latest_json
    gonogo = _read_json(args.runtime_gonogo_json)
    if isinstance(gonogo.get("go_no_go"), Mapping):
        gonogo = dict(gonogo["go_no_go"])
    report = build_evidence_starvation_report(
        report_date=str(args.report_date),
        executable_symbols=_symbols(str(args.executable_symbols)),
        shadow_symbols=_symbols(str(args.shadow_symbols)),
        candidates=_read_jsonl(args.candidates_jsonl, report_date=str(args.report_date)),
        order_intents=_read_jsonl(args.order_intents_jsonl, report_date=str(args.report_date)),
        fills=_read_jsonl(args.fills_jsonl, report_date=str(args.report_date)),
        gate_rows=_read_jsonl(args.gate_jsonl, report_date=str(args.report_date)),
        runtime_gonogo=gonogo,
        sample_target=int(args.sample_target),
        min_daily_fills=int(args.min_daily_fills),
    )
    for path in (output_json, latest_json):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps({"path": str(output_json), "status": report["status"]}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
