"""Build per-symbol evidence for candidate, order, and fill lifecycle coverage."""

from __future__ import annotations

import argparse
import json
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


def _date_match(row: Mapping[str, Any], report_date: str) -> bool:
    return _timestamp(row).startswith(report_date)


def _timestamp(row: Mapping[str, Any]) -> str:
    return str(row.get("ts") or row.get("timestamp") or row.get("decision_ts") or row.get("submitted_at") or row.get("filled_at") or "")


def _symbols(raw: str | Sequence[str] | None) -> list[str]:
    values = raw.split(",") if isinstance(raw, str) else [str(value) for value in raw or ()]
    return sorted({value.strip().upper() for value in values if value and value.strip()})


def _symbol(row: Mapping[str, Any]) -> str:
    return str(row.get("symbol") or row.get("ticker") or "UNKNOWN").strip().upper() or "UNKNOWN"


def _submitted(row: Mapping[str, Any]) -> bool:
    status = str(row.get("status") or row.get("action") or "").strip().lower()
    return status in {"accepted", "submitted", "filled", "new", "pending_new"}


def _rejected(row: Mapping[str, Any]) -> bool:
    status = str(row.get("status") or row.get("action") or row.get("decision") or "").strip().lower()
    return status in {"reject", "rejected", "blocked", "skip", "skipped"}


def build_symbol_lifecycle_evidence_report(
    *,
    report_date: str,
    symbols: Sequence[str] = (),
    candidates: Sequence[Mapping[str, Any]] = (),
    order_intents: Sequence[Mapping[str, Any]] = (),
    fills: Sequence[Mapping[str, Any]] = (),
    gate_rows: Sequence[Mapping[str, Any]] = (),
    min_fills_per_symbol: int = 1,
) -> dict[str, Any]:
    universe = set(_symbols(symbols))
    universe.update(_symbol(row) for row in list(candidates) + list(order_intents) + list(fills) + list(gate_rows) if _symbol(row) != "UNKNOWN")
    candidate_counts = Counter(_symbol(row) for row in candidates)
    order_counts = Counter(_symbol(row) for row in order_intents if _submitted(row))
    fill_counts = Counter(_symbol(row) for row in fills)
    reject_counts = Counter(_symbol(row) for row in list(candidates) + list(gate_rows) if _rejected(row))
    rows: list[dict[str, Any]] = []
    for symbol in sorted(universe):
        timestamps = [
            _timestamp(row)
            for row in list(candidates) + list(order_intents) + list(fills) + list(gate_rows)
            if _symbol(row) == symbol and _timestamp(row)
        ]
        if candidate_counts[symbol] <= 0:
            lifecycle_status = "no_candidates"
        elif order_counts[symbol] <= 0:
            lifecycle_status = "no_submitted_orders"
        elif fill_counts[symbol] < int(min_fills_per_symbol):
            lifecycle_status = "no_fill_evidence"
        else:
            lifecycle_status = "evidence_ready"
        rows.append(
            {
                "symbol": symbol,
                "candidate_decisions": candidate_counts[symbol],
                "submitted_orders": order_counts[symbol],
                "fills": fill_counts[symbol],
                "rejected_decisions": reject_counts[symbol],
                "first_seen": min(timestamps) if timestamps else None,
                "last_seen": max(timestamps) if timestamps else None,
                "lifecycle_status": lifecycle_status,
            }
        )
    blocked = [row for row in rows if row["lifecycle_status"] != "evidence_ready"]
    status = "complete" if not blocked else "gaps"
    return {
        "schema_version": "1.0.0",
        "artifact_type": "symbol_lifecycle_evidence_report",
        "report_date": report_date,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": status,
        "recommended_next_action": "continue_sampling" if status == "complete" else "inspect_symbols_with_lifecycle_gaps",
        "summary": {
            "symbols": len(rows),
            "evidence_ready": sum(1 for row in rows if row["lifecycle_status"] == "evidence_ready"),
            "gaps": len(blocked),
            "min_fills_per_symbol": int(min_fills_per_symbol),
        },
        "symbols": rows,
        "promotion_authority": False,
        "live_money_authority": False,
    }


def _default_report_paths(report_date: str) -> tuple[Path, Path]:
    root = resolve_runtime_artifact_path("runtime/reports", default_relative="runtime/reports", for_write=True)
    compact = report_date.replace("-", "")
    return root / f"symbol_lifecycle_evidence_{compact}.json", root / "symbol_lifecycle_evidence_latest.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--symbols", default="")
    parser.add_argument("--candidates-jsonl", type=Path, default=None)
    parser.add_argument("--order-intents-jsonl", type=Path, default=None)
    parser.add_argument("--fills-jsonl", type=Path, default=None)
    parser.add_argument("--gate-jsonl", type=Path, default=None)
    parser.add_argument("--min-fills-per-symbol", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--latest-json", type=Path, default=None)
    args = parser.parse_args(argv)
    output_json, latest_json = _default_report_paths(str(args.report_date))
    output_json = args.output_json or output_json
    latest_json = args.latest_json or latest_json
    report = build_symbol_lifecycle_evidence_report(
        report_date=str(args.report_date),
        symbols=_symbols(str(args.symbols)),
        candidates=_read_jsonl(args.candidates_jsonl, report_date=str(args.report_date)),
        order_intents=_read_jsonl(args.order_intents_jsonl, report_date=str(args.report_date)),
        fills=_read_jsonl(args.fills_jsonl, report_date=str(args.report_date)),
        gate_rows=_read_jsonl(args.gate_jsonl, report_date=str(args.report_date)),
        min_fills_per_symbol=int(args.min_fills_per_symbol),
    )
    for path in (output_json, latest_json):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps({"path": str(output_json), "status": report["status"]}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
