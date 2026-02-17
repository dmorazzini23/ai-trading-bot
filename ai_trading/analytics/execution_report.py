"""Daily execution-quality rollups from TCA records."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ai_trading.logging import get_logger

logger = get_logger(__name__)


def _parse_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    src = Path(path)
    if not src.exists():
        return records
    with src.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
    return records


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = max(0.0, min(1.0, pct / 100.0)) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(len(ordered) - 1, lower + 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def build_daily_execution_report(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            str(record.get("symbol", "UNKNOWN")),
            str(record.get("sleeve", "unknown")),
            str(record.get("regime_profile", "unknown")),
            str(record.get("order_type", "unknown")),
            str(record.get("provider", "unknown")),
        )
        grouped[key].append(record)

    groups: list[dict[str, Any]] = []
    reject_count = 0
    cancel_count = 0
    partial_fill_count = 0
    blocked_count = 0
    for key, rows in sorted(grouped.items()):
        symbol, sleeve, regime, order_type, provider = key
        is_values = [float(r["is_bps"]) for r in rows if r.get("is_bps") is not None]
        spread_values = [
            float(r["spread_paid_bps"]) for r in rows if r.get("spread_paid_bps") is not None
        ]
        for row in rows:
            status = str(row.get("status", "")).lower()
            if status == "rejected":
                reject_count += 1
            if status == "canceled":
                cancel_count += 1
            if bool(row.get("partial_fill", False)):
                partial_fill_count += 1
            if bool(row.get("blocked_by_gate", False)):
                blocked_count += 1

        groups.append(
            {
                "symbol": symbol,
                "sleeve": sleeve,
                "regime_profile": regime,
                "order_type": order_type,
                "provider": provider,
                "count": len(rows),
                "is_bps_p50": _percentile(is_values, 50.0),
                "is_bps_p90": _percentile(is_values, 90.0),
                "spread_paid_bps_p50": _percentile(spread_values, 50.0),
                "spread_paid_bps_p90": _percentile(spread_values, 90.0),
            }
        )

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "records": len(records),
        "groups": groups,
        "totals": {
            "rejects": reject_count,
            "cancels": cancel_count,
            "partial_fills": partial_fill_count,
            "blocked_by_gates": blocked_count,
        },
    }
    return report


def _write_csv(path: Path, groups: list[dict[str, Any]]) -> None:
    fieldnames = [
        "symbol",
        "sleeve",
        "regime_profile",
        "order_type",
        "provider",
        "count",
        "is_bps_p50",
        "is_bps_p90",
        "spread_paid_bps_p50",
        "spread_paid_bps_p90",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in groups:
            writer.writerow({name: row.get(name) for name in fieldnames})


def write_daily_execution_report(
    *,
    tca_path: str,
    output_dir: str,
    formats: tuple[str, ...] = ("json", "csv"),
    rollup_tz: str = "UTC",
) -> dict[str, Any]:
    records = _parse_jsonl(tca_path)
    report = build_daily_execution_report(records)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tz_name = str(rollup_tz or "UTC").strip() or "UTC"
    try:
        rollup_zone = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        logger.warning("EXECUTION_REPORT_INVALID_ROLLUP_TZ", extra={"rollup_tz": tz_name})
        rollup_zone = UTC
        tz_name = "UTC"
    day = datetime.now(rollup_zone).strftime("%Y%m%d")

    normalized = {fmt.strip().lower() for fmt in formats}
    if "json" in normalized:
        json_path = out_dir / f"execution_report_{day}.json"
        json_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    if "csv" in normalized:
        csv_path = out_dir / f"execution_report_{day}.csv"
        _write_csv(csv_path, report["groups"])

    logger.info(
        "EXECUTION_REPORT_WRITTEN",
        extra={
            "output_dir": str(out_dir),
            "records": len(records),
            "rollup_tz": tz_name,
            "rollup_day": day,
        },
    )
    return report
