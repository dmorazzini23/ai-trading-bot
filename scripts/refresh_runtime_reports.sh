#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f "${ROOT_DIR}/venv/bin/activate" ]]; then
  echo "missing virtualenv activate script: ${ROOT_DIR}/venv/bin/activate" >&2
  exit 1
fi

source "${ROOT_DIR}/venv/bin/activate"

if [[ -z "${REPORT_DATE:-}" ]]; then
  REPORT_DATE="$(TZ=America/New_York date +%Y-%m-%d)"
fi
export REPORT_DATE

python - <<'PY'
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

from ai_trading.env import ensure_dotenv_loaded
from ai_trading.tools.runtime_performance_report import (
    build_report,
    evaluate_go_no_go,
    resolve_runtime_gonogo_thresholds,
    resolve_runtime_report_paths,
)

REPORT_DATE = os.environ.get("REPORT_DATE", "").strip()
if not REPORT_DATE:
    REPORT_DATE = datetime.now().strftime("%Y-%m-%d")


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _pctl(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    idx = int(round((pct / 100.0) * (len(sorted_values) - 1)))
    idx = max(0, min(len(sorted_values) - 1, idx))
    return sorted_values[idx]


ensure_dotenv_loaded()
paths = resolve_runtime_report_paths()
report = build_report(
    trade_history_path=Path(paths["trade_history"]),
    gate_summary_path=Path(paths["gate_summary"]),
    tca_path=Path(paths["tca"]) if paths.get("tca") else None,
    gate_log_path=Path(paths["gate_log"]) if paths.get("gate_log") else None,
)
report["go_no_go"] = evaluate_go_no_go(
    report,
    thresholds=resolve_runtime_gonogo_thresholds(),
)

runtime_root = Path("/var/lib/ai-trading-bot/runtime")
report_targets = [
    runtime_root / "runtime_performance_report_latest.json",
    runtime_root / "daily_performance_report.json",
    runtime_root / "reports/runtime_performance_report_latest.json",
]
for target in report_targets:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

trade = report.get("trade_history", {})
gate = report.get("gate_effectiveness", {})
trade_day = next(
    (row for row in trade.get("daily_trade_stats", []) if row.get("date") == REPORT_DATE),
    {},
)
expectancy_day = next(
    (row for row in trade.get("daily_expectancy", []) if row.get("date") == REPORT_DATE),
    {},
)
gate_day = next(
    (row for row in gate.get("daily_gate_stats", []) if row.get("date") == REPORT_DATE),
    {},
)

fills_path = runtime_root / "fill_events.jsonl"
fills = _jsonl_rows(fills_path)
fill_rows = 0
fill_side_counts: Counter[str] = Counter()
fill_symbol_counts: Counter[str] = Counter()
fill_slip_values: list[float] = []
fill_symbol_slip_sum: defaultdict[str, float] = defaultdict(float)
for row in fills:
    ts = str(row.get("ts") or "")
    if not ts.startswith(REPORT_DATE):
        continue
    fill_rows += 1
    symbol = str(row.get("symbol") or "UNKNOWN")
    side = str(row.get("side") or "").lower()
    fill_side_counts[side] += 1
    fill_symbol_counts[symbol] += 1
    slip = row.get("slippage_bps")
    if slip is not None:
        try:
            slip_f = float(slip)
        except (TypeError, ValueError):
            continue
        fill_slip_values.append(slip_f)
        fill_symbol_slip_sum[symbol] += slip_f

tca_path = runtime_root / "tca_records.jsonl"
tca_rows = _jsonl_rows(tca_path)
tca_count = 0
tca_filled = 0
tca_terminal_nonfill = 0
tca_is_values: list[float] = []
tca_latency_values: list[float] = []
tca_symbol_is_sum: defaultdict[str, float] = defaultdict(float)
tca_symbol_is_count: Counter[str] = Counter()
for row in tca_rows:
    ts = str(row.get("ts") or "")
    if not ts.startswith(REPORT_DATE):
        continue
    tca_count += 1
    symbol = str(row.get("symbol") or "UNKNOWN")
    qty = row.get("qty") or 0
    fill_price = row.get("fill_price")
    try:
        qty_f = float(qty)
    except (TypeError, ValueError):
        qty_f = 0.0
    if fill_price is not None and qty_f > 0:
        tca_filled += 1
    if bool(row.get("pending_terminal_nonfill")):
        tca_terminal_nonfill += 1
    is_bps = row.get("is_bps")
    if is_bps is not None:
        try:
            is_f = float(is_bps)
        except (TypeError, ValueError):
            is_f = None
        if is_f is not None:
            tca_is_values.append(is_f)
            tca_symbol_is_sum[symbol] += is_f
            tca_symbol_is_count[symbol] += 1
    latency = row.get("fill_latency_ms")
    if latency is not None:
        try:
            tca_latency_values.append(float(latency))
        except (TypeError, ValueError):
            pass

order_path = runtime_root / "order_events.jsonl"
order_rows = _jsonl_rows(order_path)
order_status_counts: Counter[str] = Counter()
order_final_counts: Counter[str] = Counter()
order_latency_values: list[float] = []
for row in order_rows:
    ts = str(row.get("ts") or "")
    if not ts.startswith(REPORT_DATE):
        continue
    event = str(row.get("event") or "").lower()
    new_status = str(row.get("new_status") or "").lower()
    if event == "status_transition":
        order_status_counts[new_status] += 1
    if event == "final_state":
        final_status = str(row.get("final_status") or new_status).lower()
        order_final_counts[final_status] += 1
    latency_ms = row.get("latency_ms")
    if latency_ms is not None:
        try:
            order_latency_values.append(float(latency_ms))
        except (TypeError, ValueError):
            pass

worst_is = []
for symbol, total in tca_symbol_is_sum.items():
    sample_count = tca_symbol_is_count[symbol]
    mean_is = total / sample_count if sample_count else 0.0
    worst_is.append((symbol, mean_is, sample_count, total))
worst_is.sort(key=lambda row: row[1])

symbol_slip_sorted = sorted(fill_symbol_slip_sum.items(), key=lambda row: row[1])

breakdown = {
    "date": REPORT_DATE,
    "daily_trade_stats": trade_day,
    "daily_expectancy": expectancy_day,
    "daily_gate_stats": gate_day,
    "fills": {
        "rows": fill_rows,
        "side_counts": dict(fill_side_counts),
        "slippage_bps_avg": (
            sum(fill_slip_values) / len(fill_slip_values)
            if fill_slip_values
            else None
        ),
        "slippage_bps_p50": _pctl(fill_slip_values, 50),
        "slippage_bps_p95": _pctl(fill_slip_values, 95),
        "top_symbols_by_total_slippage_bps": symbol_slip_sorted[:10],
        "top_symbols_by_fill_count": fill_symbol_counts.most_common(10),
    },
    "tca": {
        "rows": tca_count,
        "filled_rows": tca_filled,
        "terminal_nonfills": tca_terminal_nonfill,
        "fill_rate": (tca_filled / tca_count) if tca_count else None,
        "is_bps_avg": (sum(tca_is_values) / len(tca_is_values)) if tca_is_values else None,
        "is_bps_p50": _pctl(tca_is_values, 50),
        "is_bps_p95": _pctl(tca_is_values, 95),
        "fill_latency_ms_avg": (
            (sum(tca_latency_values) / len(tca_latency_values))
            if tca_latency_values
            else None
        ),
        "fill_latency_ms_p50": _pctl(tca_latency_values, 50),
        "fill_latency_ms_p95": _pctl(tca_latency_values, 95),
        "worst_symbols_by_mean_is_bps": worst_is[:10],
    },
    "orders": {
        "status_transition_new_status": dict(order_status_counts),
        "final_state_status": dict(order_final_counts),
        "order_event_latency_ms_avg": (
            (sum(order_latency_values) / len(order_latency_values))
            if order_latency_values
            else None
        ),
        "order_event_latency_ms_p50": _pctl(order_latency_values, 50),
        "order_event_latency_ms_p95": _pctl(order_latency_values, 95),
        "order_event_latency_ms_max": max(order_latency_values) if order_latency_values else None,
    },
    "top_loss_drivers_from_trade_history": trade.get("top_loss_drivers", {}),
}

breakdown_path = runtime_root / "reports" / (
    f"today_negative_breakdown_{REPORT_DATE.replace('-', '')}.json"
)
breakdown_path.parent.mkdir(parents=True, exist_ok=True)
breakdown_path.write_text(
    json.dumps(breakdown, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print(
    json.dumps(
        {
            "report_date": REPORT_DATE,
            "report_targets": [str(target) for target in report_targets],
            "breakdown_path": str(breakdown_path),
        },
        sort_keys=True,
    )
)
PY

echo "runtime reports refreshed for ${REPORT_DATE}"
