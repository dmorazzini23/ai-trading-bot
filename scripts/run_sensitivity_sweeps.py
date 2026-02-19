#!/usr/bin/env python3
"""Run after-hours threshold/min-support sweeps and persist ranked outputs."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_trading.config.management import reload_env
from ai_trading.training.after_hours import run_after_hours_training


def _parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def _parse_now(raw: str | None) -> datetime:
    if not raw:
        return datetime.now(UTC)
    text = raw.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _extract_selected_candidate(output: dict[str, Any]) -> dict[str, Any]:
    selected = {}
    for candidate in output.get("candidate_metrics", []):
        if bool(candidate.get("selected")):
            selected = dict(candidate)
            break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--thresholds",
        default="0.48,0.50,0.52,0.54,0.56",
        help="Comma-separated default threshold candidates",
    )
    parser.add_argument(
        "--min-support-values",
        default="20,25,30,40",
        help="Comma-separated min threshold support values",
    )
    parser.add_argument(
        "--now",
        default=None,
        help="Override now in ISO8601 (UTC recommended), e.g. 2026-02-18T22:10:00Z",
    )
    parser.add_argument(
        "--output",
        default="runtime/research_reports/after_hours_sensitivity_sweep.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    now_utc = _parse_now(args.now)
    thresholds = _parse_float_list(args.thresholds)
    min_support_values = _parse_int_list(args.min_support_values)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        for min_support in min_support_values:
            os.environ["AI_TRADING_AFTER_HOURS_DEFAULT_THRESHOLD"] = str(threshold)
            os.environ["AI_TRADING_AFTER_HOURS_MIN_THRESHOLD_SUPPORT"] = str(min_support)
            reload_env()
            output = run_after_hours_training(now=now_utc)
            selected = _extract_selected_candidate(output)
            rows.append(
                {
                    "threshold": threshold,
                    "min_support": min_support,
                    "status": output.get("status"),
                    "governance_status": output.get("governance_status"),
                    "model": selected.get("name"),
                    "support": selected.get("support"),
                    "expectancy_bps": selected.get("mean_expectancy_bps"),
                    "max_drawdown_bps": selected.get("max_drawdown_bps"),
                    "turnover_ratio": selected.get("turnover_ratio"),
                    "hit_rate_stability": selected.get("hit_rate_stability"),
                    "edge_gates": output.get("edge_gates"),
                    "sensitivity_sweep": output.get("sensitivity_sweep"),
                    "report_path": output.get("report_path"),
                }
            )

    ranked = sorted(
        rows,
        key=lambda row: (
            float(row.get("expectancy_bps") or -1e9),
            -float(row.get("max_drawdown_bps") or 1e9),
            -float(row.get("turnover_ratio") or 1e9),
        ),
        reverse=True,
    )
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "forced_now_utc": now_utc.isoformat(),
        "thresholds": thresholds,
        "min_support_values": min_support_values,
        "ranked_results": ranked,
        "top_10": ranked[:10],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "rows": len(ranked)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
