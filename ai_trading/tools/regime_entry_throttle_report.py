"""Build a report-only regime entry throttle artifact."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from ai_trading.runtime.artifacts import resolve_runtime_artifact_path
from ai_trading.runtime.regime_entry_throttle import (
    build_regime_entry_throttle_report,
    evaluate_regime_entry_throttle,
)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _to_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int:
    parsed = _to_float(value)
    return max(0, int(parsed)) if parsed is not None else 0


def _default_path(relative: str) -> Path:
    return resolve_runtime_artifact_path(relative, default_relative=relative)


def _aggregate_live_cost_evidence(
    live_cost_model: Mapping[str, Any],
    health: Mapping[str, Any],
) -> dict[str, Any]:
    rows = live_cost_model.get("by_symbol_side_session")
    if not isinstance(rows, list):
        rows = live_cost_model.get("by_symbol_side_session_order_type_volatility")
    sample_count = 0
    spread_values: list[float] = []
    high_volatility = False
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            sample_count += _to_int(row.get("sample_count"))
            spread = _to_float(row.get("p90_spread_bps") or row.get("mean_spread_bps"))
            if spread is not None:
                spread_values.append(spread)
            vol_bucket = str(row.get("volatility_bucket") or "").strip().lower()
            if vol_bucket in {"high", "elevated", "wide"}:
                high_volatility = True
    provider = health.get("data_provider")
    if not isinstance(provider, Mapping):
        provider = {}
    provider_status = str(provider.get("status") or "").strip().lower()
    generated_at = live_cost_model.get("generated_at")
    status = live_cost_model.get("status")
    if isinstance(status, Mapping):
        generated_at = generated_at or status.get("generated_at")
    return {
        "observed_at": generated_at,
        "sample_count": sample_count,
        "spread_bps": max(spread_values) if spread_values else None,
        "volatility_bps": 100.0 if high_volatility else 25.0,
        "provider_healthy": provider_status in {"healthy", "ready", "warming_up"},
        "provider_state": provider_status or "unknown",
    }


def build_report(
    *,
    live_cost_model: Mapping[str, Any],
    health: Mapping[str, Any],
    report_date: str,
    enforce: bool,
    live_canary: bool,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(UTC)
    evidence = _aggregate_live_cost_evidence(live_cost_model, health)
    evaluation = evaluate_regime_entry_throttle(
        evidence,
        now=current,
        live_canary=live_canary,
        enforce=enforce,
    )
    report = build_regime_entry_throttle_report([evaluation], report_date=None)
    report["report_date"] = report_date
    report["mode"] = "enforced" if enforce else "report_only"
    report["evidence"] = evidence
    return report


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--live-cost-model-json", type=Path, default=None)
    parser.add_argument("--health-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--enforce", action="store_true")
    parser.add_argument("--live-canary", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(
        live_cost_model=_read_json(args.live_cost_model_json or _default_path("runtime/live_cost_model_latest.json")),
        health=_read_json(args.health_json or _default_path("runtime/health_latest.json")),
        report_date=str(args.report_date),
        enforce=bool(args.enforce),
        live_canary=bool(args.live_canary),
    )
    _write_json(args.output_json, report)
    sys.stdout.write(json.dumps({"path": str(args.output_json), "status": "ready"}, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
