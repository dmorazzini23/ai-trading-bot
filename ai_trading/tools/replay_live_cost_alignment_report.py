"""Build replay/live cost-alignment diagnostics from generated artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from ai_trading.config.management import get_env
from ai_trading.replay.live_cost_alignment import resolve_live_cost_alignments
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(parsed)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return int(default)
    return int(parsed)


def _default_path(relative: str) -> Path:
    return resolve_runtime_artifact_path(relative, default_relative=relative)


def _live_cost_rows_as_replay_rows(
    live_cost_model: Mapping[str, Any],
    *,
    fallback_cost_bps: float,
) -> list[dict[str, Any]]:
    rows = live_cost_model.get("by_symbol_side_session_order_type_volatility")
    if not isinstance(rows, list):
        rows = live_cost_model.get("by_symbol_side_session")
    out: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        out.append(
            {
                "symbol": symbol,
                "side": row.get("side") or "buy",
                "session_bucket": row.get("session_regime") or "unknown",
                "order_type": row.get("order_type") or "unknown",
                "volatility_bucket": row.get("volatility_bucket") or "unknown",
                "fallback_cost_bps": float(fallback_cost_bps),
            }
        )
    return out


def _replay_rows_from_report(
    replay_report: Mapping[str, Any],
    *,
    fallback_cost_bps: float,
) -> list[dict[str, Any]]:
    rows = replay_report.get("replay_cost_rows")
    if isinstance(rows, list):
        return [dict(row) for row in rows if isinstance(row, Mapping)]
    bucket_summary = replay_report.get("replay_bucket_summary")
    out: list[dict[str, Any]] = []
    if isinstance(bucket_summary, Mapping):
        for raw_key, row in bucket_summary.items():
            if not isinstance(row, Mapping):
                continue
            parts = str(raw_key).split("|")
            symbol = str(row.get("symbol") or (parts[0] if parts else "")).strip().upper()
            if not symbol:
                continue
            out.append(
                {
                    "symbol": symbol,
                    "side": row.get("side") or (parts[1] if len(parts) > 1 else "buy"),
                    "session_bucket": row.get("session_regime")
                    or row.get("session_bucket")
                    or (parts[2] if len(parts) > 2 else "unknown"),
                    "order_type": row.get("order_type") or "unknown",
                    "volatility_bucket": row.get("volatility_bucket") or "unknown",
                    "fallback_cost_bps": row.get("fallback_cost_bps", fallback_cost_bps),
                }
            )
    return out


def build_replay_live_cost_alignment_report(
    *,
    live_cost_model: Mapping[str, Any],
    replay_report: Mapping[str, Any],
    fallback_cost_bps: float,
    min_samples: int,
    max_age_seconds: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return artifact-only replay/live cost realism diagnostics."""

    replay_rows = _replay_rows_from_report(replay_report, fallback_cost_bps=fallback_cost_bps)
    source = "replay_report"
    if not replay_rows:
        replay_rows = _live_cost_rows_as_replay_rows(
            live_cost_model,
            fallback_cost_bps=fallback_cost_bps,
        )
        source = "live_cost_buckets"
    alignment = resolve_live_cost_alignments(
        live_cost_model,
        replay_rows,
        now=now,
        max_age_seconds=float(max_age_seconds),
        min_samples=max(1, int(min_samples)),
    )
    summary = dict(alignment.get("summary") or {})
    stale_count = _to_int(summary.get("stale_count"))
    optimism_count = _to_int(summary.get("optimism_count"))
    count = _to_int(summary.get("count"))
    model_status = str((live_cost_model.get("status") or {}).get("status") or live_cost_model.get("status") or "missing")
    acceptable = bool(count > 0 and model_status in {"ready", "ok"} and stale_count == 0)
    realism = "acceptable"
    if count <= 0:
        realism = "unavailable"
    elif stale_count > 0:
        realism = "stale"
    elif optimism_count > 0:
        realism = "conservative_fallback_clamped_optimism"
    return {
        "schema_version": "1.0.0",
        "artifact_type": "replay_live_cost_alignment_report",
        "generated_at": (now or datetime.now(UTC)).astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "status": "ready" if count > 0 else "missing_evidence",
        "source": source,
        "cost_realism": {
            "acceptable": acceptable,
            "status": realism,
            "model_status": model_status,
            "fallback_cost_bps": float(max(0.0, fallback_cost_bps)),
            "min_samples": max(1, int(min_samples)),
            "max_age_seconds": float(max_age_seconds),
        },
        "summary": summary,
        "items": alignment.get("items", []),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-cost-model-json", type=Path, default=None)
    parser.add_argument("--replay-report-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--fallback-cost-bps", type=float, default=None)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--max-age-seconds", type=float, default=86_400.0)
    args = parser.parse_args(argv)
    fallback = (
        float(args.fallback_cost_bps)
        if args.fallback_cost_bps is not None
        else _to_float(get_env("AI_TRADING_REPLAY_FALLBACK_COST_BPS", "3.0", cast=str), 3.0)
    )
    report = build_replay_live_cost_alignment_report(
        live_cost_model=_read_json(args.live_cost_model_json or _default_path("runtime/live_cost_model_latest.json")),
        replay_report=_read_json(args.replay_report_json or _default_path("runtime/replay_governance_refresh_latest.json")),
        fallback_cost_bps=fallback,
        min_samples=int(args.min_samples),
        max_age_seconds=float(args.max_age_seconds),
    )
    _write_json(args.output_json, report)
    sys.stdout.write(json.dumps({"path": str(args.output_json), "status": report["status"]}, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
