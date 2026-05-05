"""Join shadow quote telemetry to replay candidates and estimate veto impact."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ai_trading.logging import get_logger
from ai_trading.tools.ml_shadow_report import _load_shadow_rows

logger = get_logger(__name__)


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _parse_ts(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed


def _bucket_ts(value: Any, bucket: str, *, time_of_day: bool = False) -> str | None:
    ts = _parse_ts(value)
    if ts is None:
        return None
    floored = ts.floor(bucket)
    if time_of_day:
        return str(floored.strftime("%H:%M"))
    return str(floored.isoformat())


def _shadow_ts(row: Mapping[str, Any]) -> Any:
    market = row.get("market")
    if isinstance(market, Mapping):
        return market.get("bar_timestamp") or market.get("quote_timestamp") or row.get("ts")
    return row.get("ts")


def _cost_fields(row: Mapping[str, Any]) -> tuple[float | None, float | None]:
    spread: float | None = None
    quote_age: float | None = None
    for source in (row.get("cost"), row.get("market"), row):
        if not isinstance(source, Mapping):
            continue
        spread = spread if spread is not None else _finite_float(source.get("spread_bps"))
        quote_age = quote_age if quote_age is not None else _finite_float(source.get("quote_age_ms"))
    return spread, quote_age


def _gate_reasons(
    *,
    spread_bps: float | None,
    quote_age_ms: float | None,
    max_spread_bps: float,
    max_quote_age_ms: float,
    reject_missing: bool,
) -> list[str]:
    reasons: list[str] = []
    if spread_bps is None:
        if reject_missing:
            reasons.append("missing_spread")
    elif spread_bps > max_spread_bps:
        reasons.append("wide_spread")
    if quote_age_ms is None:
        if reject_missing:
            reasons.append("missing_quote_age")
    elif quote_age_ms > max_quote_age_ms:
        reasons.append("stale_quote")
    return reasons


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "MICROSTRUCTURE_BRIDGE_INVALID_JSONL_ROW",
                    extra={"path": str(path), "line_number": line_number},
                )
                continue
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    return rows


def _shadow_quote_index(
    rows: list[dict[str, Any]],
    *,
    bucket: str,
    time_of_day: bool,
) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[tuple[float | None, float | None]]] = {}
    for row in rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        bucket_value = _bucket_ts(_shadow_ts(row), bucket, time_of_day=time_of_day)
        if not symbol or bucket_value is None:
            continue
        spread, quote_age = _cost_fields(row)
        grouped.setdefault((symbol, bucket_value), []).append((spread, quote_age))
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for key, values in grouped.items():
        spreads = [value[0] for value in values if value[0] is not None]
        quote_ages = [value[1] for value in values if value[1] is not None]
        index[key] = {
            "shadow_rows": len(values),
            "spread_bps": float(np.mean(spreads)) if spreads else None,
            "quote_age_ms": float(np.mean(quote_ages)) if quote_ages else None,
        }
    return index


def _candidate_ts(row: Mapping[str, Any]) -> Any:
    return row.get("ts") or row.get("submit_ts") or row.get("entry_ts")


def _candidate_markout(row: Mapping[str, Any]) -> float | None:
    for key in ("net_markout_bps", "markout_bps", "pnl_bps"):
        value = _finite_float(row.get(key))
        if value is not None:
            return value
    return None


def _summary(values: list[float]) -> dict[str, Any]:
    positive = sum(1 for value in values if value > 0.0)
    return {
        "samples": len(values),
        "mean_bps": float(np.mean(values)) if values else None,
        "total_bps": float(np.sum(values)) if values else 0.0,
        "positive_rate": float(positive / len(values)) if values else None,
    }


def build_microstructure_bridge_report(args: argparse.Namespace) -> dict[str, Any]:
    shadow_rows = _load_shadow_rows(Path(args.shadow_jsonl))
    candidates = _load_jsonl(Path(args.accepted_candidates_jsonl))
    match_time_of_day = bool(getattr(args, "match_time_of_day", False))
    quote_index = _shadow_quote_index(
        shadow_rows,
        bucket=str(args.bucket),
        time_of_day=match_time_of_day,
    )
    joined_rows: list[dict[str, Any]] = []
    retained_markouts: list[float] = []
    rejected_markouts: list[float] = []
    missing_join = 0
    reason_counts: Counter[str] = Counter()
    for candidate in candidates:
        symbol = str(candidate.get("symbol") or "").strip().upper()
        bucket_value = _bucket_ts(
            _candidate_ts(candidate),
            str(args.bucket),
            time_of_day=match_time_of_day,
        )
        markout = _candidate_markout(candidate)
        quote = quote_index.get((symbol, bucket_value)) if bucket_value is not None else None
        if quote is None:
            missing_join += 1
            spread = None
            quote_age = None
        else:
            spread = _finite_float(quote.get("spread_bps"))
            quote_age = _finite_float(quote.get("quote_age_ms"))
        reasons = _gate_reasons(
            spread_bps=spread,
            quote_age_ms=quote_age,
            max_spread_bps=float(args.max_spread_bps),
            max_quote_age_ms=float(args.max_quote_age_ms),
            reject_missing=bool(args.reject_missing),
        )
        reason_counts.update(reasons)
        if markout is not None:
            if reasons:
                rejected_markouts.append(markout)
            else:
                retained_markouts.append(markout)
        joined_rows.append(
            {
                "symbol": symbol,
                "bucket": bucket_value,
                "markout_bps": markout,
                "spread_bps": spread,
                "quote_age_ms": quote_age,
                "would_reject": bool(reasons),
                "reasons": reasons,
                "shadow_rows": int(quote.get("shadow_rows", 0)) if quote else 0,
            }
        )
    all_markouts = retained_markouts + rejected_markouts
    report = {
        "schema_version": "1.0.0",
        "artifact_type": "microstructure_replay_bridge_report",
        "generated_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "shadow_jsonl": str(args.shadow_jsonl),
            "accepted_candidates_jsonl": str(args.accepted_candidates_jsonl),
            "bucket": str(args.bucket),
            "match_time_of_day": match_time_of_day,
        },
        "thresholds": {
            "max_spread_bps": float(args.max_spread_bps),
            "max_quote_age_ms": float(args.max_quote_age_ms),
            "reject_missing": bool(args.reject_missing),
        },
        "join": {
            "shadow_rows": int(len(shadow_rows)),
            "candidate_rows": int(len(candidates)),
            "joined_rows": int(len(candidates) - missing_join),
            "missing_join_rows": int(missing_join),
            "missing_join_rate": (
                float(missing_join / len(candidates)) if candidates else None
            ),
        },
        "gate": {
            "would_reject_count": int(sum(1 for row in joined_rows if row["would_reject"])),
            "would_reject_rate": (
                float(sum(1 for row in joined_rows if row["would_reject"]) / len(joined_rows))
                if joined_rows
                else None
            ),
            "reason_counts": dict(sorted(reason_counts.items())),
        },
        "markout": {
            "all": _summary(all_markouts),
            "retained": _summary(retained_markouts),
            "rejected": _summary(rejected_markouts),
            "retained_minus_all_mean_bps": (
                (_summary(retained_markouts)["mean_bps"] or 0.0)
                - (_summary(all_markouts)["mean_bps"] or 0.0)
                if all_markouts and retained_markouts
                else None
            ),
        },
        "joined_examples": joined_rows[:50],
        "recommendation": (
            "eligible_for_enforcement_review"
            if retained_markouts
            and rejected_markouts
            and (_summary(retained_markouts)["mean_bps"] or -1e9)
            > (_summary(all_markouts)["mean_bps"] or 1e9)
            and (_summary(rejected_markouts)["mean_bps"] or 1e9) < 0.0
            else "keep_shadow_only"
        ),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    logger.info(
        "MICROSTRUCTURE_REPLAY_BRIDGE_WRITTEN",
        extra={"path": str(output_path), "candidate_rows": len(candidates)},
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-jsonl", type=Path, required=True)
    parser.add_argument("--accepted-candidates-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bucket", type=str, default="min")
    parser.add_argument(
        "--match-time-of-day",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Join by symbol and HH:MM bucket, ignoring calendar date.",
    )
    parser.add_argument("--max-spread-bps", type=float, default=25.0)
    parser.add_argument("--max-quote-age-ms", type=float, default=1500.0)
    parser.add_argument("--reject-missing", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    build_microstructure_bridge_report(_build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
