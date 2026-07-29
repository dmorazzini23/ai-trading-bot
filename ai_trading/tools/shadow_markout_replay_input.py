"""Export resolved shadow markouts as a bounded research-only JSONL dataset."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from ai_trading.runtime.atomic_io import atomic_write_text


DEFAULT_MAX_ROWS = 50_000
_FORBIDDEN_OUTPUT_NAMES = frozenset({"governance_tca_recent.jsonl"})
_FALSE_AUTHORITY_FIELDS = (
    "fill_based_evidence",
    "promotion_eligible",
    "runtime_authority",
    "promotion_authority",
    "live_money_authority",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _validate_shadow_authority(
    payload: Mapping[str, Any],
    *,
    context: str,
) -> None:
    if payload.get("evidence_type") != "shadow_counterfactual":
        raise ValueError(f"{context} evidence type is not shadow counterfactual")
    if payload.get("evidence_partition") != "shadow":
        raise ValueError(f"{context} evidence partition is not shadow")
    if payload.get("research_only") is not True:
        raise ValueError(f"{context} is not explicitly research-only")
    for field in _FALSE_AUTHORITY_FIELDS:
        if payload.get(field) is not False:
            raise ValueError(f"{context} has invalid authority field {field}")


def _validated_outcomes(
    report: Mapping[str, Any],
) -> Sequence[Mapping[str, Any]]:
    _validate_shadow_authority(report, context="markout report")
    if list(report.get("horizons_bars") or []) != [1, 3, 5]:
        raise ValueError("markout report horizons must be exactly 1,3,5")
    invariant = report.get("coverage_invariant")
    if not isinstance(invariant, Mapping) or invariant.get("passed") is not True:
        raise ValueError("markout report coverage invariant did not pass")
    outcomes = report.get("outcomes")
    if not isinstance(outcomes, list):
        raise ValueError("markout report outcomes are missing")
    validated: list[Mapping[str, Any]] = []
    for index, raw_outcome in enumerate(outcomes):
        if not isinstance(raw_outcome, Mapping):
            raise ValueError(f"markout outcome {index} is invalid")
        _validate_shadow_authority(
            raw_outcome,
            context=f"markout outcome {index}",
        )
        if raw_outcome.get("executed") is not False:
            raise ValueError(f"markout outcome {index} is mixed with execution evidence")
        validated.append(raw_outcome)
    return validated


def build_shadow_markout_replay_rows(
    report: Mapping[str, Any],
    *,
    source_report_sha256: str,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> list[dict[str, Any]]:
    """Return deterministic resolved-only rows from one governed markout report."""

    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    outcomes = _validated_outcomes(report)
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, outcome in enumerate(outcomes):
        if outcome.get("label_status") != "resolved":
            continue
        outcome_id = str(outcome.get("outcome_id") or "").strip()
        correlation_id = str(outcome.get("correlation_id") or "").strip()
        symbol = str(outcome.get("symbol") or "").strip().upper()
        side = str(outcome.get("side") or "").strip().lower()
        horizon = int(outcome.get("horizon_bars") or 0)
        entry_price = _safe_float(outcome.get("entry_price"))
        exit_price = _safe_float(outcome.get("exit_price"))
        net_edge = _safe_float(outcome.get("net_markout_bps"))
        gross_edge = _safe_float(outcome.get("gross_markout_bps"))
        cost = _safe_float(outcome.get("round_trip_cost_bps"))
        if (
            not outcome_id
            or not correlation_id
            or not symbol
            or side not in {"buy", "sell"}
            or horizon not in {1, 3, 5}
            or entry_price is None
            or entry_price <= 0.0
            or exit_price is None
            or exit_price <= 0.0
            or net_edge is None
            or gross_edge is None
            or cost is None
        ):
            raise ValueError(f"resolved markout outcome {index} is incomplete")
        if outcome_id in seen_ids:
            raise ValueError(f"duplicate resolved markout outcome id {outcome_id}")
        seen_ids.add(outcome_id)
        rows.append(
            {
                "schema_version": "1.0.0",
                "replay_row_id": outcome_id,
                "outcome_id": outcome_id,
                "correlation_id": correlation_id,
                "symbol": symbol,
                "side": side,
                "source_timestamp": outcome.get("source_timestamp"),
                "decision_timestamp": outcome.get("decision_timestamp"),
                "label_end_timestamp": outcome.get("label_end_timestamp"),
                "horizon_bars": horizon,
                "bar_timeframe": "1Min",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_markout_bps": gross_edge,
                "round_trip_cost_bps": cost,
                "net_markout_bps": net_edge,
                "quote_age_ms": outcome.get("quote_age_ms"),
                "spread_bps": outcome.get("spread_bps"),
                "order_type": outcome.get("order_type"),
                "session": outcome.get("session"),
                "market_regime": outcome.get("market_regime"),
                "volatility_regime": outcome.get("volatility_regime"),
                "trend_regime": outcome.get("trend_regime"),
                "execution_profile": outcome.get("execution_profile"),
                "submitted": bool(outcome.get("submitted")),
                "controlled_skip": bool(outcome.get("controlled_skip")),
                "source_report_sha256": source_report_sha256,
                "evidence_type": "shadow_counterfactual",
                "evidence_partition": "shadow",
                "research_only": True,
                "fill_based_evidence": False,
                "promotion_eligible": False,
                "runtime_authority": False,
                "promotion_authority": False,
                "live_money_authority": False,
            }
        )
    rows.sort(
        key=lambda row: (
            str(row.get("source_timestamp") or ""),
            str(row["symbol"]),
            int(row["horizon_bars"]),
            str(row["outcome_id"]),
        )
    )
    if len(rows) > max_rows:
        raise ValueError(
            f"resolved markout rows exceed bounded maximum: {len(rows)} > {max_rows}"
        )
    return rows


def _validate_output_path(path: Path) -> None:
    if path.name.lower() in _FORBIDDEN_OUTPUT_NAMES:
        raise ValueError(
            "shadow markout research output cannot target replay governance input"
        )


def write_shadow_markout_replay_input(
    *,
    markout_report_path: Path,
    output_jsonl: Path,
    latest_jsonl: Path | None = None,
    manifest_json: Path | None = None,
    latest_manifest_json: Path | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> dict[str, Any]:
    """Validate one markout report and atomically write its research dataset."""

    for path in (
        output_jsonl,
        latest_jsonl,
        manifest_json,
        latest_manifest_json,
    ):
        if path is not None:
            _validate_output_path(path)
    try:
        source_bytes = markout_report_path.read_bytes()
        parsed = json.loads(source_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("markout report is unreadable") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError("markout report must be a JSON object")
    source_sha256 = _sha256_bytes(source_bytes)
    rows = build_shadow_markout_replay_rows(
        parsed,
        source_report_sha256=source_sha256,
        max_rows=max_rows,
    )
    serialized = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    output_targets = [output_jsonl]
    if latest_jsonl is not None and latest_jsonl != output_jsonl:
        output_targets.append(latest_jsonl)
    for target in output_targets:
        atomic_write_text(target, serialized)
    content_sha256 = _sha256_bytes(serialized.encode("utf-8"))
    manifest = {
        "schema_version": "1.0.0",
        "artifact_type": "shadow_markout_replay_input_manifest",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "report_date": parsed.get("report_date"),
        "source_report_path": str(markout_report_path.resolve()),
        "source_report_sha256": source_sha256,
        "output_jsonl": str(output_jsonl.resolve()),
        "content_sha256": content_sha256,
        "row_count": len(rows),
        "max_rows": int(max_rows),
        "resolved_only": True,
        "horizons_bars": [1, 3, 5],
        "bar_timeframe": "1Min",
        "evidence_type": "shadow_counterfactual",
        "evidence_partition": "shadow",
        "research_only": True,
        "fill_based_evidence": False,
        "promotion_eligible": False,
        "runtime_authority": False,
        "promotion_authority": False,
        "live_money_authority": False,
    }
    manifest_targets = [
        target
        for target in (manifest_json, latest_manifest_json)
        if target is not None
    ]
    for target in dict.fromkeys(manifest_targets):
        atomic_write_text(
            target,
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markout-report-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--latest-jsonl", type=Path, default=None)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--latest-manifest-json", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    args = parser.parse_args(argv)
    try:
        manifest = write_shadow_markout_replay_input(
            markout_report_path=args.markout_report_json,
            output_jsonl=args.output_jsonl,
            latest_jsonl=args.latest_jsonl,
            manifest_json=args.manifest_json,
            latest_manifest_json=args.latest_manifest_json,
            max_rows=int(args.max_rows),
        )
    except ValueError as exc:
        parser.error(str(exc))
    sys.stdout.write(json.dumps(manifest, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MAX_ROWS",
    "build_shadow_markout_replay_rows",
    "main",
    "write_shadow_markout_replay_input",
]
