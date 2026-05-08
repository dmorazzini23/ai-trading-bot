"""Build a model-data drift monitor artifact from baseline and current evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


_DRIFT_CATEGORIES = (
    "feature",
    "label",
    "calibration",
    "live_cost",
    "symbol",
    "provider",
    "regime",
)


def _parse_ts(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _mapping(payload: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)
    return current if isinstance(current, Mapping) else {}


def _freshness(payload: Mapping[str, Any], *, max_age_hours: float, now: datetime) -> dict[str, Any]:
    generated = _parse_ts(payload.get("generated_at") or payload.get("as_of") or payload.get("timestamp"))
    age_hours = None
    if generated is not None:
        age_hours = max(0.0, (now - generated).total_seconds() / 3600.0)
    return {
        "fresh": bool(generated is not None and generated <= now and age_hours is not None and age_hours <= max_age_hours),
        "generated_at": generated.isoformat().replace("+00:00", "Z") if generated else None,
        "age_hours": age_hours,
        "max_age_hours": float(max_age_hours),
    }


def _relative_delta(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None
    denominator = max(abs(float(baseline)), 1e-9)
    return abs(float(current) - float(baseline)) / denominator


def _numeric_drift(
    *,
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    keys: tuple[str, ...],
    rel_threshold: float,
    abs_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in keys:
        base_value = _to_float(baseline.get(key))
        current_value = _to_float(current.get(key))
        if base_value is None or current_value is None:
            continue
        abs_delta = abs(current_value - base_value)
        rel_delta = _relative_delta(current_value, base_value)
        drifted = bool(abs_delta > abs_threshold and (rel_delta or 0.0) > rel_threshold)
        rows.append(
            {
                "metric": key,
                "baseline": base_value,
                "current": current_value,
                "abs_delta": abs_delta,
                "relative_delta": rel_delta,
                "drifted": drifted,
            }
        )
    return rows


def _distribution(payload: Mapping[str, Any]) -> dict[str, float]:
    raw = payload.get("counts") if isinstance(payload.get("counts"), Mapping) else payload
    values: dict[str, float] = {}
    for key, value in raw.items():
        parsed = _to_float(value)
        if parsed is not None and parsed >= 0.0:
            values[str(key)] = parsed
    total = sum(values.values())
    if total <= 0.0:
        return {}
    return {key: value / total for key, value in values.items()}


def _distribution_drift(
    *,
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    threshold: float,
) -> dict[str, Any]:
    base = _distribution(baseline)
    curr = _distribution(current)
    keys = sorted(set(base) | set(curr))
    total_variation = 0.5 * sum(abs(curr.get(key, 0.0) - base.get(key, 0.0)) for key in keys)
    largest = [
        {
            "bucket": key,
            "baseline_share": base.get(key, 0.0),
            "current_share": curr.get(key, 0.0),
            "abs_delta": abs(curr.get(key, 0.0) - base.get(key, 0.0)),
        }
        for key in keys
    ]
    def _abs_delta(item: Mapping[str, Any]) -> float:
        value = item.get("abs_delta")
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    largest.sort(key=_abs_delta, reverse=True)
    return {
        "total_variation": total_variation,
        "threshold": float(threshold),
        "drifted": bool(total_variation > threshold),
        "largest_bucket_moves": largest[:5],
    }


def _feature_drift(baseline: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    feature_rows: list[dict[str, Any]] = []
    for name, base_value in baseline.items():
        if not isinstance(base_value, Mapping):
            continue
        current_value = current.get(name)
        if not isinstance(current_value, Mapping):
            continue
        metrics = _numeric_drift(
            baseline=base_value,
            current=current_value,
            keys=("mean", "std", "missing_rate", "zero_rate"),
            rel_threshold=0.25,
            abs_threshold=0.02,
        )
        drifted = any(bool(row["drifted"]) for row in metrics)
        feature_rows.append({"feature": name, "drifted": drifted, "metrics": metrics})
    return {
        "status": "drift" if any(row["drifted"] for row in feature_rows) else "ok",
        "drifted": any(row["drifted"] for row in feature_rows),
        "features": feature_rows,
    }


def _status_from_rows(rows: list[dict[str, Any]]) -> tuple[str, bool]:
    drifted = any(bool(row.get("drifted")) for row in rows)
    return ("drift" if drifted else "ok", drifted)


def build_model_data_drift_monitor(
    *,
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    max_baseline_age_hours: float = 168.0,
    max_current_age_hours: float = 48.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Compare current model data evidence with a freshness-gated baseline."""

    generated = (now or datetime.now(UTC)).astimezone(UTC)
    baseline_freshness = _freshness(baseline, max_age_hours=max_baseline_age_hours, now=generated)
    current_freshness = _freshness(current, max_age_hours=max_current_age_hours, now=generated)
    reasons: list[str] = []
    if not baseline:
        reasons.append("baseline_missing")
    elif not baseline_freshness["fresh"]:
        reasons.append("baseline_stale")
    if not current:
        reasons.append("current_missing")
    elif not current_freshness["fresh"]:
        reasons.append("current_stale")

    feature = _feature_drift(_mapping(baseline, "features"), _mapping(current, "features"))
    label_rows = _numeric_drift(
        baseline=_mapping(baseline, "labels"),
        current=_mapping(current, "labels"),
        keys=("positive_rate", "mean_return_bps", "event_rate", "class_balance"),
        rel_threshold=0.20,
        abs_threshold=0.02,
    )
    label_status, label_drifted = _status_from_rows(label_rows)
    calibration_rows = _numeric_drift(
        baseline=_mapping(baseline, "calibration"),
        current=_mapping(current, "calibration"),
        keys=("capture_ratio", "brier_score", "ece", "win_rate"),
        rel_threshold=0.20,
        abs_threshold=0.02,
    )
    calibration_status, calibration_drifted = _status_from_rows(calibration_rows)
    live_cost_rows = _numeric_drift(
        baseline=_mapping(baseline, "live_cost"),
        current=_mapping(current, "live_cost"),
        keys=("mean_total_cost_bps", "p90_total_cost_bps", "mean_quote_age_ms", "p90_spread_bps"),
        rel_threshold=0.25,
        abs_threshold=1.0,
    )
    live_cost_status, live_cost_drifted = _status_from_rows(live_cost_rows)
    symbol = _distribution_drift(
        baseline=_mapping(baseline, "symbols"),
        current=_mapping(current, "symbols"),
        threshold=0.20,
    )
    provider = _distribution_drift(
        baseline=_mapping(baseline, "providers"),
        current=_mapping(current, "providers"),
        threshold=0.15,
    )
    regime = _distribution_drift(
        baseline=_mapping(baseline, "regimes"),
        current=_mapping(current, "regimes"),
        threshold=0.20,
    )
    checks = {
        "feature": feature,
        "label": {"status": label_status, "drifted": label_drifted, "metrics": label_rows},
        "calibration": {
            "status": calibration_status,
            "drifted": calibration_drifted,
            "metrics": calibration_rows,
        },
        "live_cost": {
            "status": live_cost_status,
            "drifted": live_cost_drifted,
            "metrics": live_cost_rows,
        },
        "symbol": {"status": "drift" if symbol["drifted"] else "ok", **symbol},
        "provider": {"status": "drift" if provider["drifted"] else "ok", **provider},
        "regime": {"status": "drift" if regime["drifted"] else "ok", **regime},
    }
    drift_categories = [
        category
        for category in _DRIFT_CATEGORIES
        if bool(checks[category].get("drifted"))
    ]
    status = "blocked" if reasons else ("drift_detected" if drift_categories else "ok")
    return {
        "schema_version": "1.0.0",
        "artifact_type": "model_data_drift_monitor",
        "generated_at": generated.isoformat().replace("+00:00", "Z"),
        "status": status,
        "reasons": reasons,
        "freshness": {
            "baseline": baseline_freshness,
            "current": current_freshness,
        },
        "summary": {
            "covered_categories": list(_DRIFT_CATEGORIES),
            "drift_categories": drift_categories,
            "drift_count": len(drift_categories),
            "baseline_fresh": baseline_freshness["fresh"],
            "current_fresh": current_freshness["fresh"],
        },
        "checks": checks,
        "operator_note": "Baseline freshness is fail-closed; stale baselines block drift recommendations.",
    }


def _default_path(path_value: str, *, for_write: bool = False) -> Path:
    return resolve_runtime_artifact_path(path_value, default_relative=path_value, for_write=for_write)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-json", type=Path, default=None)
    parser.add_argument("--current-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--max-baseline-age-hours", type=float, default=168.0)
    parser.add_argument("--max-current-age-hours", type=float, default=48.0)
    args = parser.parse_args(argv)

    report = build_model_data_drift_monitor(
        baseline=_read_json(args.baseline_json),
        current=_read_json(args.current_json),
        max_baseline_age_hours=max(0.0, float(args.max_baseline_age_hours)),
        max_current_age_hours=max(0.0, float(args.max_current_age_hours)),
    )
    output = args.output_json or _default_path("runtime/model_data_drift_monitor_latest.json", for_write=True)
    report["paths"] = {
        "baseline": str(args.baseline_json) if args.baseline_json is not None else None,
        "current": str(args.current_json) if args.current_json is not None else None,
        "report": str(output),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps({"path": str(output), "status": report["status"]}, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
