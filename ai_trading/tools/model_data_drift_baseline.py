"""Build governed model-data drift evidence and immutable baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EVIDENCE_CONTRACT_VERSION = "1.0.0"
_REQUIRED_CATEGORIES = (
    "features",
    "labels",
    "calibration",
    "live_cost",
    "symbols",
    "providers",
    "regimes",
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


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _first_float(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if (value := _finite(row.get(key))) is not None:
            return value
    return None


def _row_ts(row: Mapping[str, Any]) -> datetime | None:
    for key in ("ts", "entry_time", "bar_ts", "timestamp", "pending_resolved_ts"):
        if (parsed := _parse_ts(row.get(key))) is not None:
            return parsed
    return None


def _windowed(
    rows: Iterable[Mapping[str, Any]],
    *,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        timestamp = _row_ts(row)
        if timestamp is not None and start <= timestamp <= end:
            selected.append(dict(row))
    return selected


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(percentile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _feature_stats(rows: Sequence[Mapping[str, Any]], *keys: str) -> dict[str, float] | None:
    values = [value for row in rows if (value := _first_float(row, *keys)) is not None]
    if not values:
        return None
    row_count = max(1, len(rows))
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "missing_rate": float(1.0 - (len(values) / row_count)),
        "zero_rate": float(sum(1 for value in values if value == 0.0) / len(values)),
        "observations": float(len(values)),
    }


def _distribution(rows: Sequence[Mapping[str, Any]], *keys: str) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for row in rows:
        for key in keys:
            value = str(row.get(key) or "").strip().lower()
            if value and value not in {"none", "null", "unknown"}:
                counts[value] += 1
                break
    return {"counts": dict(sorted(counts.items())), "observations": int(sum(counts.values()))}


def _coverage(payload: Mapping[str, Any], *, min_samples: int) -> dict[str, Any]:
    categories: dict[str, dict[str, Any]] = {}
    for category in _REQUIRED_CATEGORIES:
        value = payload.get(category)
        mapping = value if isinstance(value, Mapping) else {}
        if category == "features":
            observations = sum(
                int(_finite(item.get("observations")) or 0)
                for item in mapping.values()
                if isinstance(item, Mapping)
            )
            available = bool(mapping) and observations >= min_samples
        elif category in {"symbols", "providers", "regimes"}:
            observations = int(_finite(mapping.get("observations")) or 0)
            available = bool(mapping.get("counts")) and observations >= min_samples
        else:
            observations = int(_finite(mapping.get("observations")) or 0)
            available = observations >= min_samples
        categories[category] = {
            "available": bool(available),
            "observations": observations,
            "min_samples": int(min_samples),
        }
    missing = [name for name, detail in categories.items() if not detail["available"]]
    return {
        "complete": not missing,
        "missing_categories": missing,
        "categories": categories,
        "min_samples": int(min_samples),
    }


def build_model_data_drift_evidence(
    *,
    fills: Sequence[Mapping[str, Any]],
    tca_rows: Sequence[Mapping[str, Any]],
    generated_at: datetime | None = None,
    lookback_days: int = 30,
    min_samples: int = 25,
    model_id: str = "",
    model_hash: str = "",
    dataset_hash: str = "",
    model_role: str = "",
    sources: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Normalize execution evidence into the drift-monitor contract."""

    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)
    start = generated - timedelta(days=max(1, int(lookback_days)))
    fill_window = _windowed(fills, start=start, end=generated)
    tca_window = _windowed(tca_rows, start=start, end=generated)

    realized_rows = [
        row
        for row in fill_window
        if _first_float(row, "realized_net_edge_bps", "net_edge_bps", "markout_bps")
        is not None
    ]
    realized = [
        float(_first_float(row, "realized_net_edge_bps", "net_edge_bps", "markout_bps") or 0.0)
        for row in realized_rows
    ]
    expected_pairs = [
        (expected, realized_value)
        for row in realized_rows
        if (expected := _first_float(row, "expected_net_edge_bps", "expected_edge_bps"))
        is not None
        and (
            realized_value := _first_float(
                row,
                "realized_net_edge_bps",
                "net_edge_bps",
                "markout_bps",
            )
        )
        is not None
    ]
    total_costs = [
        abs(slippage) + max(0.0, fee)
        for row in fill_window
        if (slippage := _first_float(row, "slippage_bps", "slippage_drag_bps")) is not None
        for fee in [_first_float(row, "fee_bps") or 0.0]
    ]
    quote_ages = [
        value
        for row in tca_window
        if (value := _first_float(row, "decision_quote_age_ms", "quote_age_ms")) is not None
    ]
    spreads = [
        value
        for row in tca_window
        if (
            value := _first_float(
                row,
                "decision_spread_bps",
                "spread_bps",
                "spread_paid_bps",
            )
        )
        is not None
    ]

    features: dict[str, dict[str, float]] = {}
    for name, rows, keys in (
        ("serving_confidence", fill_window, ("confidence",)),
        ("expected_net_edge_bps", fill_window, ("expected_net_edge_bps", "expected_edge_bps")),
        ("slippage_bps", fill_window, ("slippage_bps", "slippage_drag_bps")),
        ("fill_latency_ms", tca_window, ("fill_latency_ms",)),
        ("spread_paid_bps", tca_window, ("spread_paid_bps", "decision_spread_bps")),
    ):
        if (stats := _feature_stats(rows, *keys)) is not None:
            features[name] = stats

    positive_rate = (
        float(sum(1 for value in realized if value > 0.0) / len(realized)) if realized else None
    )
    expected_sum = sum(pair[0] for pair in expected_pairs)
    capture_ratio = (
        float(sum(pair[1] for pair in expected_pairs) / expected_sum)
        if expected_pairs and abs(expected_sum) > 1e-9
        else None
    )
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "artifact_type": "model_data_drift_evidence",
        "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
        "generated_at": generated.isoformat().replace("+00:00", "Z"),
        "observation_window": {
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": generated.isoformat().replace("+00:00", "Z"),
            "lookback_days": max(1, int(lookback_days)),
        },
        "model": {
            "model_id": str(model_id or "") or None,
            "model_hash": str(model_hash or "") or None,
            "dataset_hash": str(dataset_hash or "") or None,
            "registry_role": str(model_role or "") or None,
        },
        "features": features,
        "labels": {
            "positive_rate": positive_rate,
            "class_balance": positive_rate,
            "mean_return_bps": float(statistics.fmean(realized)) if realized else None,
            "event_rate": float(len(realized_rows) / len(fill_window)) if fill_window else None,
            "observations": len(realized),
        },
        "calibration": {
            "capture_ratio": capture_ratio,
            "win_rate": positive_rate,
            "observations": len(expected_pairs),
        },
        "live_cost": {
            "mean_total_cost_bps": float(statistics.fmean(total_costs)) if total_costs else None,
            "p90_total_cost_bps": _percentile(total_costs, 0.90),
            "mean_quote_age_ms": float(statistics.fmean(quote_ages)) if quote_ages else None,
            "p90_spread_bps": _percentile(spreads, 0.90),
            "observations": len(total_costs),
        },
        "symbols": _distribution(realized_rows, "symbol"),
        "providers": _distribution(tca_window, "provider", "data_provider"),
        "regimes": _distribution(tca_window, "market_regime"),
        "sources": [dict(source) for source in sources],
        "sample_counts": {
            "fills": len(fill_window),
            "realized_labels": len(realized),
            "calibration_pairs": len(expected_pairs),
            "tca_rows": len(tca_window),
        },
        "promotion_authority": False,
        "live_money_authority": False,
    }
    payload["coverage"] = _coverage(payload, min_samples=max(1, int(min_samples)))
    payload["status"] = "ready" if payload["coverage"]["complete"] else "insufficient_evidence"
    return payload


def build_governed_drift_baseline(
    evidence: Mapping[str, Any],
    *,
    baseline_id: str,
    approved_by: str,
    approved_at: datetime | None = None,
) -> dict[str, Any]:
    """Promote complete normalized evidence to a non-authoritative baseline."""

    coverage = evidence.get("coverage")
    if not isinstance(coverage, Mapping) or not bool(coverage.get("complete")):
        raise ValueError("baseline_evidence_incomplete")
    if evidence.get("artifact_type") != "model_data_drift_evidence":
        raise ValueError("baseline_evidence_contract_invalid")
    if evidence.get("evidence_contract_version") != EVIDENCE_CONTRACT_VERSION:
        raise ValueError("baseline_evidence_contract_incompatible")
    identifier = str(baseline_id or "").strip()
    approver = str(approved_by or "").strip()
    if not identifier:
        raise ValueError("baseline_id_required")
    if not approver:
        raise ValueError("baseline_approver_required")
    approved = (approved_at or datetime.now(UTC)).astimezone(UTC)
    baseline = dict(evidence)
    baseline.update(
        {
            "schema_version": "1.0.0",
            "artifact_type": "model_data_drift_baseline",
            "baseline_id": identifier,
            "generated_at": approved.isoformat().replace("+00:00", "Z"),
            "status": "approved",
            "approval": {
                "approved": True,
                "approved_by": approver,
                "approved_at": approved.isoformat().replace("+00:00", "Z"),
                "automatic_roll_forward": False,
            },
            "promotion_authority": False,
            "live_money_authority": False,
        }
    )
    return baseline


def _source_descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def model_identity_from_registry(registry: Mapping[str, Any]) -> dict[str, str | None]:
    """Read the governed active identity without granting promotion authority."""

    for role in ("challenger", "champion"):
        raw_model = registry.get(f"active_{role}")
        if not isinstance(raw_model, Mapping):
            raw_model = registry.get(role)
        if not isinstance(raw_model, Mapping):
            continue
        model_id = str(raw_model.get("model_id") or "").strip()
        if not model_id:
            continue
        return {
            "model_id": model_id,
            "model_hash": str(raw_model.get("model_hash") or "").strip() or None,
            "dataset_hash": str(
                raw_model.get("dataset_fingerprint")
                or raw_model.get("dataset_hash")
                or ""
            ).strip()
            or None,
            "registry_role": role,
        }
    return {
        "model_id": None,
        "model_hash": None,
        "dataset_hash": None,
        "registry_role": None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fills-jsonl", type=Path, required=True)
    parser.add_argument("--tca-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--min-samples", type=int, default=25)
    parser.add_argument("--model-id", default="")
    parser.add_argument("--model-hash", default="")
    parser.add_argument("--dataset-hash", default="")
    parser.add_argument("--model-registry-json", type=Path, default=None)
    parser.add_argument("--baseline-id", default="")
    parser.add_argument("--approved-by", default="")
    args = parser.parse_args(argv)

    for path in (args.fills_jsonl, args.tca_jsonl):
        if not path.is_file():
            parser.error(f"evidence source does not exist: {path}")
    model_identity = {
        "model_id": str(args.model_id or "").strip() or None,
        "model_hash": str(args.model_hash or "").strip() or None,
        "dataset_hash": str(args.dataset_hash or "").strip() or None,
        "registry_role": None,
    }
    source_paths = [args.fills_jsonl, args.tca_jsonl]
    if args.model_registry_json is not None:
        if not args.model_registry_json.is_file():
            parser.error(f"model registry does not exist: {args.model_registry_json}")
        try:
            registry = json.loads(args.model_registry_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"model registry is unreadable: {exc}")
        if not isinstance(registry, Mapping):
            parser.error("model registry must contain a JSON object")
        governed_identity = model_identity_from_registry(registry)
        if not governed_identity["model_id"]:
            parser.error("model registry has no active governed champion or challenger identity")
        for key in ("model_id", "model_hash", "dataset_hash", "registry_role"):
            if not model_identity.get(key):
                model_identity[key] = governed_identity.get(key)
        source_paths.append(args.model_registry_json)

    evidence = build_model_data_drift_evidence(
        fills=_read_jsonl(args.fills_jsonl),
        tca_rows=_read_jsonl(args.tca_jsonl),
        lookback_days=max(1, int(args.lookback_days)),
        min_samples=max(1, int(args.min_samples)),
        model_id=str(model_identity["model_id"] or ""),
        model_hash=str(model_identity["model_hash"] or ""),
        dataset_hash=str(model_identity["dataset_hash"] or ""),
        model_role=str(model_identity["registry_role"] or ""),
        sources=tuple(_source_descriptor(path) for path in source_paths),
    )
    payload: Mapping[str, Any] = evidence
    if str(args.baseline_id or "").strip() or str(args.approved_by or "").strip():
        payload = build_governed_drift_baseline(
            evidence,
            baseline_id=str(args.baseline_id),
            approved_by=str(args.approved_by),
        )
    if args.output_json.exists():
        parser.error(f"refusing to overwrite immutable drift artifact: {args.output_json}")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(
        json.dumps(
            {
                "path": str(args.output_json),
                "status": payload.get("status"),
                "artifact_type": payload.get("artifact_type"),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
