"""Daily execution-quality rollups from TCA records."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ai_trading.config.management import get_env
from ai_trading.logging import get_logger
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path

logger = get_logger(__name__)

_PHASE2_BASELINE_DEFAULT_PATH = "runtime/phase2_execution_baseline.json"


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


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_env_float(name: str) -> float | None:
    raw = str(get_env(name, "", cast=str) or "").strip()
    return _safe_float(raw)


def _phase2_baseline_path() -> Path:
    configured = str(
        get_env(
            "AI_TRADING_ROADMAP_PHASE2_BASELINE_PATH",
            _PHASE2_BASELINE_DEFAULT_PATH,
            cast=str,
        )
        or _PHASE2_BASELINE_DEFAULT_PATH
    ).strip()
    return Path(
        resolve_runtime_artifact_path(
            configured,
            default_relative=_PHASE2_BASELINE_DEFAULT_PATH,
            for_write=False,
        )
    )


def _load_phase2_baseline_payload() -> tuple[dict[str, Any], Path]:
    path = _phase2_baseline_path()
    if not path.exists():
        return {}, path
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning(
            "EXECUTION_REPORT_PHASE2_BASELINE_LOAD_FAILED",
            extra={"path": str(path)},
            exc_info=True,
        )
        return {}, path
    if not isinstance(parsed, dict):
        logger.warning(
            "EXECUTION_REPORT_PHASE2_BASELINE_INVALID",
            extra={"path": str(path), "reason": "root_not_object"},
        )
        return {}, path
    return dict(parsed), path


def _baseline_from_payload(payload: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    baselines_raw = payload.get("baselines")
    baselines = dict(baselines_raw) if isinstance(baselines_raw, Mapping) else {}
    for key in keys:
        if key in baselines:
            value = _safe_float(baselines.get(key))
            if value is not None:
                return float(value)
    for key in keys:
        value = _safe_float(payload.get(key))
        if value is not None:
            return float(value)
    return None


def _resolve_phase2_baseline_value(
    *,
    env_key: str,
    payload: Mapping[str, Any],
    payload_keys: tuple[str, ...],
) -> tuple[float | None, str]:
    env_value = _optional_env_float(env_key)
    if env_value is not None:
        return float(env_value), "env"
    file_value = _baseline_from_payload(payload, payload_keys)
    if file_value is not None:
        return float(file_value), "file"
    return None, "none"


def _parse_record_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _build_phase2_execution_edge_summary(
    records: list[dict[str, Any]],
    *,
    window_days: int | None = None,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    enabled = bool(get_env("AI_TRADING_ROADMAP_PHASE2_ENABLED", False, cast=bool))
    resolved_window_days = (
        int(window_days)
        if window_days is not None
        else int(get_env("AI_TRADING_ROADMAP_PHASE2_WINDOW_DAYS", 30, cast=int))
    )
    resolved_window_days = max(1, int(resolved_window_days))
    min_slippage_improvement_pct = float(
        get_env("AI_TRADING_ROADMAP_PHASE2_MIN_SLIPPAGE_IMPROVEMENT_PCT", 10.0, cast=float)
    )
    max_fill_rate_degradation_pct = float(
        get_env("AI_TRADING_ROADMAP_PHASE2_MAX_FILL_RATE_DEGRADATION_PCT", 5.0, cast=float)
    )
    max_reject_rate = float(get_env("AI_TRADING_ROADMAP_PHASE2_MAX_REJECT_RATE", 0.03, cast=float))
    max_execution_drift_bps = float(
        get_env("AI_TRADING_ROADMAP_PHASE2_MAX_EXECUTION_DRIFT_BPS", 25.0, cast=float)
    )
    max_stale_pending_increase = float(
        get_env("AI_TRADING_ROADMAP_PHASE2_MAX_STALE_PENDING_INCREASE", 0.0, cast=float)
    )
    baseline_payload, baseline_path = _load_phase2_baseline_payload()

    baseline_slippage_median_bps, baseline_slippage_source = _resolve_phase2_baseline_value(
        env_key="AI_TRADING_ROADMAP_PHASE2_BASELINE_SLIPPAGE_MEDIAN_BPS",
        payload=baseline_payload,
        payload_keys=("slippage_median_abs_bps", "baseline_slippage_median_abs_bps"),
    )
    baseline_fill_rate, baseline_fill_rate_source = _resolve_phase2_baseline_value(
        env_key="AI_TRADING_ROADMAP_PHASE2_BASELINE_FILL_RATE",
        payload=baseline_payload,
        payload_keys=("target_limit_fill_rate", "baseline_target_limit_fill_rate"),
    )
    baseline_stale_pending_count, baseline_stale_pending_source = _resolve_phase2_baseline_value(
        env_key="AI_TRADING_ROADMAP_PHASE2_BASELINE_STALE_PENDING_COUNT",
        payload=baseline_payload,
        payload_keys=("stale_pending_count", "baseline_stale_pending_count"),
    )

    resolved_now_utc = now_utc or datetime.now(UTC)
    window_start = resolved_now_utc - timedelta(days=resolved_window_days)
    window_records: list[dict[str, Any]] = []
    for record in records:
        ts = _parse_record_ts(record.get("ts"))
        if ts is None or ts >= window_start:
            window_records.append(record)

    attempted = 0
    rejected = 0
    slippage_values: list[float] = []
    drift_values: list[float] = []
    stale_pending_count = 0
    limit_rows: list[tuple[bool, bool]] = []
    for row in window_records:
        status = str(row.get("status", "")).strip().lower()
        if status:
            attempted += 1
        if status == "rejected":
            rejected += 1
        if bool(row.get("pending_terminal_nonfill", False)):
            stale_pending_count += 1

        is_bps = _safe_float(row.get("is_bps"))
        if is_bps is not None and status not in {"rejected", "canceled"} and not bool(
            row.get("pending_event", False)
        ):
            slippage_values.append(abs(float(is_bps)))

        drift = _safe_float(row.get("execution_drift_bps"))
        if drift is None and is_bps is not None:
            expected_is = _safe_float(row.get("expected_is_bps"))
            if expected_is is not None:
                drift = abs(float(is_bps) - float(expected_is))
        if drift is not None:
            drift_values.append(abs(float(drift)))

        order_type = str(row.get("order_type", "")).strip().lower()
        if order_type in {"limit", "stop_limit"}:
            has_target_offset = any(
                _safe_float(row.get(name)) is not None
                for name in (
                    "target_limit_offset_bps",
                    "target_offset_bps",
                    "limit_offset_bps",
                    "midpoint_offset_bps",
                )
            )
            is_fill = status in {"filled", "partially_filled"} or bool(row.get("partial_fill", False))
            limit_rows.append((is_fill, has_target_offset))

    target_limit_rows = [row for row in limit_rows if row[1]]
    if not target_limit_rows:
        target_limit_rows = list(limit_rows)
    target_limit_attempted = len(target_limit_rows)
    target_limit_filled = sum(1 for is_fill, _ in target_limit_rows if is_fill)

    current_slippage_median_bps = _percentile(slippage_values, 50.0)
    current_fill_rate = (
        float(target_limit_filled) / float(target_limit_attempted)
        if target_limit_attempted > 0
        else None
    )
    current_reject_rate = float(rejected) / float(attempted) if attempted > 0 else None
    current_execution_drift_p90_bps = _percentile(drift_values, 90.0)
    current_stale_pending_count = float(stale_pending_count)

    slippage_improvement_pct: float | None = None
    if (
        baseline_slippage_median_bps is not None
        and abs(float(baseline_slippage_median_bps)) > 0.0
        and current_slippage_median_bps is not None
    ):
        slippage_improvement_pct = (
            (float(baseline_slippage_median_bps) - float(current_slippage_median_bps))
            / abs(float(baseline_slippage_median_bps))
        ) * 100.0

    fill_rate_degradation_pct: float | None = None
    if baseline_fill_rate is not None and float(baseline_fill_rate) > 0.0 and current_fill_rate is not None:
        fill_rate_degradation_pct = max(
            0.0,
            ((float(baseline_fill_rate) - float(current_fill_rate)) / float(baseline_fill_rate)) * 100.0,
        )

    gates: dict[str, bool | None] = {
        "slippage_improvement": (
            slippage_improvement_pct >= float(min_slippage_improvement_pct)
            if slippage_improvement_pct is not None
            else None
        ),
        "fill_rate_degradation": (
            fill_rate_degradation_pct <= float(max_fill_rate_degradation_pct)
            if fill_rate_degradation_pct is not None
            else None
        ),
        "reject_rate_slo": (
            current_reject_rate <= float(max_reject_rate) if current_reject_rate is not None else None
        ),
        "execution_drift_slo": (
            current_execution_drift_p90_bps <= float(max_execution_drift_bps)
            if current_execution_drift_p90_bps is not None
            else None
        ),
        "stale_pending_incidents": (
            current_stale_pending_count
            <= float(baseline_stale_pending_count) + float(max_stale_pending_increase)
            if baseline_stale_pending_count is not None
            else None
        ),
    }
    effective_gates = {name: bool(value) if value is not None else False for name, value in gates.items()}
    gate_passed = bool(all(effective_gates.values())) if enabled else False

    return {
        "enabled": enabled,
        "gate_passed": gate_passed,
        "window_days": resolved_window_days,
        "window_start_utc": window_start.isoformat(),
        "evaluated_at_utc": resolved_now_utc.isoformat(),
        "records_in_window": len(window_records),
        "attempted_orders": attempted,
        "metrics": {
            "slippage_median_abs_bps": current_slippage_median_bps,
            "target_limit_fill_rate": current_fill_rate,
            "reject_rate": current_reject_rate,
            "execution_drift_p90_bps": current_execution_drift_p90_bps,
            "stale_pending_count": current_stale_pending_count,
            "target_limit_attempted": target_limit_attempted,
            "target_limit_filled": target_limit_filled,
        },
        "baselines": {
            "slippage_median_abs_bps": baseline_slippage_median_bps,
            "target_limit_fill_rate": baseline_fill_rate,
            "stale_pending_count": baseline_stale_pending_count,
            "sources": {
                "slippage_median_abs_bps": baseline_slippage_source,
                "target_limit_fill_rate": baseline_fill_rate_source,
                "stale_pending_count": baseline_stale_pending_source,
            },
            "path": str(baseline_path),
        },
        "deltas": {
            "slippage_improvement_pct": slippage_improvement_pct,
            "fill_rate_degradation_pct": fill_rate_degradation_pct,
            "stale_pending_increase": (
                current_stale_pending_count - float(baseline_stale_pending_count)
                if baseline_stale_pending_count is not None
                else None
            ),
        },
        "thresholds": {
            "min_slippage_improvement_pct": min_slippage_improvement_pct,
            "max_fill_rate_degradation_pct": max_fill_rate_degradation_pct,
            "max_reject_rate": max_reject_rate,
            "max_execution_drift_bps": max_execution_drift_bps,
            "max_stale_pending_increase": max_stale_pending_increase,
        },
        "gates": gates,
        "effective_gates": effective_gates,
    }


def build_phase2_execution_edge_summary(
    records: list[dict[str, Any]],
    *,
    window_days: int | None = None,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    return _build_phase2_execution_edge_summary(
        records,
        window_days=window_days,
        now_utc=now_utc,
    )


def load_execution_records(path: str | Path) -> list[dict[str, Any]]:
    return _parse_jsonl(path)


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
        "roadmap": {
            "phase_2_execution_edge": _build_phase2_execution_edge_summary(records),
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
        rollup_zone: tzinfo = ZoneInfo(tz_name)
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
