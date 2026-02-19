"""TCA record rollups and bounded execution cost calibration."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from ai_trading.config.management import get_env
from ai_trading.execution.cost_model import CostModel
from ai_trading.logging import get_logger

logger = get_logger(__name__)


def _parse_ts(raw: Any) -> datetime | None:
    if raw is None:
        return None
    text = str(raw).strip()
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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def load_tca_records(
    path: str | Path,
    *,
    lookback_days: int | None = None,
    require_filled: bool = False,
) -> list[dict[str, Any]]:
    """Load JSONL TCA records with optional lookback filtering."""

    target = Path(path)
    if not target.exists():
        return []
    cutoff: datetime | None = None
    if lookback_days is not None and lookback_days > 0:
        cutoff = datetime.now(UTC) - timedelta(days=int(lookback_days))

    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            ts = _parse_ts(data.get("ts"))
            if cutoff is not None and (ts is None or ts < cutoff):
                continue
            if require_filled:
                status = str(data.get("status", "")).strip().lower()
                if status not in {"filled", "partially_filled", "partial_fill"}:
                    continue
            rows.append(data)
    return rows


def summarize_tca_records(records: list[dict[str, Any]]) -> dict[str, float]:
    """Compute compact TCA summary for calibration/reporting."""

    if not records:
        return {
            "sample_count": 0.0,
            "median_is_bps": 0.0,
            "p90_is_bps": 0.0,
            "median_spread_paid_bps": 0.0,
            "mean_fill_latency_ms": 0.0,
        }

    is_values: list[float] = []
    spread_values: list[float] = []
    latency_values: list[float] = []
    for row in records:
        try:
            is_bps = abs(float(row.get("is_bps", 0.0) or 0.0))
        except (TypeError, ValueError):
            is_bps = 0.0
        if math.isfinite(is_bps) and is_bps > 0:
            is_values.append(is_bps)
        try:
            spread_bps = float(row.get("spread_paid_bps", 0.0) or 0.0)
        except (TypeError, ValueError):
            spread_bps = 0.0
        if math.isfinite(spread_bps) and spread_bps > 0:
            spread_values.append(spread_bps)
        try:
            latency_ms = float(row.get("fill_latency_ms", 0.0) or 0.0)
        except (TypeError, ValueError):
            latency_ms = 0.0
        if math.isfinite(latency_ms) and latency_ms > 0:
            latency_values.append(latency_ms)

    sorted_is = sorted(is_values)
    if sorted_is:
        idx = int(math.ceil(0.9 * len(sorted_is))) - 1
        idx = max(0, min(idx, len(sorted_is) - 1))
        p90 = sorted_is[idx]
        med_is = median(sorted_is)
    else:
        p90 = 0.0
        med_is = 0.0
    med_spread = median(spread_values) if spread_values else 0.0
    mean_latency = (
        float(sum(latency_values) / len(latency_values)) if latency_values else 0.0
    )
    return {
        "sample_count": float(len(is_values)),
        "median_is_bps": float(med_is),
        "p90_is_bps": float(p90),
        "median_spread_paid_bps": float(med_spread),
        "mean_fill_latency_ms": float(mean_latency),
    }


def calibrate_cost_model_from_tca(
    *,
    tca_path: str | Path | None = None,
    model_path: str | Path | None = None,
    lookback_days: int | None = None,
) -> dict[str, Any]:
    """Load TCA data, calibrate bounded cost model, and persist params."""

    tca_target = Path(
        str(
            tca_path
            or get_env("AI_TRADING_TCA_PATH", "runtime/tca_records.jsonl")
        )
    )
    model_target = Path(
        str(
            model_path
            or get_env(
                "AI_TRADING_EXEC_COST_MODEL_PATH",
                "runtime/execution_cost_model.json",
            )
        )
    )
    lookback = (
        int(lookback_days)
        if lookback_days is not None
        else int(get_env("AI_TRADING_AFTER_HOURS_COST_FLOOR_LOOKBACK_DAYS", "45", cast=int))
    )
    min_samples = int(
        get_env("AI_TRADING_AFTER_HOURS_COST_FLOOR_MIN_SAMPLES", "80", cast=int)
    )
    require_filled = _as_bool(
        get_env("AI_TRADING_AFTER_HOURS_COST_FLOOR_REQUIRE_FILLED", "1")
    )
    quantile = float(
        get_env("AI_TRADING_AFTER_HOURS_COST_FLOOR_QUANTILE", "0.55", cast=float)
    )
    outlier_bps = float(
        get_env("AI_TRADING_AFTER_HOURS_COST_FLOOR_OUTLIER_BPS", "120", cast=float)
    )

    records = load_tca_records(
        tca_target,
        lookback_days=lookback,
        require_filled=require_filled,
    )
    summary = summarize_tca_records(records)
    model = CostModel.load(model_target)
    before = model.to_dict()
    updated = model.calibrate(
        records,
        min_samples=min_samples,
        quantile=quantile,
        outlier_bps=outlier_bps,
    )
    model.save(model_target)
    after = model.to_dict()

    result = {
        "tca_path": str(tca_target),
        "model_path": str(model_target),
        "records": len(records),
        "summary": summary,
        "before": before,
        "after": after,
        "calibrated": after.get("version") == updated.version,
    }
    logger.info(
        "TCA_COST_MODEL_CALIBRATED",
        extra={
            "records": len(records),
            "model_path": str(model_target),
            "version": after.get("version"),
        },
    )
    return result
