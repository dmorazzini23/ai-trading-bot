"""Delayed reference-feed reconciliation for execution decisions."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

from ai_trading.analytics.feed_drift import (
    build_symbol_reliability_scores,
    compute_drift_metrics,
    derive_signal_agreement,
    fetch_reference_minute_bar_snapshot,
)
from ai_trading.config.management import get_env
from ai_trading.logging import get_logger
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path

logger = get_logger(__name__)


def _resolve_artifact_path(env_key: str, default_relative: str, *, for_write: bool) -> Path:
    configured = str(get_env(env_key, default_relative, cast=str, resolve_aliases=False) or default_relative)
    return cast(
        Path,
        resolve_runtime_artifact_path(
            configured,
            default_relative=default_relative,
            for_write=for_write,
        ),
    )


def _iter_jsonl(path: Path, *, max_rows: int) -> Iterable[dict[str, Any]]:
    if max_rows <= 0 or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        return []
    return rows[-max_rows:]


def _coerce_ts(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def run_reference_reconciliation_once(
    *,
    max_rows: int | None = None,
    min_lag_minutes: int | None = None,
) -> dict[str, Any]:
    """Reconcile recent decisions against delayed reference bars."""

    configured_max_rows = int(get_env("AI_TRADING_REFERENCE_RECONCILE_LOOKBACK", 500, cast=int, resolve_aliases=False) or 500)
    configured_min_lag = int(get_env("AI_TRADING_REFERENCE_RECONCILE_MIN_LAG_MINUTES", 16, cast=int, resolve_aliases=False) or 16)
    configured_reliability_lookback = int(
        get_env("AI_TRADING_FEED_RELIABILITY_LOOKBACK", 2000, cast=int, resolve_aliases=False) or 2000
    )
    configured_drift_disagreement_bps = float(
        get_env("AI_TRADING_FEED_DISAGREEMENT_BPS", 25.0, cast=float, resolve_aliases=False) or 25.0
    )
    configured_reliability_min_samples = int(
        get_env("AI_TRADING_FEED_RELIABILITY_MIN_SAMPLES", 3, cast=int, resolve_aliases=False) or 3
    )
    row_limit = max(1, int(max_rows if max_rows is not None else configured_max_rows))
    lag_minutes = max(1, int(min_lag_minutes if min_lag_minutes is not None else configured_min_lag))
    reliability_lookback = max(10, int(configured_reliability_lookback))

    decisions_path = _resolve_artifact_path(
        "AI_TRADING_DUAL_FEED_DECISIONS_PATH",
        "runtime/dual_feed_decisions.jsonl",
        for_write=False,
    )
    output_path = _resolve_artifact_path(
        "AI_TRADING_REFERENCE_RECONCILIATION_PATH",
        "runtime/reference_reconciliation.jsonl",
        for_write=True,
    )
    reliability_path = _resolve_artifact_path(
        "AI_TRADING_FEED_RELIABILITY_PATH",
        "runtime/feed_reliability_scores.json",
        for_write=True,
    )

    decisions = list(_iter_jsonl(decisions_path, max_rows=row_limit))
    reconciled = list(_iter_jsonl(output_path, max_rows=row_limit * 2))
    reconciled_ids = {
        str(row.get("decision_id") or "").strip()
        for row in reconciled
        if str(row.get("decision_id") or "").strip()
    }
    now_utc = datetime.now(UTC)
    processed = 0
    written = 0
    new_rows: list[dict[str, Any]] = []

    if not decisions:
        return {"processed": 0, "written": 0, "path": str(output_path), "reliability_path": str(reliability_path)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for row in decisions:
            decision_id = str(row.get("decision_id") or "").strip()
            if not decision_id or decision_id in reconciled_ids:
                continue
            decision_ts = _coerce_ts(row.get("decision_ts") or row.get("ts"))
            if decision_ts is None:
                continue
            age_minutes = (now_utc - decision_ts).total_seconds() / 60.0
            if age_minutes < float(lag_minutes):
                continue
            symbol = str(row.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            processed += 1
            reference_snapshot = fetch_reference_minute_bar_snapshot(
                symbol,
                decision_ts=decision_ts,
                feed=str(row.get("reference_feed") or ""),
            )
            metrics = compute_drift_metrics(
                execution_price=_safe_float(row.get("execution_price")),
                reference_price=_safe_float(reference_snapshot.get("price")),
                execution_bid=_safe_float(row.get("execution_bid")),
                execution_ask=_safe_float(row.get("execution_ask")),
                reference_bid=_safe_float(row.get("reference_bid")),
                reference_ask=_safe_float(row.get("reference_ask")),
                execution_volume=_safe_float(row.get("execution_volume")),
                reference_volume=_safe_float(reference_snapshot.get("volume")),
            )
            context = row.get("context")
            context_map = dict(context) if isinstance(context, Mapping) else {}
            signal_metrics = derive_signal_agreement(
                outcome=str(row.get("outcome") or ""),
                side=row.get("side"),
                signal_side=context_map.get("signal_side") or row.get("signal_side"),
                signal_strength=context_map.get("signal_strength") or row.get("signal_strength"),
                signal_confidence=context_map.get("signal_confidence") or row.get("signal_confidence"),
                price_drift_bps=metrics.get("price_drift_bps"),
                drift_disagreement_bps=configured_drift_disagreement_bps,
            )
            signal_weight_value = _safe_float(
                context_map.get("signal_weight") if context_map else row.get("signal_weight")
            )
            if signal_weight_value is not None:
                signal_metrics["signal_weight"] = float(signal_weight_value)
            payload = {
                "ts": now_utc.isoformat(),
                "decision_id": decision_id,
                "decision_ts": decision_ts.isoformat(),
                "symbol": symbol,
                "outcome": row.get("outcome"),
                "side": row.get("side"),
                "quantity": row.get("quantity"),
                "execution_feed": row.get("execution_feed"),
                "reference_feed": reference_snapshot.get("feed"),
                "execution_price": row.get("execution_price"),
                "reference_price": reference_snapshot.get("price"),
                "reference_bar_ts": reference_snapshot.get("bar_ts"),
                "metrics": metrics,
                "signal_agreement": signal_metrics.get("signal_agreement"),
                "signal_disagreement": signal_metrics.get("signal_disagreement"),
                "signal_metrics": signal_metrics,
            }
            handle.write(json.dumps(payload, sort_keys=True, default=str))
            handle.write("\n")
            written += 1
            new_rows.append(payload)

    recent_reconciliations = list(_iter_jsonl(output_path, max_rows=reliability_lookback))
    if new_rows:
        recent_reconciliations.extend(new_rows)
    scored_rows = recent_reconciliations[-reliability_lookback:]
    reliability_scores = build_symbol_reliability_scores(
        scored_rows,
        drift_disagreement_bps=configured_drift_disagreement_bps,
        min_samples=max(1, configured_reliability_min_samples),
    )
    reliability_payload = {
        "ts": now_utc.isoformat(),
        "lookback_rows": len(scored_rows),
        "scores": reliability_scores,
    }
    reliability_path.parent.mkdir(parents=True, exist_ok=True)
    with reliability_path.open("w", encoding="utf-8") as handle:
        json.dump(reliability_payload, handle, sort_keys=True, indent=2, default=str)

    logger.info(
        "REFERENCE_RECONCILE_COMPLETE",
        extra={
            "processed": processed,
            "written": written,
            "decisions_path": str(decisions_path),
            "output_path": str(output_path),
            "reliability_path": str(reliability_path),
            "scored_symbols": len(reliability_scores),
        },
    )
    return {
        "processed": processed,
        "written": written,
        "path": str(output_path),
        "reliability_path": str(reliability_path),
        "scored_symbols": len(reliability_scores),
    }


__all__ = ["run_reference_reconciliation_once"]
