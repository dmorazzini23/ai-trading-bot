"""Build a rolling live execution cost model from runtime telemetry."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

from ai_trading.config.management import get_env
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


def _parse_ts(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
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


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _to_float(value)
        if parsed is not None:
            return parsed
    return None


def _first_text(*values: Any, default: str = "") -> str:
    for value in values:
        if value in (None, ""):
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def _mapping(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = row.get(key)
    return value if isinstance(value, Mapping) else {}


def _metric(row: Mapping[str, Any], *keys: str) -> float | None:
    context = _mapping(row, "context")
    cost = _mapping(row, "cost")
    market = _mapping(row, "market")
    for key in keys:
        parsed = _first_float(row.get(key), cost.get(key), market.get(key), context.get(key))
        if parsed is not None:
            return parsed
    return None


def _text_metric(row: Mapping[str, Any], *keys: str, default: str = "") -> str:
    context = _mapping(row, "context")
    cost = _mapping(row, "cost")
    market = _mapping(row, "market")
    values: list[Any] = []
    for key in keys:
        values.extend((row.get(key), cost.get(key), market.get(key), context.get(key)))
    return _first_text(*values, default=default)


def _percentile(values: Iterable[float], q: float) -> float | None:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return None
    quantile = max(0.0, min(float(q), 1.0))
    if len(clean) == 1:
        return float(clean[0])
    raw_index = quantile * (len(clean) - 1)
    lo = int(math.floor(raw_index))
    hi = int(math.ceil(raw_index))
    if lo == hi:
        return float(clean[lo])
    return float(clean[lo] + ((clean[hi] - clean[lo]) * (raw_index - lo)))


def _mean(values: Iterable[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError:
        return []
    with handle:
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


def _session_regime(row: Mapping[str, Any], ts: datetime | None) -> str:
    explicit = _text_metric(
        row,
        "session_regime",
        "market_session",
        "session",
        default="",
    ).lower()
    if explicit in {"opening", "midday", "closing", "offhours"}:
        return explicit
    if ts is None:
        return "unknown"
    try:
        from zoneinfo import ZoneInfo

        ts_et = ts.astimezone(ZoneInfo("America/New_York"))
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        return "unknown"
    minute = (ts_et.hour * 60) + ts_et.minute
    if ts_et.weekday() >= 5 or minute < (9 * 60 + 30) or minute >= (16 * 60):
        return "offhours"
    if minute < (10 * 60 + 15):
        return "opening"
    if minute >= (15 * 60 + 15):
        return "closing"
    return "midday"


def _timestamp(row: Mapping[str, Any]) -> datetime | None:
    for key in (
        "ts",
        "timestamp",
        "filled_at",
        "executed_at",
        "first_fill_ts",
        "submit_ts",
        "event_ts",
    ):
        parsed = _parse_ts(row.get(key))
        if parsed is not None:
            return parsed
    return None


def _observation_from_row(
    row: Mapping[str, Any],
    *,
    source: str,
    cutoff: datetime,
) -> dict[str, Any] | None:
    ts = _timestamp(row)
    if ts is None or ts < cutoff:
        return None
    symbol = _text_metric(row, "symbol", "asset_symbol", default="").upper()
    if not symbol:
        return None
    side = _text_metric(row, "side", "order_side", default="unknown").lower()
    if side in {"", "unknown"}:
        side = "unknown"
    session_regime = _session_regime(row, ts)
    spread_bps = _metric(
        row,
        "spread_bps",
        "quoted_spread_bps",
        "bid_ask_spread_bps",
        "quote_spread_bps",
    )
    quote_age_ms = _metric(row, "quote_age_ms", "quote_staleness_ms", "quote_age")
    slippage_bps = _metric(
        row,
        "slippage_bps",
        "arrival_slippage_bps",
        "realized_slippage_bps",
        "implementation_shortfall_bps",
        "is_bps",
    )
    commission_bps = _metric(row, "commission_bps", "fee_bps")
    explicit_total_cost_bps = _metric(
        row,
        "total_cost_bps",
        "execution_cost_bps",
        "realized_cost_bps",
    )
    if (
        spread_bps is None
        and quote_age_ms is None
        and slippage_bps is None
        and explicit_total_cost_bps is None
    ):
        return None
    half_spread_bps = None if spread_bps is None else max(float(spread_bps), 0.0) / 2.0
    adverse_slippage_bps = (
        None if slippage_bps is None else max(float(slippage_bps), 0.0)
    )
    commission_component = max(float(commission_bps or 0.0), 0.0)
    if explicit_total_cost_bps is not None:
        modeled_total_cost_bps = max(float(explicit_total_cost_bps), 0.0)
    else:
        modeled_total_cost_bps = (
            (half_spread_bps or 0.0)
            + (adverse_slippage_bps or 0.0)
            + commission_component
        )
    return {
        "source": source,
        "ts": ts,
        "symbol": symbol,
        "side": side,
        "session_regime": session_regime,
        "spread_bps": spread_bps,
        "quote_age_ms": quote_age_ms,
        "slippage_bps": slippage_bps,
        "adverse_slippage_bps": adverse_slippage_bps,
        "half_spread_bps": half_spread_bps,
        "commission_bps": commission_component,
        "modeled_total_cost_bps": modeled_total_cost_bps,
    }


def _summary_row(
    *,
    key: tuple[str, str, str],
    observations: list[dict[str, Any]],
    min_samples: int,
) -> dict[str, Any]:
    symbol, side, session_regime = key
    spread_values = [
        float(value)
        for obs in observations
        if (value := _to_float(obs.get("spread_bps"))) is not None
    ]
    quote_age_values = [
        float(value)
        for obs in observations
        if (value := _to_float(obs.get("quote_age_ms"))) is not None
    ]
    slippage_values = [
        float(value)
        for obs in observations
        if (value := _to_float(obs.get("slippage_bps"))) is not None
    ]
    adverse_slippage_values = [
        float(value)
        for obs in observations
        if (value := _to_float(obs.get("adverse_slippage_bps"))) is not None
    ]
    total_cost_values = [
        float(value)
        for obs in observations
        if (value := _to_float(obs.get("modeled_total_cost_bps"))) is not None
    ]
    last_observed_at = max(obs["ts"] for obs in observations if isinstance(obs.get("ts"), datetime))
    sample_count = len(total_cost_values)
    return {
        "symbol": symbol,
        "side": side,
        "session_regime": session_regime,
        "event_count": int(len(observations)),
        "sample_count": int(sample_count),
        "sufficient_samples": bool(sample_count >= int(min_samples)),
        "mean_spread_bps": _mean(spread_values),
        "p90_spread_bps": _percentile(spread_values, 0.90),
        "mean_quote_age_ms": _mean(quote_age_values),
        "p90_quote_age_ms": _percentile(quote_age_values, 0.90),
        "mean_slippage_bps": _mean(slippage_values),
        "p90_adverse_slippage_bps": _percentile(adverse_slippage_values, 0.90),
        "mean_total_cost_bps": _mean(total_cost_values),
        "p90_total_cost_bps": _percentile(total_cost_values, 0.90),
        "last_observed_at": last_observed_at.isoformat().replace("+00:00", "Z"),
    }


def build_live_cost_model(
    *,
    events_path: Path,
    fill_events_path: Path | None = None,
    tca_path: Path | None = None,
    window_minutes: int = 390,
    min_samples: int = 5,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return a rolling cost-model artifact from recent live execution rows."""

    generated_at = now.astimezone(UTC) if now is not None else datetime.now(UTC)
    cutoff = generated_at - timedelta(minutes=max(1, int(window_minutes)))
    observations: list[dict[str, Any]] = []
    sources = {
        "execution_quality_events": events_path,
        "fill_events": fill_events_path,
        "tca_records": tca_path,
    }
    for source, path in sources.items():
        if path is None:
            continue
        for row in _read_jsonl(path):
            observation = _observation_from_row(row, source=source, cutoff=cutoff)
            if observation is not None:
                observations.append(observation)

    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for obs in observations:
        key = (
            str(obs["symbol"]),
            str(obs["side"]),
            str(obs["session_regime"]),
        )
        buckets[key].append(obs)
    by_symbol_side_session = [
        _summary_row(key=key, observations=rows, min_samples=max(1, int(min_samples)))
        for key, rows in sorted(buckets.items())
        if rows
    ]
    total_costs = [
        float(value)
        for obs in observations
        if (value := _to_float(obs.get("modeled_total_cost_bps"))) is not None
    ]
    sufficient_rows = [
        row for row in by_symbol_side_session if bool(row.get("sufficient_samples"))
    ]
    available = bool(total_costs)
    status = "ready" if sufficient_rows else ("warming_up" if available else "unavailable")
    return {
        "schema_version": "1.0.0",
        "artifact_type": "live_cost_model",
        "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
        "source": "runtime_execution_telemetry",
        "window": {
            "minutes": int(max(1, window_minutes)),
            "cutoff": cutoff.isoformat().replace("+00:00", "Z"),
            "event_count": int(len(observations)),
            "sample_count": int(len(total_costs)),
            "bucket_count": int(len(by_symbol_side_session)),
            "sufficient_bucket_count": int(len(sufficient_rows)),
            "min_samples": int(max(1, min_samples)),
        },
        "status": {
            "available": available,
            "status": status,
            "mode": "observe",
            "reason": (
                "ok"
                if status == "ready"
                else ("insufficient_samples" if available else "no_recent_samples")
            ),
        },
        "observed": {
            "mean_total_cost_bps": _mean(total_costs),
            "p90_total_cost_bps": _percentile(total_costs, 0.90),
        },
        "by_symbol_side_session": by_symbol_side_session,
        "paths": {
            "execution_quality_events": str(events_path),
            "fill_events": str(fill_events_path) if fill_events_path is not None else None,
            "tca_records": str(tca_path) if tca_path is not None else None,
        },
    }


def _default_path(env_key: str, default_relative: str) -> Path:
    configured = str(
        get_env(env_key, default_relative, cast=str, resolve_aliases=False)
        or default_relative
    ).strip()
    return resolve_runtime_artifact_path(configured, default_relative=default_relative)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events-jsonl",
        default="",
        help="Execution-quality events JSONL path.",
    )
    parser.add_argument("--fill-events-jsonl", default="", help="Fill events JSONL path.")
    parser.add_argument("--tca-jsonl", default="", help="TCA records JSONL path.")
    parser.add_argument("--output-json", default="", help="Output live cost model JSON path.")
    parser.add_argument("--window-minutes", type=int, default=390)
    parser.add_argument("--min-samples", type=int, default=5)
    args = parser.parse_args(argv)

    events_path = (
        Path(args.events_jsonl).expanduser()
        if str(args.events_jsonl or "").strip()
        else _default_path(
            "AI_TRADING_EXEC_QUALITY_EVENTS_PATH",
            "runtime/execution_quality_events.jsonl",
        )
    )
    fill_events_path = (
        Path(args.fill_events_jsonl).expanduser()
        if str(args.fill_events_jsonl or "").strip()
        else _default_path("AI_TRADING_FILL_EVENTS_PATH", "runtime/fill_events.jsonl")
    )
    tca_path = (
        Path(args.tca_jsonl).expanduser()
        if str(args.tca_jsonl or "").strip()
        else _default_path("AI_TRADING_TCA_RECORDS_PATH", "runtime/tca_records.jsonl")
    )
    output_path = (
        Path(args.output_json).expanduser()
        if str(args.output_json or "").strip()
        else _default_path(
            "AI_TRADING_LIVE_COST_MODEL_PATH",
            "runtime/live_cost_model_latest.json",
        )
    )
    report = build_live_cost_model(
        events_path=events_path,
        fill_events_path=fill_events_path,
        tca_path=tca_path,
        window_minutes=max(1, int(args.window_minutes)),
        min_samples=max(1, int(args.min_samples)),
    )
    report["paths"]["report"] = str(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(
        json.dumps({"path": str(output_path), "status": report["status"]}, sort_keys=True)
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
