"""Build daily ML shadow evaluation reports from runtime JSONL telemetry."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd

from ai_trading.data.historical_bars import load_historical_bars
from ai_trading.logging import get_logger

logger = get_logger(__name__)


def _load_shadow_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "ML_SHADOW_REPORT_INVALID_JSONL_ROW",
                    extra={"path": str(path), "line_number": line_number},
                )
                continue
            if not isinstance(payload, Mapping):
                continue
            if str(payload.get("mode") or "ml_signal_shadow") != "ml_signal_shadow":
                continue
            rows.append(dict(payload))
    return rows


def _bar_timestamp_text(row: Mapping[str, Any]) -> str:
    market = row.get("market")
    if isinstance(market, Mapping):
        return str(market.get("bar_timestamp") or "")
    return ""


def _is_daily_frame_row(row: Mapping[str, Any]) -> bool:
    return "T00:00:00" in _bar_timestamp_text(row)


def _parse_timestamp(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    try:
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return cast(pd.Timestamp, parsed)


def _telemetry_timestamp(row: Mapping[str, Any]) -> pd.Timestamp | None:
    return _parse_timestamp(row.get("ts"))


def _frame_filtered_rows(
    rows: list[dict[str, Any]],
    frame_filter: str,
) -> list[dict[str, Any]]:
    normalized = str(frame_filter or "all").strip().lower()
    if normalized == "all":
        return rows
    if normalized == "minute":
        return [row for row in rows if not _is_daily_frame_row(row)]
    if normalized == "daily":
        return [row for row in rows if _is_daily_frame_row(row)]
    raise ValueError(f"Unsupported frame filter: {frame_filter}")


def _provider_healthy_primary(row: Mapping[str, Any]) -> bool:
    provider = row.get("provider")
    if not isinstance(provider, Mapping):
        return False
    status = str(provider.get("status") or "").strip().lower()
    active = str(provider.get("active") or "").strip().lower()
    primary = str(provider.get("primary") or "").strip().lower()
    if bool(provider.get("using_backup")):
        return False
    if not primary or not active or active != primary:
        return False
    return status in {"healthy", "ok", "ready"}


def _provider_filtered_rows(
    rows: list[dict[str, Any]],
    provider_filter: str,
) -> list[dict[str, Any]]:
    normalized = str(provider_filter or "all").strip().lower()
    if normalized == "all":
        return rows
    if normalized == "healthy-primary":
        return [row for row in rows if _provider_healthy_primary(row)]
    raise ValueError(f"Unsupported provider filter: {provider_filter}")


def _since_filtered_rows(
    rows: list[dict[str, Any]],
    since: pd.Timestamp | None,
) -> list[dict[str, Any]]:
    if since is None:
        return rows
    return [
        row
        for row in rows
        if (row_ts := _telemetry_timestamp(row)) is not None and row_ts >= since
    ]


def _filter_shadow_rows(
    rows: list[dict[str, Any]],
    *,
    frame_filter: str,
    provider_filter: str,
    since: pd.Timestamp | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    after_since = _since_filtered_rows(rows, since)
    after_frame = _frame_filtered_rows(after_since, frame_filter)
    after_provider = _provider_filtered_rows(after_frame, provider_filter)
    return after_provider, {
        "raw_rows": int(len(rows)),
        "after_since": int(len(after_since)),
        "after_frame_filter": int(len(after_frame)),
        "after_provider_filter": int(len(after_provider)),
    }


def _bool_value(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _load_bars_by_symbol(data_dir: Path, timestamp_col: str) -> dict[str, pd.DataFrame]:
    bars: dict[str, pd.DataFrame] = {}
    if not data_dir.is_dir():
        return bars
    for csv_path in sorted(data_dir.glob("*.csv")):
        symbol = csv_path.stem.upper()
        try:
            frame, _report = load_historical_bars(csv_path, timestamp_col=timestamp_col)
        except (OSError, ValueError, TypeError) as exc:
            logger.warning(
                "ML_SHADOW_REPORT_BAR_LOAD_FAILED",
                extra={"symbol": symbol, "path": str(csv_path), "error": str(exc)},
            )
            continue
        if isinstance(frame.index, pd.DatetimeIndex) and not frame.empty:
            bars[symbol] = frame.sort_index(kind="stable")
    return bars


def _row_timestamp(row: Mapping[str, Any]) -> pd.Timestamp | None:
    market = row.get("market")
    raw_ts: Any = None
    if isinstance(market, Mapping):
        raw_ts = market.get("bar_timestamp") or market.get("quote_timestamp")
    if raw_ts in (None, ""):
        raw_ts = row.get("ts")
    return _parse_timestamp(raw_ts)


def _entry_close(row: Mapping[str, Any]) -> float | None:
    market = row.get("market")
    if isinstance(market, Mapping):
        return _finite_float(market.get("entry_close"))
    return None


def _net_markout_bps(
    row: Mapping[str, Any],
    bars_by_symbol: Mapping[str, pd.DataFrame],
    *,
    horizon_bars: int,
    fee_bps: float,
    slippage_bps: float,
) -> float | None:
    symbol = str(row.get("symbol") or "").strip().upper()
    frame = bars_by_symbol.get(symbol)
    if frame is None or frame.empty:
        return None
    timestamp = _row_timestamp(row)
    if timestamp is None:
        return None
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        return None
    position = int(index.searchsorted(timestamp, side="left"))
    if position >= len(frame):
        return None
    future_position = position + max(1, int(horizon_bars))
    if future_position >= len(frame):
        return None
    entry = _entry_close(row)
    if entry is None or entry <= 0.0:
        entry = _finite_float(frame["close"].iloc[position])
    future = _finite_float(frame["close"].iloc[future_position])
    if entry is None or future is None or entry <= 0.0:
        return None
    gross = ((future / entry) - 1.0) * 10000.0
    costs = (2.0 * max(0.0, float(fee_bps))) + (2.0 * max(0.0, float(slippage_bps)))
    return float(gross - costs)


def _decision_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    champion_trade = sum(1 for row in rows if _bool_value(row, "champion_would_trade"))
    challenger_trade = sum(1 for row in rows if _bool_value(row, "challenger_would_trade"))
    both_trade = sum(
        1
        for row in rows
        if _bool_value(row, "champion_would_trade")
        and _bool_value(row, "challenger_would_trade")
    )
    champion_only = sum(
        1
        for row in rows
        if _bool_value(row, "champion_would_trade")
        and not _bool_value(row, "challenger_would_trade")
    )
    challenger_only = sum(
        1
        for row in rows
        if _bool_value(row, "challenger_would_trade")
        and not _bool_value(row, "champion_would_trade")
    )
    neither = total - both_trade - champion_only - challenger_only
    agreement = sum(
        1
        for row in rows
        if _bool_value(row, "champion_would_trade")
        == _bool_value(row, "challenger_would_trade")
    )
    deltas = [
        value
        for row in rows
        if (value := _finite_float(row.get("probability_delta"))) is not None
    ]
    spreads = [
        value
        for row in rows
        if isinstance(row.get("market"), Mapping)
        and (value := _finite_float(cast(Mapping[str, Any], row["market"]).get("spread_bps")))
        is not None
    ]
    skew_breaches = sum(
        1
        for row in rows
        if isinstance(row.get("skew"), Mapping)
        and bool(cast(Mapping[str, Any], row["skew"]).get("breached"))
    )
    return {
        "rows": total,
        "agreement_count": agreement,
        "agreement_rate": _rate(agreement, total),
        "champion_trade_count": champion_trade,
        "challenger_trade_count": challenger_trade,
        "both_trade_count": both_trade,
        "champion_only_count": champion_only,
        "challenger_only_count": challenger_only,
        "neither_trade_count": neither,
        "mean_probability_delta": _mean(deltas),
        "mean_spread_bps": _mean(spreads),
        "skew_breach_count": skew_breaches,
        "skew_breach_rate": _rate(skew_breaches, total),
    }


def _hour_bucket(row: Mapping[str, Any]) -> str | None:
    timestamp = _row_timestamp(row)
    if timestamp is None:
        return None
    return f"{timestamp.hour:02d}:00Z"


def _markout_summary(
    rows: list[dict[str, Any]],
    bars_by_symbol: Mapping[str, pd.DataFrame],
    *,
    horizon_bars: int,
    fee_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    champion_markouts: list[float] = []
    challenger_markouts: list[float] = []
    shadow_only_markouts: list[float] = []
    by_symbol: dict[str, list[float]] = {}
    by_hour: dict[str, list[float]] = {}
    for row in rows:
        markout = _net_markout_bps(
            row,
            bars_by_symbol,
            horizon_bars=horizon_bars,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )
        if markout is None:
            continue
        champion_would_trade = _bool_value(row, "champion_would_trade")
        challenger_would_trade = _bool_value(row, "challenger_would_trade")
        if champion_would_trade:
            champion_markouts.append(markout)
        if challenger_would_trade:
            challenger_markouts.append(markout)
            symbol = str(row.get("symbol") or "").strip().upper() or "UNKNOWN"
            by_symbol.setdefault(symbol, []).append(markout)
            if (hour_bucket := _hour_bucket(row)) is not None:
                by_hour.setdefault(hour_bucket, []).append(markout)
        if challenger_would_trade and not champion_would_trade:
            shadow_only_markouts.append(markout)
    symbol_rows = [
        {
            "symbol": symbol,
            "samples": len(values),
            "mean_net_markout_bps": _mean(values),
            "positive_rate": _rate(sum(1 for value in values if value > 0.0), len(values)),
        }
        for symbol, values in by_symbol.items()
    ]
    def _symbol_sort_key(item: Mapping[str, Any]) -> tuple[bool, float]:
        mean_value = _finite_float(item.get("mean_net_markout_bps"))
        return (mean_value is None, mean_value if mean_value is not None else -1e9)

    symbol_rows.sort(key=_symbol_sort_key, reverse=True)
    hour_rows = [
        {
            "hour": hour,
            "samples": len(values),
            "mean_net_markout_bps": _mean(values),
            "positive_rate": _rate(sum(1 for value in values if value > 0.0), len(values)),
        }
        for hour, values in by_hour.items()
    ]
    hour_rows.sort(key=_symbol_sort_key, reverse=True)
    return {
        "horizon_bars": int(horizon_bars),
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
        "champion_samples": len(champion_markouts),
        "champion_mean_net_markout_bps": _mean(champion_markouts),
        "champion_positive_rate": _rate(
            sum(1 for value in champion_markouts if value > 0.0),
            len(champion_markouts),
        ),
        "challenger_samples": len(challenger_markouts),
        "challenger_mean_net_markout_bps": _mean(challenger_markouts),
        "challenger_positive_rate": _rate(
            sum(1 for value in challenger_markouts if value > 0.0),
            len(challenger_markouts),
        ),
        "shadow_only_samples": len(shadow_only_markouts),
        "shadow_only_mean_net_markout_bps": _mean(shadow_only_markouts),
        "shadow_only_positive_rate": _rate(
            sum(1 for value in shadow_only_markouts if value > 0.0),
            len(shadow_only_markouts),
        ),
        "best_symbols": symbol_rows[:15],
        "worst_symbols": list(reversed(symbol_rows[-15:])),
        "best_hours": hour_rows[:10],
        "worst_hours": list(reversed(hour_rows[-10:])),
    }


def _parse_horizon_bars(args: argparse.Namespace) -> list[int]:
    primary = max(1, int(getattr(args, "horizon_bars", 1)))
    raw_list = str(getattr(args, "horizon_bars_list", "") or "").strip()
    if not raw_list:
        return [primary]
    horizons: list[int] = [primary]
    for token in raw_list.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            horizon = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid horizon in --horizon-bars-list: {token}") from exc
        horizons.append(max(1, horizon))
    return sorted(set(horizons)) or [primary]


def _sample_gate(
    rows: list[dict[str, Any]],
    *,
    min_informational_rows: int,
    min_review_rows: int,
) -> dict[str, Any]:
    row_count = int(len(rows))
    informational = max(1, int(min_informational_rows))
    review = max(informational, int(min_review_rows))
    if row_count >= review:
        status = "review_eligible"
    elif row_count >= informational:
        status = "early_warning"
    else:
        status = "insufficient"
    return {
        "status": status,
        "rows": row_count,
        "min_informational_rows": informational,
        "min_review_rows": review,
        "review_eligible": bool(row_count >= review),
    }


def _provider_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: Counter[str] = Counter()
    active: Counter[str] = Counter()
    healthy_primary = 0
    using_backup = 0
    missing = 0
    for row in rows:
        provider = row.get("provider")
        if not isinstance(provider, Mapping):
            missing += 1
            continue
        statuses[str(provider.get("status") or "unknown").strip().lower() or "unknown"] += 1
        active[str(provider.get("active") or "unknown").strip().lower() or "unknown"] += 1
        if _provider_healthy_primary(row):
            healthy_primary += 1
        if bool(provider.get("using_backup")):
            using_backup += 1
    total = len(rows)
    return {
        "rows": int(total),
        "missing_provider_rows": int(missing),
        "healthy_primary_rows": int(healthy_primary),
        "healthy_primary_rate": _rate(healthy_primary, total),
        "using_backup_rows": int(using_backup),
        "using_backup_rate": _rate(using_backup, total),
        "statuses": dict(sorted(statuses.items())),
        "active_providers": dict(sorted(active.items())),
    }


def _cost_observation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    spreads: list[float] = []
    quote_ages: list[float] = []
    missing_spread = 0
    missing_quote_age = 0
    for row in rows:
        market = row.get("market")
        cost = row.get("cost")
        spread: float | None = None
        quote_age: float | None = None
        if isinstance(cost, Mapping):
            spread = _finite_float(cost.get("spread_bps"))
            quote_age = _finite_float(cost.get("quote_age_ms"))
        if isinstance(market, Mapping):
            spread = spread if spread is not None else _finite_float(market.get("spread_bps"))
            quote_age = quote_age if quote_age is not None else _finite_float(market.get("quote_age_ms"))
        if spread is None:
            missing_spread += 1
        else:
            spreads.append(spread)
        if quote_age is None:
            missing_quote_age += 1
        else:
            quote_ages.append(quote_age)
    return {
        "rows": int(len(rows)),
        "mean_spread_bps": _mean(spreads),
        "max_spread_bps": max(spreads) if spreads else None,
        "mean_quote_age_ms": _mean(quote_ages),
        "max_quote_age_ms": max(quote_ages) if quote_ages else None,
        "missing_spread_rows": int(missing_spread),
        "missing_quote_age_rows": int(missing_quote_age),
    }


def _cost_fields(row: Mapping[str, Any]) -> tuple[float | None, float | None]:
    spread: float | None = None
    quote_age: float | None = None
    market = row.get("market")
    cost = row.get("cost")
    if isinstance(cost, Mapping):
        spread = _finite_float(cost.get("spread_bps"))
        quote_age = _finite_float(cost.get("quote_age_ms"))
    if isinstance(market, Mapping):
        spread = spread if spread is not None else _finite_float(market.get("spread_bps"))
        quote_age = quote_age if quote_age is not None else _finite_float(market.get("quote_age_ms"))
    return spread, quote_age


def _decision_type(row: Mapping[str, Any]) -> str:
    champion = _bool_value(row, "champion_would_trade")
    challenger = _bool_value(row, "challenger_would_trade")
    if champion and challenger:
        return "both_trade"
    if champion:
        return "champion_only"
    if challenger:
        return "challenger_only"
    return "neither_trade"


def _row_side(row: Mapping[str, Any]) -> str:
    for source in (row, row.get("market"), row.get("cost")):
        if not isinstance(source, Mapping):
            continue
        raw_side = source.get("side") or source.get("order_side") or source.get("signal_side")
        if raw_side not in (None, ""):
            return str(raw_side).strip().lower() or "unknown"
    return "unknown"


def _row_provider(row: Mapping[str, Any]) -> str:
    provider = row.get("provider")
    if not isinstance(provider, Mapping):
        return "unknown"
    active = str(provider.get("active") or "").strip().lower()
    status = str(provider.get("status") or "").strip().lower()
    if active and status:
        return f"{active}:{status}"
    return active or status or "unknown"


def _quantile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(values, min(max(float(quantile), 0.0), 1.0)))


def _cost_group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    spreads: list[float] = []
    quote_ages: list[float] = []
    missing_spread = 0
    missing_quote_age = 0
    for row in rows:
        spread, quote_age = _cost_fields(row)
        if spread is None:
            missing_spread += 1
        else:
            spreads.append(spread)
        if quote_age is None:
            missing_quote_age += 1
        else:
            quote_ages.append(quote_age)
    return {
        "rows": int(len(rows)),
        "mean_spread_bps": _mean(spreads),
        "p50_spread_bps": _quantile(spreads, 0.50),
        "p90_spread_bps": _quantile(spreads, 0.90),
        "max_spread_bps": max(spreads) if spreads else None,
        "mean_quote_age_ms": _mean(quote_ages),
        "p50_quote_age_ms": _quantile(quote_ages, 0.50),
        "p90_quote_age_ms": _quantile(quote_ages, 0.90),
        "max_quote_age_ms": max(quote_ages) if quote_ages else None,
        "missing_spread_rows": int(missing_spread),
        "missing_quote_age_rows": int(missing_quote_age),
    }


def _grouped_cost_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _build_group(key_func: Any) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            key = str(key_func(row) or "unknown")
            grouped.setdefault(key, []).append(row)
        records = [
            {"key": key, **_cost_group_summary(group_rows)}
            for key, group_rows in grouped.items()
        ]
        records.sort(key=lambda item: (-int(item["rows"]), str(item["key"])))
        return records

    return {
        "by_symbol": _build_group(
            lambda row: str(row.get("symbol") or "").strip().upper() or "UNKNOWN"
        ),
        "by_hour": _build_group(lambda row: _hour_bucket(row) or "unknown"),
        "by_provider": _build_group(_row_provider),
        "by_side": _build_group(_row_side),
        "by_decision_type": _build_group(_decision_type),
    }


def _microstructure_reasons(
    row: Mapping[str, Any],
    *,
    max_spread_bps: float,
    max_quote_age_ms: float,
    reject_missing: bool,
) -> list[str]:
    spread, quote_age = _cost_fields(row)
    reasons: list[str] = []
    if spread is None:
        if reject_missing:
            reasons.append("missing_spread")
    elif spread > max_spread_bps:
        reasons.append("wide_spread")
    if quote_age is None:
        if reject_missing:
            reasons.append("missing_quote_age")
    elif quote_age > max_quote_age_ms:
        reasons.append("stale_quote")
    return reasons


def _microstructure_shadow_gate_summary(
    rows: list[dict[str, Any]],
    *,
    max_spread_bps: float,
    max_quote_age_ms: float,
    reject_missing: bool,
) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    decision_reject_counts: Counter[str] = Counter()
    symbol_reject_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    would_reject = 0
    for row in rows:
        decision = _decision_type(row)
        decision_counts[decision] += 1
        reasons = _microstructure_reasons(
            row,
            max_spread_bps=max_spread_bps,
            max_quote_age_ms=max_quote_age_ms,
            reject_missing=reject_missing,
        )
        if not reasons:
            continue
        would_reject += 1
        decision_reject_counts[decision] += 1
        symbol = str(row.get("symbol") or "").strip().upper() or "UNKNOWN"
        symbol_reject_counts[symbol] += 1
        reason_counts.update(reasons)
        if len(examples) < 20:
            spread, quote_age = _cost_fields(row)
            examples.append(
                {
                    "ts": row.get("ts"),
                    "symbol": symbol,
                    "decision_type": decision,
                    "spread_bps": spread,
                    "quote_age_ms": quote_age,
                    "reasons": reasons,
                }
            )
    total = len(rows)
    champion_only_rows = decision_counts.get("champion_only", 0)
    champion_only_rejects = decision_reject_counts.get("champion_only", 0)
    return {
        "rows": int(total),
        "mode": "shadow_only",
        "max_spread_bps": float(max_spread_bps),
        "max_quote_age_ms": float(max_quote_age_ms),
        "reject_missing": bool(reject_missing),
        "would_reject_count": int(would_reject),
        "would_reject_rate": _rate(would_reject, total),
        "reason_counts": dict(sorted(reason_counts.items())),
        "decision_counts": dict(sorted(decision_counts.items())),
        "decision_reject_counts": dict(sorted(decision_reject_counts.items())),
        "champion_only_would_reject_count": int(champion_only_rejects),
        "champion_only_would_reject_rate": _rate(champion_only_rejects, champion_only_rows),
        "top_symbols_by_rejects": [
            {"symbol": symbol, "rejects": int(count)}
            for symbol, count in symbol_reject_counts.most_common(20)
        ],
        "examples": examples,
    }


def _microstructure_alert_summary(
    rows: list[dict[str, Any]],
    *,
    max_spread_bps: float,
    max_quote_age_ms: float,
    max_missing_rate: float,
    max_stale_rate: float,
    max_wide_spread_rate: float,
) -> dict[str, Any]:
    total = len(rows)
    missing_quote = 0
    missing_spread = 0
    stale = 0
    wide = 0
    for row in rows:
        spread, quote_age = _cost_fields(row)
        if spread is None:
            missing_spread += 1
        elif spread > max_spread_bps:
            wide += 1
        if quote_age is None:
            missing_quote += 1
        elif quote_age > max_quote_age_ms:
            stale += 1
    missing_any = sum(
        1
        for row in rows
        if (lambda values: values[0] is None or values[1] is None)(_cost_fields(row))
    )
    missing_rate = _rate(missing_any, total) or 0.0
    stale_rate = _rate(stale, total) or 0.0
    wide_rate = _rate(wide, total) or 0.0
    breaches = {
        "missing_quote_telemetry": bool(missing_rate > max_missing_rate),
        "stale_quotes": bool(stale_rate > max_stale_rate),
        "wide_spreads": bool(wide_rate > max_wide_spread_rate),
    }
    return {
        "rows": int(total),
        "breached": any(breaches.values()),
        "breaches": breaches,
        "thresholds": {
            "max_spread_bps": float(max_spread_bps),
            "max_quote_age_ms": float(max_quote_age_ms),
            "max_missing_rate": float(max_missing_rate),
            "max_stale_rate": float(max_stale_rate),
            "max_wide_spread_rate": float(max_wide_spread_rate),
        },
        "observed": {
            "missing_any_rows": int(missing_any),
            "missing_any_rate": missing_rate,
            "missing_spread_rows": int(missing_spread),
            "missing_quote_age_rows": int(missing_quote),
            "stale_quote_rows": int(stale),
            "stale_quote_rate": stale_rate,
            "wide_spread_rows": int(wide),
            "wide_spread_rate": wide_rate,
        },
    }


def build_shadow_report(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input_jsonl)
    raw_rows = _load_shadow_rows(input_path)
    frame_filter = str(getattr(args, "frame_filter", "all") or "all")
    provider_filter = str(getattr(args, "provider_filter", "all") or "all")
    since_text = str(getattr(args, "since", "") or "").strip()
    since_ts = _parse_timestamp(since_text) if since_text else None
    if since_text and since_ts is None:
        raise ValueError(f"Invalid --since timestamp: {since_text}")
    rows, filter_counts = _filter_shadow_rows(
        raw_rows,
        frame_filter=frame_filter,
        provider_filter=provider_filter,
        since=since_ts,
    )
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    if args.data_dir:
        bars_by_symbol = _load_bars_by_symbol(Path(args.data_dir), str(args.timestamp_col))
    symbols = Counter(str(row.get("symbol") or "").strip().upper() for row in rows)
    frame_counts = Counter(
        "daily" if _is_daily_frame_row(row) else "minute"
        for row in raw_rows
    )
    horizons = _parse_horizon_bars(args)
    markout_summaries = {
        str(horizon): _markout_summary(
            rows,
            bars_by_symbol,
            horizon_bars=horizon,
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slippage_bps),
        )
        for horizon in horizons
    } if bars_by_symbol else None
    primary_horizon = int(getattr(args, "horizon_bars", horizons[0]))
    report = {
        "schema_version": "1.0.0",
        "artifact_type": "ml_shadow_report",
        "generated_at": datetime.now(UTC).isoformat(),
        "input_jsonl": str(input_path),
        "frame_filter": frame_filter,
        "provider_filter": provider_filter,
        "since": since_ts.isoformat() if since_ts is not None else None,
        "raw_rows": int(len(raw_rows)),
        "filtered_rows": int(len(rows)),
        "filter_counts": filter_counts,
        "raw_frame_counts": dict(sorted(frame_counts.items())),
        "sample_gate": _sample_gate(
            rows,
            min_informational_rows=int(getattr(args, "min_informational_rows", 100)),
            min_review_rows=int(getattr(args, "min_review_rows", 500)),
        ),
        "decision_summary": _decision_summary(rows),
        "provider_summary": _provider_summary(rows),
        "cost_observation_summary": _cost_observation_summary(rows),
        "cost_breakdowns": _grouped_cost_report(rows),
        "microstructure_shadow_gate": _microstructure_shadow_gate_summary(
            rows,
            max_spread_bps=float(getattr(args, "microstructure_max_spread_bps", 35.0)),
            max_quote_age_ms=float(getattr(args, "microstructure_max_quote_age_ms", 5000.0)),
            reject_missing=bool(getattr(args, "microstructure_reject_missing", True)),
        ),
        "microstructure_alerts": _microstructure_alert_summary(
            rows,
            max_spread_bps=float(getattr(args, "microstructure_max_spread_bps", 35.0)),
            max_quote_age_ms=float(getattr(args, "microstructure_max_quote_age_ms", 5000.0)),
            max_missing_rate=float(getattr(args, "alert_max_missing_rate", 0.01)),
            max_stale_rate=float(getattr(args, "alert_max_stale_rate", 0.05)),
            max_wide_spread_rate=float(getattr(args, "alert_max_wide_spread_rate", 0.10)),
        ),
        "markout_summary": (
            markout_summaries.get(str(primary_horizon))
            if markout_summaries is not None
            else None
        ),
        "markout_summaries": markout_summaries,
        "top_symbols_by_rows": [
            {"symbol": symbol, "rows": int(count)}
            for symbol, count in symbols.most_common(20)
            if symbol
        ],
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    logger.info(
        "ML_SHADOW_REPORT_WRITTEN",
        extra={
            "path": str(output_path),
            "rows": int(len(rows)),
            "raw_rows": int(len(raw_rows)),
            "frame_filter": frame_filter,
        },
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an ML shadow evaluation report from runtime JSONL telemetry."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--timestamp-col", type=str, default="timestamp")
    parser.add_argument("--horizon-bars", type=int, default=1)
    parser.add_argument(
        "--horizon-bars-list",
        type=str,
        default="1,3,5,15",
        help="Comma-separated markout horizons to summarize.",
    )
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument(
        "--since",
        type=str,
        default="",
        help="Only include telemetry rows whose top-level ts is at or after this UTC timestamp.",
    )
    parser.add_argument(
        "--provider-filter",
        choices=("all", "healthy-primary"),
        default="all",
        help="Filter rows by provider health before summarizing.",
    )
    parser.add_argument("--min-informational-rows", type=int, default=100)
    parser.add_argument("--min-review-rows", type=int, default=500)
    parser.add_argument(
        "--microstructure-max-spread-bps",
        type=float,
        default=35.0,
        help="Shadow-only gate threshold for wide-spread rows.",
    )
    parser.add_argument(
        "--microstructure-max-quote-age-ms",
        type=float,
        default=5000.0,
        help="Shadow-only gate threshold for stale quote rows.",
    )
    parser.add_argument(
        "--microstructure-reject-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Count missing quote telemetry as a shadow-only gate rejection.",
    )
    parser.add_argument(
        "--alert-max-missing-rate",
        type=float,
        default=0.01,
        help="Alert when missing spread or quote-age telemetry exceeds this row rate.",
    )
    parser.add_argument(
        "--alert-max-stale-rate",
        type=float,
        default=0.05,
        help="Alert when stale quote rows exceed this row rate.",
    )
    parser.add_argument(
        "--alert-max-wide-spread-rate",
        type=float,
        default=0.10,
        help="Alert when wide-spread rows exceed this row rate.",
    )
    parser.add_argument(
        "--frame-filter",
        choices=("all", "minute", "daily"),
        default="all",
        help="Filter shadow rows by bar timestamp class before summarizing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    build_shadow_report(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
