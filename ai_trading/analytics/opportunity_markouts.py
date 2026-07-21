"""Leakage-safe counterfactual markouts for recorded decision opportunities."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
import math
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from ai_trading.core.evidence_lineage import (
    deterministic_opportunity_correlation_id,
    normalize_evidence_timestamp,
    normalize_opportunity_side,
)


DEFAULT_GOVERNED_SYMBOLS: tuple[str, ...] = ("AAPL", "AMZN", "MSFT")
DEFAULT_MARKOUT_HORIZONS: tuple[int, ...] = (1, 3, 5)
_OUTCOME_VERSION = "opportunity-markout-v1"
_EASTERN = ZoneInfo("America/New_York")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _first_value(*sources: tuple[Mapping[str, Any], Sequence[str]]) -> Any:
    for source, keys in sources:
        for key in keys:
            value = source.get(key)
            if value not in (None, ""):
                return value
    return None


def _first_float(*sources: tuple[Mapping[str, Any], Sequence[str]]) -> float | None:
    return _safe_float(_first_value(*sources))


def _normalized_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "close" not in frame.columns:
        return pd.DataFrame(columns=["close"])
    normalized = frame.copy()
    if not isinstance(normalized.index, pd.DatetimeIndex):
        timestamp_column = next(
            (
                name
                for name in ("timestamp", "ts", "datetime", "date")
                if name in normalized.columns
            ),
            None,
        )
        if timestamp_column is None:
            return pd.DataFrame(columns=["close"])
        normalized.index = pd.to_datetime(
            normalized[timestamp_column],
            utc=True,
            errors="coerce",
        )
    else:
        normalized.index = pd.to_datetime(normalized.index, utc=True, errors="coerce")
    normalized = normalized[~normalized.index.isna()].copy()
    normalized["close"] = pd.to_numeric(normalized["close"], errors="coerce")
    return normalized.sort_index().loc[lambda value: ~value.index.duplicated(keep="last")]


def _regular_session(timestamp: datetime) -> tuple[str, bool]:
    eastern = timestamp.astimezone(_EASTERN)
    minute = eastern.hour * 60 + eastern.minute
    in_session = (
        eastern.weekday() < 5
        and minute >= (9 * 60 + 30)
        and minute < (16 * 60)
    )
    return eastern.date().isoformat(), in_session


def _decision_context(row: Mapping[str, Any]) -> dict[str, Any] | None:
    journal = _mapping(row.get("decision_journal"))
    metrics = _mapping(row.get("metrics"))
    metadata = _mapping(journal.get("metadata"))
    signal = _mapping(journal.get("signal"))
    order = _mapping(row.get("order"))
    order_intent = _mapping(journal.get("order_intent"))
    intent_metadata = _mapping(order_intent.get("metadata"))
    tca = _mapping(row.get("tca"))
    market_bar = _mapping(metadata.get("market_bar") or metrics.get("market_bar"))
    net_target = _mapping(row.get("net_target"))

    symbol = str(
        _first_value(
            (row, ("symbol",)),
            (journal, ("symbol",)),
            (signal, ("symbol",)),
        )
        or "UNKNOWN"
    ).strip().upper() or "UNKNOWN"
    raw_side = _first_value(
        (order, ("side",)),
        (order_intent, ("side",)),
        (signal, ("side",)),
    )
    side = normalize_opportunity_side(raw_side)
    target_shares = _safe_float(net_target.get("target_shares"))
    if side == "hold" and target_shares is not None:
        side = "buy" if target_shares > 0.0 else "sell" if target_shares < 0.0 else "hold"

    explicit_eligible = _first_value(
        (metrics, ("opportunity_eligible",)),
        (metadata, ("opportunity_eligible",)),
        (row, ("opportunity_eligible",)),
    )
    eligible = (
        bool(explicit_eligible)
        if isinstance(explicit_eligible, bool)
        else side in {"buy", "sell"}
    )
    if not eligible or side == "hold":
        return None

    source_timestamp = (
        normalize_evidence_timestamp(
            _first_value(
                (journal, ("source_timestamp",)),
                (metrics, ("source_timestamp", "source_ts")),
                (market_bar, ("ts", "timestamp")),
                (row, ("bar_ts", "source_timestamp")),
                (journal, ("bar_ts",)),
            )
        )
    )
    decision_timestamp = normalize_evidence_timestamp(
        _first_value(
            (journal, ("decision_ts",)),
            (metrics, ("decision_ts",)),
            (row, ("decision_ts", "bar_ts")),
            (journal, ("bar_ts",)),
        )
    )
    correlation_id = str(
        _first_value(
            (row, ("correlation_id",)),
            (journal, ("correlation_id",)),
            (metrics, ("correlation_id",)),
            (order_intent, ("correlation_id",)),
            (order, ("correlation_id",)),
        )
        or ""
    ).strip()
    correlation_source = "recorded"
    if not correlation_id and source_timestamp is not None:
        correlation_id = deterministic_opportunity_correlation_id(
            symbol=symbol,
            source_timestamp=source_timestamp,
            side=side,
            strategy_id=signal.get("strategy_id") or order_intent.get("strategy_id"),
        )
        correlation_source = "derived_legacy"
    if not correlation_id:
        return {
            "symbol": symbol,
            "side": side,
            "correlation_id": None,
            "correlation_source": "missing",
            "source_timestamp": source_timestamp,
            "decision_timestamp": decision_timestamp,
        }

    risk_decision = _mapping(journal.get("risk_decision"))
    gates_raw = (
        journal.get("reasons")
        or risk_decision.get("gates")
        or row.get("gates")
    )
    gates = (
        [str(value) for value in gates_raw]
        if isinstance(gates_raw, Sequence)
        and not isinstance(gates_raw, (str, bytes))
        else []
    )
    event = str(journal.get("event") or metrics.get("event") or "decision_record")
    controlled_skip = "controlled_skip" in event.lower() or any(
        "controlled_skip" in gate.lower() or gate.lower().endswith("_skip")
        for gate in gates
    )
    submitted = bool(journal.get("submitted"))
    entry_price = _first_float(
        (metadata, ("reference_price", "decision_price", "arrival_price")),
        (metrics, ("reference_price", "decision_price", "arrival_price")),
        (tca, ("decision_price", "arrival_price", "mid_at_arrival")),
        (order, ("reference_price", "price", "limit_price")),
        (order_intent, ("limit_price",)),
        (market_bar, ("close",)),
    )
    spread_bps = _first_float(
        (metadata, ("spread_bps",)),
        (metrics, ("spread_bps", "decision_spread_bps")),
        (tca, ("decision_spread_bps", "spread_bps")),
    )
    explicit_cost_bps = _first_float(
        (metrics, ("round_trip_cost_bps", "expected_cost_bps")),
        (tca, ("round_trip_cost_bps", "expected_cost_bps")),
    )
    return {
        "symbol": symbol,
        "side": side,
        "correlation_id": correlation_id,
        "correlation_source": correlation_source,
        "source_timestamp": source_timestamp,
        "decision_timestamp": decision_timestamp,
        "quote_timestamp": normalize_evidence_timestamp(journal.get("quote_timestamp")),
        "entry_price": entry_price,
        "spread_bps": spread_bps,
        "explicit_cost_bps": explicit_cost_bps,
        "quote_age_ms": _first_float(
            (metadata, ("quote_age_ms",)),
            (metrics, ("quote_age_ms", "decision_quote_age_ms")),
            (intent_metadata, ("quote_age_ms",)),
        ),
        "order_type": str(
            _first_value(
                (metadata, ("order_type",)),
                (metrics, ("order_type",)),
                (order, ("order_type", "type")),
            )
            or "unknown"
        ).lower(),
        "session": str(
            _first_value(
                (metadata, ("session_regime",)),
                (metrics, ("session_regime", "session")),
                (intent_metadata, ("session_regime",)),
            )
            or "unknown"
        ).lower(),
        "market_regime": str(
            _first_value(
                (metadata, ("market_regime",)),
                (metrics, ("market_regime",)),
                (intent_metadata, ("market_regime",)),
            )
            or "unknown"
        ).lower(),
        "volatility_regime": str(
            _first_value(
                (metadata, ("volatility_regime",)),
                (metrics, ("volatility_regime",)),
            )
            or "unknown"
        ).lower(),
        "trend_regime": str(
            _first_value(
                (metadata, ("trend_regime",)),
                (metrics, ("trend_regime",)),
            )
            or "unknown"
        ).lower(),
        "execution_profile": str(
            _first_value(
                (metadata, ("execution_profile", "regime_profile")),
                (metrics, ("execution_profile", "regime_profile")),
            )
            or "unknown"
        ).lower(),
        "event": event,
        "gates": gates,
        "controlled_skip": controlled_skip,
        "submitted": submitted,
    }


def _context_completeness(context: Mapping[str, Any]) -> tuple[int, str]:
    populated = sum(
        context.get(key) not in (None, "", "unknown", [])
        for key in (
            "entry_price",
            "quote_age_ms",
            "spread_bps",
            "order_type",
            "session",
            "market_regime",
            "execution_profile",
        )
    )
    return int(populated), str(context.get("event") or "")


def _deduplicated_contexts(
    decisions: Sequence[Mapping[str, Any]],
    governed_symbols: set[str],
) -> tuple[list[dict[str, Any]], int, int]:
    selected: dict[str, dict[str, Any]] = {}
    missing_correlation = 0
    eligible_rows = 0
    for row in decisions:
        context = _decision_context(row)
        if context is None or str(context.get("symbol")) not in governed_symbols:
            continue
        eligible_rows += 1
        correlation_id = str(context.get("correlation_id") or "")
        if not correlation_id:
            missing_correlation += 1
            continue
        prior = selected.get(correlation_id)
        if prior is None or _context_completeness(context) > _context_completeness(prior):
            selected[correlation_id] = context
    contexts = sorted(
        selected.values(),
        key=lambda row: (
            str(row.get("source_timestamp") or ""),
            str(row.get("symbol") or ""),
            str(row.get("correlation_id") or ""),
        ),
    )
    duplicate_rows = max(0, eligible_rows - missing_correlation - len(contexts))
    return contexts, duplicate_rows, missing_correlation


def _outcome_id(correlation_id: str, horizon: int) -> str:
    return f"{correlation_id}:shadow_counterfactual:h{int(horizon)}:v1"


def _base_outcome(
    context: Mapping[str, Any],
    *,
    horizon: int,
    round_trip_cost_bps: float,
) -> dict[str, Any]:
    source_timestamp = context.get("source_timestamp")
    decision_timestamp = context.get("decision_timestamp")
    quote_timestamp = context.get("quote_timestamp")
    return {
        "outcome_id": _outcome_id(str(context["correlation_id"]), horizon),
        "outcome_version": _OUTCOME_VERSION,
        "correlation_id": context["correlation_id"],
        "correlation_source": context.get("correlation_source"),
        "symbol": context["symbol"],
        "side": context["side"],
        "horizon_bars": int(horizon),
        "bar_timeframe": "1Min",
        "decision_timestamp": (
            decision_timestamp.isoformat()
            if isinstance(decision_timestamp, datetime)
            else None
        ),
        "source_timestamp": (
            source_timestamp.isoformat()
            if isinstance(source_timestamp, datetime)
            else None
        ),
        "quote_timestamp": (
            quote_timestamp.isoformat()
            if isinstance(quote_timestamp, datetime)
            else None
        ),
        "event": context.get("event"),
        "gates": list(context.get("gates") or []),
        "controlled_skip": bool(context.get("controlled_skip")),
        "submitted": bool(context.get("submitted")),
        "executed": False,
        "entry_price": context.get("entry_price"),
        "exit_price": None,
        "label_end_timestamp": None,
        "gross_markout_bps": None,
        "round_trip_cost_bps": float(round_trip_cost_bps),
        "net_markout_bps": None,
        "counterfactual_net_edge_bps": None,
        "quote_age_ms": context.get("quote_age_ms"),
        "spread_bps": context.get("spread_bps"),
        "order_type": context.get("order_type"),
        "session": context.get("session"),
        "market_regime": context.get("market_regime"),
        "volatility_regime": context.get("volatility_regime"),
        "trend_regime": context.get("trend_regime"),
        "execution_profile": context.get("execution_profile"),
        "label_status": "unavailable",
        "label_reason": "unresolved",
        "evidence_type": "shadow_counterfactual",
        "evidence_partition": "shadow",
        "fill_based_evidence": False,
        "promotion_eligible": False,
        "runtime_authority": False,
        "promotion_authority": False,
        "live_money_authority": False,
    }


def _resolve_context_horizon(
    context: Mapping[str, Any],
    frame: pd.DataFrame,
    *,
    horizon: int,
    fee_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    spread_bps = max(0.0, _safe_float(context.get("spread_bps")) or 0.0)
    modeled_cost = spread_bps + (2.0 * max(0.0, float(fee_bps))) + (
        2.0 * max(0.0, float(slippage_bps))
    )
    explicit_cost = max(0.0, _safe_float(context.get("explicit_cost_bps")) or 0.0)
    round_trip_cost = max(modeled_cost, explicit_cost)
    outcome = _base_outcome(
        context,
        horizon=horizon,
        round_trip_cost_bps=round_trip_cost,
    )
    source_timestamp = context.get("source_timestamp")
    decision_timestamp = context.get("decision_timestamp")
    if not isinstance(source_timestamp, datetime):
        outcome["label_reason"] = "source_timestamp_missing"
        return outcome
    if (
        isinstance(decision_timestamp, datetime)
        and source_timestamp > decision_timestamp
    ):
        outcome["label_reason"] = "source_timestamp_after_decision"
        return outcome
    if frame.empty:
        outcome["label_reason"] = "symbol_bars_unavailable"
        return outcome
    source_key = pd.Timestamp(source_timestamp.astimezone(UTC))
    if source_key not in frame.index:
        outcome["label_reason"] = "source_bar_missing"
        return outcome
    session_date, in_session = _regular_session(source_timestamp)
    if not in_session:
        outcome["label_reason"] = "source_outside_regular_session"
        return outcome

    expected_timestamps = [
        source_timestamp + timedelta(minutes=offset)
        for offset in range(1, int(horizon) + 1)
    ]
    if any(
        not _regular_session(timestamp)[1]
        or _regular_session(timestamp)[0] != session_date
        for timestamp in expected_timestamps
    ):
        outcome["label_status"] = "censored"
        outcome["label_reason"] = "session_boundary"
        return outcome
    future_key = pd.Timestamp(expected_timestamps[-1].astimezone(UTC))
    if future_key > frame.index.max():
        outcome["label_status"] = "censored"
        outcome["label_reason"] = "insufficient_future_bars"
        return outcome
    if any(
        pd.Timestamp(timestamp.astimezone(UTC)) not in frame.index
        for timestamp in expected_timestamps
    ):
        outcome["label_status"] = "censored"
        outcome["label_reason"] = "non_contiguous_future_bars"
        return outcome

    entry_price = _safe_float(context.get("entry_price"))
    entry_source = "decision_snapshot"
    if entry_price is None or entry_price <= 0.0:
        entry_price = _safe_float(frame.at[source_key, "close"])
        entry_source = "source_bar_close"
    exit_price = _safe_float(frame.at[future_key, "close"])
    if entry_price is None or entry_price <= 0.0:
        outcome["label_reason"] = "entry_price_unavailable"
        return outcome
    if exit_price is None or exit_price <= 0.0:
        outcome["label_status"] = "censored"
        outcome["label_reason"] = "future_price_unavailable"
        return outcome
    direction = 1.0 if context["side"] == "buy" else -1.0
    gross = direction * ((float(exit_price) / float(entry_price)) - 1.0) * 10_000.0
    net = float(gross - round_trip_cost)
    outcome.update(
        {
            "entry_price": float(entry_price),
            "entry_price_source": entry_source,
            "exit_price": float(exit_price),
            "label_end_timestamp": expected_timestamps[-1].isoformat(),
            "gross_markout_bps": float(gross),
            "net_markout_bps": net,
            "counterfactual_net_edge_bps": net,
            "label_status": "resolved",
            "label_reason": "ok",
        }
    )
    return outcome


def resolve_opportunity_markouts(
    decisions: Sequence[Mapping[str, Any]],
    bars_by_symbol: Mapping[str, pd.DataFrame],
    *,
    governed_symbols: Sequence[str] = DEFAULT_GOVERNED_SYMBOLS,
    horizons: Sequence[int] = DEFAULT_MARKOUT_HORIZONS,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> dict[str, Any]:
    """Return one idempotent shadow outcome for each opportunity and horizon."""

    symbols = {
        str(symbol or "").strip().upper()
        for symbol in governed_symbols
        if str(symbol or "").strip()
    }
    parsed_horizons = sorted({int(value) for value in horizons if int(value) > 0})
    if parsed_horizons != list(DEFAULT_MARKOUT_HORIZONS):
        raise ValueError("opportunity markout horizons must be exactly 1,3,5")
    contexts, duplicate_rows, missing_correlation = _deduplicated_contexts(
        decisions,
        symbols,
    )
    normalized_frames = {
        str(symbol).strip().upper(): _normalized_frame(frame)
        for symbol, frame in bars_by_symbol.items()
        if str(symbol).strip().upper() in symbols
    }
    outcomes: list[dict[str, Any]] = []
    outcome_ids: set[str] = set()
    for context in contexts:
        frame = normalized_frames.get(str(context["symbol"]), pd.DataFrame())
        for horizon in parsed_horizons:
            outcome = _resolve_context_horizon(
                context,
                frame,
                horizon=horizon,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
            )
            outcome_id = str(outcome["outcome_id"])
            if outcome_id in outcome_ids:
                continue
            outcome_ids.add(outcome_id)
            outcomes.append(outcome)
    status_counts = Counter(str(row["label_status"]) for row in outcomes)
    reason_counts = Counter(str(row["label_reason"]) for row in outcomes)
    resolved_values = [
        float(row["net_markout_bps"])
        for row in outcomes
        if row["label_status"] == "resolved"
        and row["net_markout_bps"] is not None
    ]
    return {
        "schema_version": "1.0.0",
        "artifact_type": "opportunity_markout_report",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "governed_symbols": sorted(symbols),
        "horizons": parsed_horizons,
        "eligible_opportunities": len(contexts),
        "expected_outcomes": len(contexts) * len(parsed_horizons),
        "outcomes_emitted": len(outcomes),
        "duplicate_decision_rows_discarded": int(duplicate_rows),
        "eligible_rows_missing_correlation": int(missing_correlation),
        "outcome_ids_unique": len(outcome_ids) == len(outcomes),
        "label_status_counts": dict(sorted(status_counts.items())),
        "label_reason_counts": dict(sorted(reason_counts.items())),
        "mean_resolved_net_markout_bps": (
            float(sum(resolved_values) / len(resolved_values))
            if resolved_values
            else None
        ),
        "evidence_type": "shadow_counterfactual",
        "evidence_partition": "shadow",
        "fill_based_evidence": False,
        "promotion_eligible": False,
        "runtime_authority": False,
        "promotion_authority": False,
        "live_money_authority": False,
        "outcomes": outcomes,
    }


__all__ = [
    "DEFAULT_GOVERNED_SYMBOLS",
    "DEFAULT_MARKOUT_HORIZONS",
    "resolve_opportunity_markouts",
]
