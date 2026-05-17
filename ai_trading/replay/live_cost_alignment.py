"""Pure helpers for aligning replay costs with observed live costs."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any, Iterable, Mapping


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _finite_int(value: Any) -> int:
    parsed = _finite_float(value)
    if parsed is None:
        return 0
    return max(0, int(parsed))


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


def _normalize_side(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    if token in {"buy", "long", "cover", "buy_to_cover", "buytocover"}:
        return "buy"
    if token in {"sell", "sell_long", "selllong"}:
        return "sell"
    if token in {"short", "sell_short", "sellshort"}:
        return "sell_short"
    return token or "unknown"


def _normalize_order_type(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    if token in {"marketablelimit", "marketable_limit"}:
        return "marketable_limit"
    if token in {"market", "limit", "stop", "stop_limit", "trailing_stop"}:
        return token
    return token or "unknown"


def _normalize_bucket(value: Any) -> str:
    return str(value or "").strip().lower() or "unknown"


def _live_cost_value(row: Mapping[str, Any], metric: str) -> tuple[float | None, str | None]:
    metric_keys = (
        metric,
        "p90_total_cost_bps",
        "mean_total_cost_bps",
        "modeled_total_cost_bps",
        "total_cost_bps",
    )
    seen: set[str] = set()
    for key in metric_keys:
        if key in seen:
            continue
        seen.add(key)
        parsed = _finite_float(row.get(key))
        if parsed is not None and parsed >= 0.0:
            return float(parsed), key
    return None, None


def _freshness(
    *,
    model: Mapping[str, Any],
    row: Mapping[str, Any] | None,
    now: datetime,
    max_age_seconds: float,
) -> dict[str, Any]:
    max_age = max(0.0, float(max_age_seconds))
    generated_at = _parse_ts(model.get("generated_at"))
    row_observed_at = _parse_ts((row or {}).get("last_observed_at"))

    generated_age_seconds: float | None = None
    generated_future_dated = False
    if generated_at is not None:
        generated_delta_seconds = (now - generated_at).total_seconds()
        generated_future_dated = generated_delta_seconds < 0.0
        generated_age_seconds = max(0.0, generated_delta_seconds)

    row_age_seconds: float | None = None
    row_future_dated = False
    if row_observed_at is not None:
        row_delta_seconds = (now - row_observed_at).total_seconds()
        row_future_dated = row_delta_seconds < 0.0
        row_age_seconds = max(0.0, row_delta_seconds)

    fresh = bool(
        generated_age_seconds is not None
        and not generated_future_dated
        and generated_age_seconds <= max_age
        and not row_future_dated
        and (row_age_seconds is None or row_age_seconds <= max_age)
    )
    reason = "fresh"
    if generated_age_seconds is None:
        reason = "missing_generated_at"
    elif generated_future_dated:
        reason = "future_model"
    elif generated_age_seconds > max_age:
        reason = "stale_model"
    elif row_future_dated:
        reason = "future_bucket"
    elif row_age_seconds is not None and row_age_seconds > max_age:
        reason = "stale_bucket"

    return {
        "fresh": bool(fresh),
        "stale": not bool(fresh),
        "reason": reason,
        "generated_at": generated_at.isoformat() if generated_at is not None else None,
        "last_observed_at": (
            row_observed_at.isoformat() if row_observed_at is not None else None
        ),
        "age_seconds": generated_age_seconds,
        "row_age_seconds": row_age_seconds,
        "max_age_seconds": float(max_age),
    }


def _rows(payload: Mapping[str, Any], key: str) -> Iterable[Mapping[str, Any]]:
    raw_rows = payload.get(key)
    if not isinstance(raw_rows, list):
        return ()
    return (row for row in raw_rows if isinstance(row, Mapping))


def _matching_live_cost_rows(
    live_cost_model: Mapping[str, Any],
    *,
    symbol: str,
    side: str,
    session_bucket: str,
    order_type: str,
    volatility_bucket: str,
) -> tuple[list[Mapping[str, Any]], str]:
    detailed_rows = list(
        _rows(live_cost_model, "by_symbol_side_session_order_type_volatility")
    )
    if detailed_rows:
        matches = [
            row
            for row in detailed_rows
            if str(row.get("symbol") or "").strip().upper() == symbol
            and _normalize_side(row.get("side")) == side
            and _normalize_bucket(row.get("session_regime")) == session_bucket
            and _normalize_order_type(row.get("order_type")) == order_type
            and _normalize_bucket(row.get("volatility_bucket")) == volatility_bucket
        ]
        return matches, "by_symbol_side_session_order_type_volatility"

    base_matches = [
        row
        for row in _rows(live_cost_model, "by_symbol_side_session")
        if str(row.get("symbol") or "").strip().upper() == symbol
        and _normalize_side(row.get("side")) == side
        and _normalize_bucket(row.get("session_regime")) == session_bucket
    ]
    return base_matches, "by_symbol_side_session"


def resolve_live_cost_alignment(
    live_cost_model: Mapping[str, Any] | None,
    *,
    symbol: str,
    side: str,
    session_bucket: str,
    order_type: str,
    volatility_bucket: str,
    fallback_cost_bps: float,
    now: datetime | None = None,
    max_age_seconds: float = 86_400.0,
    min_samples: int = 5,
    cost_metric: str = "p90_total_cost_bps",
) -> dict[str, Any]:
    """Return conservative replay cost resolution and diagnostics.

    Live costs are only allowed to increase the replay cost. Stale buckets,
    insufficient samples, missing observations, and cheaper live costs resolve
    to the supplied fallback so replay diagnostics do not gain false optimism.
    """

    now_utc = (now or datetime.now(UTC)).astimezone(UTC)
    fallback = max(0.0, _finite_float(fallback_cost_bps) or 0.0)
    symbol_token = str(symbol or "").strip().upper()
    side_token = _normalize_side(side)
    session_token = _normalize_bucket(session_bucket)
    order_type_token = _normalize_order_type(order_type)
    volatility_token = _normalize_bucket(volatility_bucket)
    min_sample_count = max(1, int(min_samples))
    key = {
        "symbol": symbol_token,
        "side": side_token,
        "session_bucket": session_token,
        "order_type": order_type_token,
        "volatility_bucket": volatility_token,
    }

    if not isinstance(live_cost_model, Mapping):
        return {
            "key": key,
            "resolved_cost_bps": float(fallback),
            "fallback_cost_bps": float(fallback),
            "observed_live_cost_bps": None,
            "observed_live_cost_metric": None,
            "sample_count": 0,
            "min_samples": int(min_sample_count),
            "source": "fallback",
            "bucket_source": None,
            "alignment": "missing_live_cost_model",
            "optimistic": False,
            "pessimistic": False,
            "freshness": _freshness(
                model={},
                row=None,
                now=now_utc,
                max_age_seconds=max_age_seconds,
            ),
        }

    matches, bucket_source = _matching_live_cost_rows(
        live_cost_model,
        symbol=symbol_token,
        side=side_token,
        session_bucket=session_token,
        order_type=order_type_token,
        volatility_bucket=volatility_token,
    )
    if not matches:
        return {
            "key": key,
            "resolved_cost_bps": float(fallback),
            "fallback_cost_bps": float(fallback),
            "observed_live_cost_bps": None,
            "observed_live_cost_metric": None,
            "sample_count": 0,
            "min_samples": int(min_sample_count),
            "source": "fallback",
            "bucket_source": bucket_source,
            "alignment": "missing_live_cost_bucket",
            "optimistic": False,
            "pessimistic": False,
            "freshness": _freshness(
                model=live_cost_model,
                row=None,
                now=now_utc,
                max_age_seconds=max_age_seconds,
            ),
        }

    ranked: list[tuple[float, Mapping[str, Any], str, int]] = []
    for row in matches:
        observed, metric = _live_cost_value(row, cost_metric)
        if observed is None or metric is None:
            continue
        ranked.append((float(observed), row, metric, _finite_int(row.get("sample_count"))))
    if not ranked:
        row = matches[0]
        return {
            "key": key,
            "resolved_cost_bps": float(fallback),
            "fallback_cost_bps": float(fallback),
            "observed_live_cost_bps": None,
            "observed_live_cost_metric": None,
            "sample_count": _finite_int(row.get("sample_count")),
            "min_samples": int(min_sample_count),
            "source": "fallback",
            "bucket_source": bucket_source,
            "alignment": "missing_live_cost_metric",
            "optimistic": False,
            "pessimistic": False,
            "freshness": _freshness(
                model=live_cost_model,
                row=row,
                now=now_utc,
                max_age_seconds=max_age_seconds,
            ),
        }

    observed, row, metric, sample_count = max(ranked, key=lambda item: item[0])
    freshness = _freshness(
        model=live_cost_model,
        row=row,
        now=now_utc,
        max_age_seconds=max_age_seconds,
    )
    sufficient = bool(
        sample_count >= min_sample_count
        and bool(row.get("sufficient_samples", True))
    )
    if not bool(freshness["fresh"]):
        alignment = "stale"
        resolved = fallback
        source = "fallback"
    elif not sufficient:
        alignment = "insufficient_samples"
        resolved = fallback
        source = "fallback"
    elif observed < fallback:
        alignment = "optimism"
        resolved = fallback
        source = "fallback"
    elif observed > fallback:
        alignment = "pessimism"
        resolved = observed
        source = "live"
    else:
        alignment = "neutral"
        resolved = fallback
        source = "fallback"

    return {
        "key": key,
        "resolved_cost_bps": float(resolved),
        "fallback_cost_bps": float(fallback),
        "observed_live_cost_bps": float(observed),
        "observed_live_cost_metric": metric,
        "sample_count": int(sample_count),
        "min_samples": int(min_sample_count),
        "sufficient_samples": bool(sufficient),
        "source": source,
        "bucket_source": bucket_source,
        "alignment": alignment,
        "optimistic": alignment == "optimism",
        "pessimistic": alignment == "pessimism",
        "freshness": freshness,
    }


def resolve_live_cost_alignments(
    live_cost_model: Mapping[str, Any] | None,
    replay_costs: Iterable[Mapping[str, Any]],
    *,
    now: datetime | None = None,
    max_age_seconds: float = 86_400.0,
    min_samples: int = 5,
    cost_metric: str = "p90_total_cost_bps",
) -> dict[str, Any]:
    """Resolve live-cost alignment for a batch of replay cost rows."""

    items: list[dict[str, Any]] = []
    for row in replay_costs:
        fallback = _finite_float(
            row.get("fallback_cost_bps")
            if "fallback_cost_bps" in row
            else row.get("replay_cost_bps")
        )
        items.append(
            resolve_live_cost_alignment(
                live_cost_model,
                symbol=str(row.get("symbol") or ""),
                side=str(row.get("side") or ""),
                session_bucket=str(
                    row.get("session_bucket")
                    or row.get("session_regime")
                    or row.get("session")
                    or ""
                ),
                order_type=str(row.get("order_type") or row.get("type") or ""),
                volatility_bucket=str(
                    row.get("volatility_bucket")
                    or row.get("vol_bucket")
                    or row.get("liquidity_bucket")
                    or ""
                ),
                fallback_cost_bps=float(fallback or 0.0),
                now=now,
                max_age_seconds=max_age_seconds,
                min_samples=min_samples,
                cost_metric=cost_metric,
            )
        )
    alignment_counts: dict[str, int] = {}
    for item in items:
        label = str(item.get("alignment") or "unknown")
        alignment_counts[label] = alignment_counts.get(label, 0) + 1
    return {
        "items": items,
        "summary": {
            "count": int(len(items)),
            "alignment_counts": dict(sorted(alignment_counts.items())),
            "optimism_count": int(
                sum(1 for item in items if bool(item.get("optimistic")))
            ),
            "pessimism_count": int(
                sum(1 for item in items if bool(item.get("pessimistic")))
            ),
            "stale_count": int(
                sum(
                    1
                    for item in items
                    if bool((item.get("freshness") or {}).get("stale"))
                )
            ),
        },
    }


__all__ = [
    "resolve_live_cost_alignment",
    "resolve_live_cost_alignments",
]
