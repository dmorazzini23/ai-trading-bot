"""Feed drift helpers for execution-vs-reference diagnostics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import math
from statistics import median
from typing import Any, Iterable, Mapping

from ai_trading.alpaca_api import alpaca_get
from ai_trading.data.feed_roles import get_reference_feed
from ai_trading.logging import get_logger
from ai_trading.utils.env import get_alpaca_data_v2_base

logger = get_logger(__name__)


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    return None


def normalize_signal_side(value: Any) -> str | None:
    token = str(value or "").strip().lower()
    if not token:
        return None
    if token in {"buy", "long", "bull", "cover", "entry_long"}:
        return "buy"
    if token in {"sell", "short", "bear", "sell_short", "entry_short"}:
        return "sell"
    if token in {"flat", "none", "hold"}:
        return "flat"
    return None


def _classify_decision(outcome: str) -> str:
    token = str(outcome or "").strip().lower()
    if not token:
        return "trigger"
    if token.startswith("skip"):
        return "gate"
    gate_fragments = (
        "gate",
        "blocked",
        "required",
        "rejected",
        "stale_quote",
        "degraded_feed",
    )
    if any(fragment in token for fragment in gate_fragments):
        return "gate"
    return "trigger"


def derive_signal_agreement(
    *,
    outcome: str,
    side: Any | None = None,
    signal_side: Any | None = None,
    signal_strength: Any | None = None,
    signal_confidence: Any | None = None,
    price_drift_bps: Any | None = None,
    drift_disagreement_bps: float = 25.0,
) -> dict[str, Any]:
    """Derive explicit signal agreement/disagreement flags for a decision row."""

    resolved_signal_side = normalize_signal_side(signal_side) or normalize_signal_side(side)
    decision_class = _classify_decision(outcome)
    drift_value = _safe_float(price_drift_bps)
    try:
        drift_threshold_bps = max(float(drift_disagreement_bps), 0.0)
    except (TypeError, ValueError):
        drift_threshold_bps = 25.0

    signal_agreement: bool | None = None
    signal_disagreement: bool | None = None
    signal_disagreement_reason: str | None = None
    if drift_value is not None:
        disagreement = abs(drift_value) >= drift_threshold_bps if drift_threshold_bps > 0 else False
        signal_disagreement = bool(disagreement)
        signal_agreement = not signal_disagreement
        if signal_disagreement:
            signal_disagreement_reason = "price_drift_threshold"

    return {
        "decision_class": decision_class,
        "signal_side": resolved_signal_side,
        "signal_strength": _safe_float(signal_strength),
        "signal_confidence": _safe_float(signal_confidence),
        "signal_agreement": signal_agreement,
        "signal_disagreement": signal_disagreement,
        "signal_disagreement_reason": signal_disagreement_reason,
        "drift_threshold_bps": float(drift_threshold_bps),
    }


def _alpaca_data_get(path: str, *, params: Mapping[str, Any] | None = None) -> Mapping[str, Any] | None:
    base = get_alpaca_data_v2_base().rstrip("/")
    url = f"{base}/{path.lstrip('/')}"
    try:
        payload = alpaca_get(url, params=dict(params or {}))
    except Exception:
        logger.debug("REFERENCE_DATA_FETCH_FAILED", extra={"url": url}, exc_info=True)
        return None
    if isinstance(payload, Mapping):
        return payload
    return None


def _extract_trade_price(payload: Mapping[str, Any] | None) -> float | None:
    if not isinstance(payload, Mapping):
        return None
    candidates: list[Any] = []
    trade_payload = payload.get("trade")
    if isinstance(trade_payload, Mapping):
        candidates.extend([trade_payload.get("p"), trade_payload.get("price"), trade_payload.get("last")])
    candidates.extend([payload.get("p"), payload.get("price"), payload.get("last")])
    for candidate in candidates:
        parsed = _safe_float(candidate)
        if parsed is not None and parsed > 0:
            return parsed
    return None


def _extract_bid_ask(payload: Mapping[str, Any] | None) -> tuple[float | None, float | None]:
    if not isinstance(payload, Mapping):
        return None, None
    quote_payload = payload.get("quote")
    candidates: list[Mapping[str, Any]] = []
    if isinstance(quote_payload, Mapping):
        candidates.append(quote_payload)
    candidates.append(payload)
    for candidate in candidates:
        bid = _safe_float(candidate.get("bp") if isinstance(candidate, Mapping) else None)
        ask = _safe_float(candidate.get("ap") if isinstance(candidate, Mapping) else None)
        if bid is None and isinstance(candidate, Mapping):
            bid = _safe_float(candidate.get("bid_price"))
        if ask is None and isinstance(candidate, Mapping):
            ask = _safe_float(candidate.get("ask_price"))
        if bid is not None or ask is not None:
            return bid, ask
    return None, None


def fetch_reference_snapshot(
    symbol: str,
    *,
    feed: str | None = None,
) -> dict[str, Any]:
    """Fetch a reference snapshot for ``symbol`` using delayed SIP by default."""

    resolved_feed = get_reference_feed(feed)
    trade_payload = _alpaca_data_get(
        f"stocks/{symbol}/trades/latest",
        params={"feed": resolved_feed},
    )
    quote_payload = _alpaca_data_get(
        f"stocks/{symbol}/quotes/latest",
        params={"feed": resolved_feed},
    )
    trade_price = _extract_trade_price(trade_payload)
    bid, ask = _extract_bid_ask(quote_payload)
    mid: float | None = None
    if bid is not None and ask is not None and ask >= bid:
        mid = (bid + ask) / 2.0
    ref_price = trade_price if trade_price is not None else mid
    return {
        "symbol": symbol,
        "feed": resolved_feed,
        "price": ref_price,
        "trade_price": trade_price,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "fetched_at": datetime.now(UTC).isoformat(),
    }


def fetch_reference_minute_bar_snapshot(
    symbol: str,
    *,
    decision_ts: datetime,
    feed: str | None = None,
) -> dict[str, Any]:
    """Return nearest delayed-reference minute bar around ``decision_ts``."""

    resolved_feed = get_reference_feed(feed)
    decision_utc = decision_ts.astimezone(UTC)
    start = (decision_utc - timedelta(minutes=1)).isoformat()
    end = (decision_utc + timedelta(minutes=1)).isoformat()
    payload = _alpaca_data_get(
        f"stocks/{symbol}/bars",
        params={
            "timeframe": "1Min",
            "start": start,
            "end": end,
            "feed": resolved_feed,
            "limit": 5,
        },
    )
    bars_value = payload.get("bars") if isinstance(payload, Mapping) else None
    if isinstance(bars_value, Mapping):
        bars_value = bars_value.get(symbol)
    bars: list[Mapping[str, Any]] = []
    if isinstance(bars_value, list):
        bars = [item for item in bars_value if isinstance(item, Mapping)]
    nearest: Mapping[str, Any] | None = None
    nearest_distance = float("inf")
    for bar in bars:
        ts_raw = bar.get("t") or bar.get("timestamp")
        if not isinstance(ts_raw, str):
            continue
        try:
            bar_ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).astimezone(UTC)
        except ValueError:
            continue
        distance = abs((bar_ts - decision_utc).total_seconds())
        if distance < nearest_distance:
            nearest = bar
            nearest_distance = distance
    close = _safe_float((nearest or {}).get("c")) or _safe_float((nearest or {}).get("close"))
    volume = _safe_float((nearest or {}).get("v")) or _safe_float((nearest or {}).get("volume"))
    ts_value = (nearest or {}).get("t") or (nearest or {}).get("timestamp")
    return {
        "symbol": symbol,
        "feed": resolved_feed,
        "price": close,
        "volume": volume,
        "bar_ts": str(ts_value) if ts_value is not None else None,
        "fetched_at": datetime.now(UTC).isoformat(),
    }


def compute_drift_metrics(
    *,
    execution_price: float | None,
    reference_price: float | None,
    execution_bid: float | None = None,
    execution_ask: float | None = None,
    reference_bid: float | None = None,
    reference_ask: float | None = None,
    execution_volume: float | None = None,
    reference_volume: float | None = None,
) -> dict[str, float | None]:
    """Compute comparable execution-vs-reference drift diagnostics."""

    metrics: dict[str, float | None] = {
        "price_drift_bps": None,
        "execution_spread_bps": None,
        "reference_spread_bps": None,
        "spread_ratio": None,
        "volume_coverage_ratio": None,
    }
    if execution_price and reference_price and execution_price > 0 and reference_price > 0:
        metrics["price_drift_bps"] = ((execution_price - reference_price) / reference_price) * 10000.0
    exec_spread_bps: float | None = None
    ref_spread_bps: float | None = None
    if execution_bid and execution_ask and execution_ask >= execution_bid and execution_price and execution_price > 0:
        exec_spread_bps = ((execution_ask - execution_bid) / execution_price) * 10000.0
        metrics["execution_spread_bps"] = exec_spread_bps
    if reference_bid and reference_ask and reference_ask >= reference_bid and reference_price and reference_price > 0:
        ref_spread_bps = ((reference_ask - reference_bid) / reference_price) * 10000.0
        metrics["reference_spread_bps"] = ref_spread_bps
    if exec_spread_bps and ref_spread_bps and ref_spread_bps > 0:
        metrics["spread_ratio"] = exec_spread_bps / ref_spread_bps
    if execution_volume and reference_volume and reference_volume > 0:
        metrics["volume_coverage_ratio"] = execution_volume / reference_volume
    return metrics


def _safe_median(values: Iterable[float]) -> float | None:
    cleaned = [value for value in values if math.isfinite(value)]
    if not cleaned:
        return None
    return float(median(cleaned))


def build_symbol_reliability_scores(
    rows: Iterable[Mapping[str, Any]],
    *,
    drift_disagreement_bps: float = 25.0,
    min_samples: int = 3,
) -> dict[str, dict[str, float | int | None]]:
    """Aggregate reconciliation rows into symbol-level execution-feed reliability."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        grouped.setdefault(symbol, []).append(row)

    scores: dict[str, dict[str, float | int | None]] = {}
    drift_threshold = max(float(drift_disagreement_bps), 0.0)
    for symbol, symbol_rows in grouped.items():
        if len(symbol_rows) < max(int(min_samples), 1):
            continue
        abs_drift_values: list[float] = []
        spread_ratio_values: list[float] = []
        volume_ratio_values: list[float] = []
        trigger_disagreements = 0
        trigger_total = 0
        gate_disagreements = 0
        gate_total = 0
        for row in symbol_rows:
            metrics = row.get("metrics")
            if not isinstance(metrics, Mapping):
                continue
            try:
                drift_value = float(metrics.get("price_drift_bps"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                drift_value = math.nan
            try:
                spread_ratio = float(metrics.get("spread_ratio"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                spread_ratio = math.nan
            try:
                volume_ratio = float(metrics.get("volume_coverage_ratio"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                volume_ratio = math.nan
            if math.isfinite(drift_value):
                abs_drift_values.append(abs(drift_value))
            outcome = str(row.get("outcome") or "").strip().lower()
            signal_metrics = row.get("signal_metrics")
            signal_metrics_map = signal_metrics if isinstance(signal_metrics, Mapping) else {}
            decision_class = str(
                signal_metrics_map.get("decision_class")
                or row.get("decision_class")
                or _classify_decision(outcome)
            ).strip().lower()
            if decision_class not in {"gate", "trigger"}:
                decision_class = _classify_decision(outcome)
            disagreement = _safe_bool(signal_metrics_map.get("signal_disagreement"))
            if disagreement is None:
                disagreement = _safe_bool(row.get("signal_disagreement"))
            if disagreement is None and math.isfinite(drift_value):
                disagreement = abs(drift_value) >= drift_threshold if drift_threshold > 0 else False
            if disagreement is not None:
                if decision_class == "gate":
                    gate_total += 1
                    if disagreement:
                        gate_disagreements += 1
                else:
                    trigger_total += 1
                    if disagreement:
                        trigger_disagreements += 1
            if math.isfinite(spread_ratio) and spread_ratio > 0:
                spread_ratio_values.append(spread_ratio)
            if math.isfinite(volume_ratio) and volume_ratio > 0:
                volume_ratio_values.append(volume_ratio)

        sample_count = len(abs_drift_values)
        if sample_count < max(int(min_samples), 1):
            continue
        median_abs_drift_bps = _safe_median(abs_drift_values)
        median_spread_ratio = _safe_median(spread_ratio_values)
        median_volume_coverage = _safe_median(volume_ratio_values)
        trigger_disagreement_rate = (
            float(trigger_disagreements) / float(trigger_total) if trigger_total > 0 else None
        )
        gate_disagreement_rate = (
            float(gate_disagreements) / float(gate_total) if gate_total > 0 else None
        )

        drift_penalty = min((median_abs_drift_bps or 0.0) / 50.0, 1.0)
        spread_penalty = 0.0
        if median_spread_ratio is not None and median_spread_ratio > 0:
            spread_penalty = min(abs(math.log(median_spread_ratio, 2.0)) / 2.0, 1.0)
        volume_penalty = 1.0
        if median_volume_coverage is not None and median_volume_coverage > 0:
            volume_penalty = min(abs(1.0 - median_volume_coverage), 1.0)
        disagreement_penalty_components = [
            rate for rate in (trigger_disagreement_rate, gate_disagreement_rate) if rate is not None
        ]
        disagreement_penalty = (
            sum(disagreement_penalty_components) / len(disagreement_penalty_components)
            if disagreement_penalty_components
            else 0.0
        )
        reliability_score = max(
            0.0,
            1.0
            - (
                (0.45 * drift_penalty)
                + (0.25 * spread_penalty)
                + (0.15 * volume_penalty)
                + (0.15 * disagreement_penalty)
            ),
        )

        scores[symbol] = {
            "sample_count": sample_count,
            "median_abs_price_drift_bps": median_abs_drift_bps,
            "median_spread_ratio": median_spread_ratio,
            "median_volume_coverage_ratio": median_volume_coverage,
            "trigger_disagreement_rate": trigger_disagreement_rate,
            "trigger_agreement_rate": (
                (1.0 - trigger_disagreement_rate) if trigger_disagreement_rate is not None else None
            ),
            "gate_disagreement_rate": gate_disagreement_rate,
            "gate_agreement_rate": (
                (1.0 - gate_disagreement_rate) if gate_disagreement_rate is not None else None
            ),
            "reliability_score": reliability_score,
        }
    return scores


__all__ = [
    "fetch_reference_snapshot",
    "fetch_reference_minute_bar_snapshot",
    "compute_drift_metrics",
    "normalize_signal_side",
    "derive_signal_agreement",
    "build_symbol_reliability_scores",
]
