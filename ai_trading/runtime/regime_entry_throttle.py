"""Pure regime-specific entry throttle evidence helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time
import math
from typing import Any, Literal
from zoneinfo import ZoneInfo


Action = Literal["allow", "reduce_size", "block_new_entries", "observe"]
SessionRegime = Literal["opening", "midday", "closing"]
VolatilityRegime = Literal["normal", "high"]
SpreadRegime = Literal["normal", "wide"]
ProviderRegime = Literal["healthy", "degraded"]

_NY_TZ = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class RegimeEntryThrottleConfig:
    """Thresholds for classifying entry-throttle evidence."""

    min_sample_count: int = 30
    max_evidence_age_seconds: float = 300.0
    opening_minutes: int = 30
    closing_minutes: int = 30
    high_volatility_bps: float = 75.0
    wide_spread_bps: float = 20.0
    provider_min_success_rate: float = 0.98
    opening_closing_qty_scale: float = 0.75
    high_volatility_qty_scale: float = 0.50
    wide_spread_qty_scale: float = 0.50
    live_canary_qty_scale: float = 0.25


@dataclass(frozen=True)
class RegimeEntryThrottleEvidence:
    """Evidence snapshot used to classify entry risk."""

    observed_at: datetime | None = None
    sample_count: int | None = None
    volatility_bps: float | None = None
    spread_bps: float | None = None
    provider_healthy: bool | None = None
    provider_success_rate: float | None = None
    provider_error_rate: float | None = None


def _as_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _as_int(value: Any) -> int | None:
    parsed = _as_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "healthy", "ok"}:
        return True
    if text in {"0", "false", "no", "off", "degraded", "unhealthy"}:
        return False
    return None


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif value not in (None, ""):
        text = str(value).strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _mapping_value(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def normalize_regime_entry_throttle_evidence(
    evidence: Mapping[str, Any] | RegimeEntryThrottleEvidence | None,
) -> RegimeEntryThrottleEvidence:
    """Normalize mapping-shaped evidence into the helper dataclass."""

    if isinstance(evidence, RegimeEntryThrottleEvidence):
        return evidence
    if not isinstance(evidence, Mapping):
        return RegimeEntryThrottleEvidence()
    provider_state = str(
        _mapping_value(evidence, "provider_state", "provider_status", "provider_regime") or ""
    ).strip().lower()
    provider_healthy = _as_bool(_mapping_value(evidence, "provider_healthy", "provider_ok"))
    if provider_healthy is None and provider_state:
        provider_healthy = provider_state in {"healthy", "ok", "available", "normal"}
    return RegimeEntryThrottleEvidence(
        observed_at=_parse_datetime(
            _mapping_value(evidence, "observed_at", "updated_at", "as_of", "timestamp", "ts")
        ),
        sample_count=_as_int(_mapping_value(evidence, "sample_count", "samples", "n")),
        volatility_bps=_as_float(
            _mapping_value(evidence, "volatility_bps", "realized_volatility_bps", "vol_bps")
        ),
        spread_bps=_as_float(_mapping_value(evidence, "spread_bps", "quote_spread_bps")),
        provider_healthy=provider_healthy,
        provider_success_rate=_as_float(
            _mapping_value(evidence, "provider_success_rate", "success_rate")
        ),
        provider_error_rate=_as_float(_mapping_value(evidence, "provider_error_rate", "error_rate")),
    )


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def classify_session_regime(
    now: datetime,
    *,
    config: RegimeEntryThrottleConfig | None = None,
) -> SessionRegime:
    """Classify the regular-session time slice for entry throttling."""

    cfg = config or RegimeEntryThrottleConfig()
    local = now if now.tzinfo is not None else now.replace(tzinfo=_NY_TZ)
    local = local.astimezone(_NY_TZ)
    session_start = datetime.combine(local.date(), time(9, 30), tzinfo=_NY_TZ)
    session_end = datetime.combine(local.date(), time(16, 0), tzinfo=_NY_TZ)
    minutes_from_open = (local - session_start).total_seconds() / 60.0
    minutes_to_close = (session_end - local).total_seconds() / 60.0
    if minutes_from_open < float(cfg.opening_minutes):
        return "opening"
    if minutes_to_close <= float(cfg.closing_minutes):
        return "closing"
    return "midday"


def _classify_provider(
    evidence: RegimeEntryThrottleEvidence,
    cfg: RegimeEntryThrottleConfig,
    reasons: list[str],
) -> ProviderRegime:
    if evidence.provider_healthy is False:
        reasons.append("provider_degraded")
        return "degraded"
    if evidence.provider_error_rate is not None and evidence.provider_error_rate > (
        1.0 - float(cfg.provider_min_success_rate)
    ):
        reasons.append("provider_error_rate_high")
        return "degraded"
    if evidence.provider_success_rate is not None:
        if evidence.provider_success_rate < float(cfg.provider_min_success_rate):
            reasons.append("provider_success_rate_low")
            return "degraded"
        return "healthy"
    if evidence.provider_healthy is True:
        return "healthy"
    reasons.append("provider_evidence_missing")
    return "degraded"


def evaluate_regime_entry_throttle(
    evidence: Mapping[str, Any] | RegimeEntryThrottleEvidence | None,
    *,
    now: datetime | None = None,
    live_canary: bool = False,
    enforce: bool = True,
    config: RegimeEntryThrottleConfig | None = None,
) -> dict[str, Any]:
    """Evaluate regime evidence and return an entry-throttle decision.

    Missing or stale evidence is intentionally conservative when ``enforce`` is
    true. Use ``enforce=False`` for report-only surfacing.
    """

    cfg = config or RegimeEntryThrottleConfig()
    current = (now or datetime.now(UTC)).astimezone(UTC)
    normalized = normalize_regime_entry_throttle_evidence(evidence)
    reasons: list[str] = []

    sample_count = normalized.sample_count
    sample_sufficient = sample_count is not None and sample_count >= int(cfg.min_sample_count)
    if not sample_sufficient:
        reasons.append("sample_insufficient")

    observed_at = normalized.observed_at
    age_seconds: float | None = None
    fresh = False
    freshness_reason = "missing"
    if observed_at is not None:
        age_seconds = max(0.0, (current - observed_at.astimezone(UTC)).total_seconds())
        fresh = age_seconds <= float(cfg.max_evidence_age_seconds)
        freshness_reason = "fresh" if fresh else "stale"
    if not fresh:
        reasons.append("evidence_stale" if observed_at is not None else "evidence_missing")

    session_regime = classify_session_regime(current, config=cfg)
    volatility_regime: VolatilityRegime
    if normalized.volatility_bps is None:
        volatility_regime = "high"
        reasons.append("volatility_evidence_missing")
    elif normalized.volatility_bps >= float(cfg.high_volatility_bps):
        volatility_regime = "high"
        reasons.append("volatility_high")
    else:
        volatility_regime = "normal"

    spread_regime: SpreadRegime
    if normalized.spread_bps is None:
        spread_regime = "wide"
        reasons.append("spread_evidence_missing")
    elif normalized.spread_bps >= float(cfg.wide_spread_bps):
        spread_regime = "wide"
        reasons.append("spread_wide")
    else:
        spread_regime = "normal"

    provider_regime = _classify_provider(normalized, cfg, reasons)

    qty_scale = 1.0
    action: Action = "allow"
    hard_block = (
        not sample_sufficient
        or not fresh
        or provider_regime == "degraded"
        or normalized.volatility_bps is None
        or normalized.spread_bps is None
    )
    if live_canary and (volatility_regime == "high" or spread_regime == "wide"):
        hard_block = True
        reasons.append("live_canary_strict_regime_gate")
    if volatility_regime == "high" and spread_regime == "wide":
        hard_block = True
        reasons.append("combined_volatility_spread_risk")

    if not enforce:
        action = "observe"
        reasons.append("observe_only")
    elif hard_block:
        action = "block_new_entries"
        qty_scale = 0.0
    else:
        if session_regime in {"opening", "closing"}:
            qty_scale *= float(cfg.opening_closing_qty_scale)
            reasons.append(f"{session_regime}_session")
        if volatility_regime == "high":
            qty_scale *= float(cfg.high_volatility_qty_scale)
        if spread_regime == "wide":
            qty_scale *= float(cfg.wide_spread_qty_scale)
        if live_canary and qty_scale < 1.0:
            qty_scale = min(qty_scale, float(cfg.live_canary_qty_scale))
            reasons.append("live_canary_stricter_scale")
        if qty_scale < 1.0:
            action = "reduce_size"

    unique_reasons = list(dict.fromkeys(reasons))
    return {
        "schema_version": "1.0.0",
        "artifact_type": "regime_entry_throttle_evaluation",
        "generated_at": _iso(current),
        "action": action,
        "qty_scale": float(max(0.0, min(1.0, qty_scale))),
        "reasons": unique_reasons,
        "live_canary": bool(live_canary),
        "enforced": bool(enforce),
        "sample_sufficiency": {
            "sufficient": bool(sample_sufficient),
            "sample_count": sample_count,
            "min_sample_count": int(cfg.min_sample_count),
        },
        "freshness": {
            "fresh": bool(fresh),
            "status": freshness_reason,
            "age_seconds": age_seconds,
            "max_age_seconds": float(cfg.max_evidence_age_seconds),
            "observed_at": _iso(observed_at),
        },
        "session_regime": session_regime,
        "volatility_regime": volatility_regime,
        "spread_regime": spread_regime,
        "provider_regime": provider_regime,
        "thresholds": {
            "high_volatility_bps": float(cfg.high_volatility_bps),
            "wide_spread_bps": float(cfg.wide_spread_bps),
            "provider_min_success_rate": float(cfg.provider_min_success_rate),
        },
        "inputs": {
            "volatility_bps": normalized.volatility_bps,
            "spread_bps": normalized.spread_bps,
            "provider_healthy": normalized.provider_healthy,
            "provider_success_rate": normalized.provider_success_rate,
            "provider_error_rate": normalized.provider_error_rate,
        },
    }


def build_regime_entry_throttle_report(
    evaluations: Sequence[Mapping[str, Any]],
    *,
    report_date: str | None = None,
) -> dict[str, Any]:
    """Summarize evaluation rows for daily/reporting artifacts."""

    rows = [row for row in evaluations if isinstance(row, Mapping)]
    if report_date:
        rows = [
            row
            for row in rows
            if str(row.get("generated_at") or row.get("ts") or row.get("timestamp") or "").startswith(
                report_date
            )
        ]
    action_counts = Counter(str(row.get("action") or "unknown") for row in rows)
    reason_counts: Counter[str] = Counter()
    session_counts = Counter(str(row.get("session_regime") or "unknown") for row in rows)
    volatility_counts = Counter(str(row.get("volatility_regime") or "unknown") for row in rows)
    spread_counts = Counter(str(row.get("spread_regime") or "unknown") for row in rows)
    provider_counts = Counter(str(row.get("provider_regime") or "unknown") for row in rows)
    qty_scales: list[float] = []
    stale_rows = 0
    insufficient_rows = 0
    for row in rows:
        reasons = row.get("reasons")
        if isinstance(reasons, Sequence) and not isinstance(reasons, (str, bytes)):
            reason_counts.update(str(reason) for reason in reasons)
        parsed_scale = _as_float(row.get("qty_scale"))
        if parsed_scale is not None:
            qty_scales.append(parsed_scale)
        freshness = row.get("freshness")
        if isinstance(freshness, Mapping) and not bool(freshness.get("fresh")):
            stale_rows += 1
        sufficiency = row.get("sample_sufficiency")
        if isinstance(sufficiency, Mapping) and not bool(sufficiency.get("sufficient")):
            insufficient_rows += 1

    latest = max(rows, key=lambda row: str(row.get("generated_at") or ""), default=None)
    return {
        "schema_version": "1.0.0",
        "artifact_type": "regime_entry_throttle_report",
        "report_date": report_date,
        "generated_at": _iso(datetime.now(UTC)),
        "evaluations": len(rows),
        "actions": dict(sorted(action_counts.items())),
        "reasons": dict(sorted(reason_counts.items())),
        "sample_sufficiency": {
            "insufficient": int(insufficient_rows),
            "sufficient": int(len(rows) - insufficient_rows),
        },
        "freshness": {
            "fresh": int(len(rows) - stale_rows),
            "stale_or_missing": int(stale_rows),
        },
        "regimes": {
            "session": dict(sorted(session_counts.items())),
            "volatility": dict(sorted(volatility_counts.items())),
            "spread": dict(sorted(spread_counts.items())),
            "provider": dict(sorted(provider_counts.items())),
        },
        "qty_scale": {
            "min": min(qty_scales) if qty_scales else None,
            "avg": (sum(qty_scales) / len(qty_scales)) if qty_scales else None,
        },
        "latest": dict(latest) if latest is not None else None,
    }


__all__ = [
    "RegimeEntryThrottleConfig",
    "RegimeEntryThrottleEvidence",
    "build_regime_entry_throttle_report",
    "classify_session_regime",
    "evaluate_regime_entry_throttle",
    "normalize_regime_entry_throttle_evidence",
]
