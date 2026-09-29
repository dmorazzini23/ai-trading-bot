"""Fail-closed arithmetic for the selected live-account equity limits.

This module consumes verified broker evidence; it never creates an opening
balance or infers a cash transfer from an equity change.
"""

from __future__ import annotations

import fcntl
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from ai_trading.runtime.atomic_io import atomic_write_text


_EASTERN = ZoneInfo("America/New_York")
_MAX_AGE = timedelta(seconds=60)


class EquityRiskEvidenceError(ValueError):
    """An input cannot safely support a live opening decision."""


def _amount(
    value: Any, *, positive: bool = False, allow_negative: bool = False
) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise EquityRiskEvidenceError("equity_risk_amount_missing")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise EquityRiskEvidenceError("equity_risk_amount_invalid") from exc
    if not result.is_finite() or (positive and result <= 0) or (
        not positive and not allow_negative and result < 0
    ):
        raise EquityRiskEvidenceError("equity_risk_amount_invalid")
    return result


def _instant(value: Any) -> datetime:
    if not isinstance(value, str) or "T" not in value:
        raise EquityRiskEvidenceError("equity_risk_timestamp_missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EquityRiskEvidenceError("equity_risk_timestamp_invalid") from exc
    if parsed.tzinfo is None:
        raise EquityRiskEvidenceError("equity_risk_timestamp_unzoned")
    return parsed.astimezone(UTC)


@dataclass(frozen=True)
class VerifiedEquityObservation:
    account_id: str
    equity: Decimal
    observed_at: datetime
    received_at: datetime
    cumulative_external_flow: Decimal
    activity_window_complete: bool
    cash_flow_timing_verified: bool

    @classmethod
    def from_evidence(cls, value: Mapping[str, Any]) -> "VerifiedEquityObservation":
        if value.get("source") != "broker_account_and_complete_activities":
            raise EquityRiskEvidenceError("equity_risk_source_unverified")
        account_id = str(value.get("account_id") or "").strip()
        if not account_id:
            raise EquityRiskEvidenceError("equity_risk_account_missing")
        return cls(
            account_id=account_id,
            equity=_amount(value.get("equity"), positive=True),
            observed_at=_instant(value.get("observed_at")),
            received_at=_instant(value.get("received_at")),
            cumulative_external_flow=_amount(
                value.get("cumulative_external_flow"), allow_negative=True
            ),
            activity_window_complete=value.get("activity_window_complete") is True,
            cash_flow_timing_verified=value.get("cash_flow_timing_verified") is True,
        )


@dataclass(frozen=True)
class EquityRiskBaseline:
    account_id: str
    starting_equity: Decimal
    session_date: str
    session_start_adjusted_equity: Decimal
    high_water_adjusted_equity: Decimal
    last_observed_at: datetime

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> "EquityRiskBaseline":
        if value.get("artifact_type") != "approved_live_equity_baseline":
            raise EquityRiskEvidenceError("equity_risk_baseline_unapproved")
        account_id = str(value.get("account_id") or "").strip()
        if not account_id:
            raise EquityRiskEvidenceError("equity_risk_account_missing")
        session_date = str(value.get("session_date") or "")
        try:
            datetime.strptime(session_date, "%Y-%m-%d")
        except ValueError as exc:
            raise EquityRiskEvidenceError("equity_risk_session_invalid") from exc
        starting = _amount(value.get("starting_equity"), positive=True)
        high_water = _amount(value.get("high_water_adjusted_equity"), positive=True)
        if high_water < starting:
            raise EquityRiskEvidenceError("equity_risk_high_water_invalid")
        session_start = _amount(value.get("session_start_adjusted_equity"), positive=True)
        if high_water < session_start:
            raise EquityRiskEvidenceError("equity_risk_high_water_invalid")
        return cls(
            account_id=account_id,
            starting_equity=starting,
            session_date=session_date,
            session_start_adjusted_equity=session_start,
            high_water_adjusted_equity=high_water,
            last_observed_at=_instant(value.get("last_observed_at")),
        )


def load_approved_baseline(path: Path) -> EquityRiskBaseline:
    """A missing or damaged high-water record cannot silently reset risk."""

    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EquityRiskEvidenceError("equity_risk_baseline_missing") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EquityRiskEvidenceError("equity_risk_baseline_corrupt") from exc
    if not isinstance(record, dict):
        raise EquityRiskEvidenceError("equity_risk_baseline_corrupt")
    return EquityRiskBaseline.from_record(record)


def evaluate_selected_limits(
    observation: VerifiedEquityObservation,
    baseline: EquityRiskBaseline,
    *,
    projected_gross_notional: Decimal,
    estimated_cost_reserve: Decimal,
    now: datetime | None = None,
) -> dict[str, str | bool]:
    """Reserve full long exposure against both verified account loss ceilings."""

    current_time = (now or datetime.now(UTC)).astimezone(UTC)
    if observation.account_id != baseline.account_id:
        raise EquityRiskEvidenceError("equity_risk_account_conflict")
    if not observation.activity_window_complete or not observation.cash_flow_timing_verified:
        raise EquityRiskEvidenceError("equity_risk_cash_flow_unverified")
    if not (baseline.last_observed_at <= observation.observed_at <= observation.received_at <= current_time):
        raise EquityRiskEvidenceError("equity_risk_causality_invalid")
    if current_time - observation.received_at > _MAX_AGE:
        raise EquityRiskEvidenceError("equity_risk_snapshot_stale")
    if observation.observed_at.astimezone(_EASTERN).date().isoformat() != baseline.session_date:
        raise EquityRiskEvidenceError("equity_risk_session_rollover")
    gross = _amount(projected_gross_notional)
    cost = _amount(estimated_cost_reserve)
    adjusted = observation.equity - observation.cumulative_external_flow
    if adjusted <= 0:
        raise EquityRiskEvidenceError("equity_risk_adjusted_equity_invalid")
    peak = max(baseline.high_water_adjusted_equity, adjusted)
    daily_loss = max(Decimal(0), baseline.session_start_adjusted_equity - adjusted)
    capital_loss = max(Decimal(0), baseline.starting_equity - adjusted)
    drawdown = max(Decimal(0), peak - adjusted)
    reserve = gross + cost
    daily_limit = baseline.session_start_adjusted_equity * Decimal("0.01")
    drawdown_limit = peak * Decimal("0.03")
    return {
        "allowed": daily_loss + reserve < daily_limit and drawdown + reserve < drawdown_limit,
        "adjusted_equity": str(adjusted),
        "high_water_adjusted_equity": str(peak),
        "daily_loss": str(daily_loss),
        "loss_from_starting_capital": str(capital_loss),
        "drawdown": str(drawdown),
        "reserved_loss": str(reserve),
        "daily_limit": str(daily_limit),
        "drawdown_limit": str(drawdown_limit),
    }


def persist_high_water(
    path: Path, observation: VerifiedEquityObservation, *, now: datetime | None = None
) -> None:
    """Advance a previously approved high water atomically, never bootstrap it."""

    lock_path = path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            baseline = load_approved_baseline(path)
            evaluate_selected_limits(
                observation, baseline,
                projected_gross_notional=Decimal(0),
                estimated_cost_reserve=Decimal(0),
                now=now,
            )
            adjusted = observation.equity - observation.cumulative_external_flow
            record = json.loads(path.read_text(encoding="utf-8"))
            record["high_water_adjusted_equity"] = str(
                max(baseline.high_water_adjusted_equity, adjusted)
            )
            record["last_observed_at"] = observation.observed_at.isoformat()
            atomic_write_text(path, json.dumps(record, sort_keys=True) + "\n")
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
