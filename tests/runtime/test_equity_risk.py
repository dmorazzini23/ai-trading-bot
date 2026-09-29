"""Selected account-equity limits require verified, durable evidence."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from ai_trading.runtime.equity_risk import (
    EquityRiskBaseline,
    EquityRiskEvidenceError,
    VerifiedEquityObservation,
    evaluate_selected_limits,
    load_approved_baseline,
    persist_high_water,
)


NOW = datetime(2026, 9, 29, 15, 0, tzinfo=UTC)


def _observation(**updates: object) -> VerifiedEquityObservation:
    payload: dict[str, object] = {
        "source": "broker_account_and_complete_activities",
        "account_id": "live-account-1",
        "equity": "995",
        "observed_at": (NOW - timedelta(seconds=2)).isoformat(),
        "received_at": (NOW - timedelta(seconds=1)).isoformat(),
        "cumulative_external_flow": "0",
        "activity_window_complete": True,
        "cash_flow_timing_verified": True,
    }
    payload.update(updates)
    return VerifiedEquityObservation.from_evidence(payload)


def _record(**updates: object) -> dict[str, object]:
    record: dict[str, object] = {
        "artifact_type": "approved_live_equity_baseline",
        "account_id": "live-account-1",
        "starting_equity": "1000",
        "session_date": "2026-09-29",
        "session_start_adjusted_equity": "1000",
        "high_water_adjusted_equity": "1000",
        "last_observed_at": (NOW - timedelta(hours=1)).isoformat(),
    }
    record.update(updates)
    return record


def _assess(
    observation: VerifiedEquityObservation | None = None,
    baseline: EquityRiskBaseline | None = None,
    gross: str = "0",
    cost: str = "0",
) -> dict[str, str | bool]:
    return evaluate_selected_limits(
        observation or _observation(),
        baseline or EquityRiskBaseline.from_record(_record()),
        projected_gross_notional=Decimal(gross),
        estimated_cost_reserve=Decimal(cost),
        now=NOW,
    )


def test_daily_limit_includes_unrealized_equity_and_full_order_reserve() -> None:
    assert _assess(gross="4.99")["allowed"] is True
    result = _assess(gross="5")
    assert result["allowed"] is False
    assert result["daily_loss"] == "5"
    assert result["loss_from_starting_capital"] == "5"
    assert result["daily_limit"] == "10.00"


def test_drawdown_uses_persistent_peak_and_cost_reserve() -> None:
    baseline = EquityRiskBaseline.from_record(_record(high_water_adjusted_equity="1100"))
    result = _assess(_observation(equity="1070"), baseline, gross="2", cost="1")
    assert result["allowed"] is False
    assert result["drawdown"] == "30"
    assert result["drawdown_limit"] == "33.00"


def test_deposit_and_withdrawal_are_neutralized() -> None:
    assert _assess(_observation(equity="1095", cumulative_external_flow="100"))["daily_loss"] == "5"
    assert _assess(_observation(equity="895", cumulative_external_flow="-100"))["daily_loss"] == "5"


@pytest.mark.parametrize("changes,reason", [
    ({"account_id": "different"}, "equity_risk_account_conflict"),
    ({"activity_window_complete": False}, "equity_risk_cash_flow_unverified"),
    ({"cash_flow_timing_verified": False}, "equity_risk_cash_flow_unverified"),
    ({"observed_at": (NOW - timedelta(seconds=62)).isoformat(),
      "received_at": (NOW - timedelta(seconds=61)).isoformat()}, "equity_risk_snapshot_stale"),
    ({"observed_at": (NOW + timedelta(days=1)).isoformat()}, "equity_risk_causality_invalid"),
    ({"observed_at": "2026-09-30T15:00:00+00:00", "received_at": "2026-09-30T15:00:01+00:00"}, "equity_risk_causality_invalid"),
])
def test_invalid_observations_block(changes: dict[str, object], reason: str) -> None:
    with pytest.raises(EquityRiskEvidenceError, match=reason):
        _assess(_observation(**changes))


def test_session_rollover_requires_new_approved_baseline() -> None:
    prior = EquityRiskBaseline.from_record(_record(session_date="2026-09-28"))
    with pytest.raises(EquityRiskEvidenceError, match="equity_risk_session_rollover"):
        _assess(baseline=prior)


def test_missing_or_corrupt_baseline_never_bootstraps(tmp_path: Path) -> None:
    path = tmp_path / "risk.json"
    with pytest.raises(EquityRiskEvidenceError, match="equity_risk_baseline_missing"):
        load_approved_baseline(path)
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(EquityRiskEvidenceError, match="equity_risk_baseline_corrupt"):
        load_approved_baseline(path)
    assert path.read_text(encoding="utf-8") == "{broken"


def test_high_water_advances_across_restart_without_reset(tmp_path: Path) -> None:
    path = tmp_path / "risk.json"
    path.write_text(json.dumps(_record()), encoding="utf-8")
    persist_high_water(path, _observation(equity="1010"), now=NOW)
    assert load_approved_baseline(path).high_water_adjusted_equity == Decimal("1010")
    persist_high_water(path, _observation(equity="995"), now=NOW)
    assert load_approved_baseline(path).high_water_adjusted_equity == Decimal("1010")


def test_untrusted_source_and_date_only_timestamp_rejected() -> None:
    with pytest.raises(EquityRiskEvidenceError, match="equity_risk_source_unverified"):
        _observation(source="strategy_payload")
    with pytest.raises(EquityRiskEvidenceError, match="equity_risk_timestamp_missing"):
        _observation(observed_at="2026-09-29")
