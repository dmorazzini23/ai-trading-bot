from __future__ import annotations

from datetime import date
from pathlib import Path

from ai_trading.governance.rollout import (
    BurnInPolicy,
    CapitalRampPolicy,
    RolloutState,
    apply_rollout_policies,
    load_rollout_state,
    save_rollout_state,
)


def test_burn_in_promotes_after_paper_cycles_and_days() -> None:
    state = RolloutState()
    burn_in = BurnInPolicy(
        enabled=True,
        min_paper_cycles=2,
        min_paper_days=1,
        require_policy_hash_stable=True,
        require_config_hash_stable=True,
    )
    ramp = CapitalRampPolicy(
        enabled=False,
        phases=(1.0,),
        min_cycles_per_phase=1,
        max_pacing_hit_rate_pct=100.0,
        max_pending_oldest_age_sec=10_000.0,
        max_calibration_ece=1.0,
        max_calibration_brier=1.0,
        downgrade_on_breach=False,
    )

    state, summary = apply_rollout_policies(
        state=state,
        burn_in=burn_in,
        ramp=ramp,
        execution_mode="paper",
        policy_hash="hash_a",
        config_hash="cfg_a",
        today=date(2026, 3, 1),
        telemetry={},
    )
    assert summary["burn_in_ready"] is False
    assert summary["burn_in_paper_cycles"] == 1

    state, summary = apply_rollout_policies(
        state=state,
        burn_in=burn_in,
        ramp=ramp,
        execution_mode="paper",
        policy_hash="hash_a",
        config_hash="cfg_a",
        today=date(2026, 3, 1),
        telemetry={},
    )
    assert summary["burn_in_ready"] is True
    assert summary["burn_in_paper_cycles"] == 2

    _, summary = apply_rollout_policies(
        state=state,
        burn_in=burn_in,
        ramp=ramp,
        execution_mode="live",
        policy_hash="hash_a",
        config_hash="cfg_a",
        today=date(2026, 3, 2),
        telemetry={},
    )
    assert summary["burn_in_ready"] is True
    assert summary["burn_in_reason"] == ""


def test_burn_in_blocks_live_on_hash_mismatch() -> None:
    state = RolloutState(
        burn_in_paper_cycles=10,
        burn_in_paper_days=("2026-03-01", "2026-03-02"),
        burn_in_policy_hash="hash_a",
        burn_in_config_hash="cfg_a",
    )
    burn_in = BurnInPolicy(
        enabled=True,
        min_paper_cycles=2,
        min_paper_days=2,
        require_policy_hash_stable=True,
        require_config_hash_stable=True,
    )
    ramp = CapitalRampPolicy(
        enabled=False,
        phases=(1.0,),
        min_cycles_per_phase=1,
        max_pacing_hit_rate_pct=100.0,
        max_pending_oldest_age_sec=10_000.0,
        max_calibration_ece=1.0,
        max_calibration_brier=1.0,
        downgrade_on_breach=False,
    )

    _, summary = apply_rollout_policies(
        state=state,
        burn_in=burn_in,
        ramp=ramp,
        execution_mode="live",
        policy_hash="hash_b",
        config_hash="cfg_a",
        today=date(2026, 3, 3),
        telemetry={},
    )
    assert summary["burn_in_ready"] is False
    assert summary["burn_in_reason"] == "BURN_IN_POLICY_HASH_MISMATCH"


def test_capital_ramp_upgrades_and_downgrades_on_breach() -> None:
    state = RolloutState()
    burn_in = BurnInPolicy(
        enabled=False,
        min_paper_cycles=1,
        min_paper_days=1,
        require_policy_hash_stable=False,
        require_config_hash_stable=False,
    )
    ramp = CapitalRampPolicy(
        enabled=True,
        phases=(0.25, 0.50, 1.00),
        min_cycles_per_phase=2,
        max_pacing_hit_rate_pct=30.0,
        max_pending_oldest_age_sec=240.0,
        max_calibration_ece=0.15,
        max_calibration_brier=0.35,
        downgrade_on_breach=True,
    )
    healthy = {
        "order_pacing_cap_hit_rate_pct": 5.0,
        "pending_oldest_age_sec": 0.0,
        "live_calibration_ece": 0.01,
        "live_calibration_brier": 0.10,
    }
    breached = {
        "order_pacing_cap_hit_rate_pct": 50.0,
        "pending_oldest_age_sec": 0.0,
        "live_calibration_ece": 0.01,
        "live_calibration_brier": 0.10,
    }

    state, summary = apply_rollout_policies(
        state=state,
        burn_in=burn_in,
        ramp=ramp,
        execution_mode="live",
        policy_hash="hash_a",
        config_hash="cfg_a",
        today=date(2026, 3, 1),
        telemetry=healthy,
    )
    assert summary["capital_ramp"]["phase_index"] == 0
    assert summary["capital_ramp"]["multiplier"] == 0.25

    state, summary = apply_rollout_policies(
        state=state,
        burn_in=burn_in,
        ramp=ramp,
        execution_mode="live",
        policy_hash="hash_a",
        config_hash="cfg_a",
        today=date(2026, 3, 1),
        telemetry=healthy,
    )
    assert summary["capital_ramp"]["phase_index"] == 1
    assert summary["capital_ramp"]["multiplier"] == 0.50

    state, summary = apply_rollout_policies(
        state=state,
        burn_in=burn_in,
        ramp=ramp,
        execution_mode="live",
        policy_hash="hash_a",
        config_hash="cfg_a",
        today=date(2026, 3, 1),
        telemetry=healthy,
    )
    state, summary = apply_rollout_policies(
        state=state,
        burn_in=burn_in,
        ramp=ramp,
        execution_mode="live",
        policy_hash="hash_a",
        config_hash="cfg_a",
        today=date(2026, 3, 1),
        telemetry=healthy,
    )
    assert summary["capital_ramp"]["phase_index"] == 2
    assert summary["capital_ramp"]["multiplier"] == 1.00

    state, summary = apply_rollout_policies(
        state=state,
        burn_in=burn_in,
        ramp=ramp,
        execution_mode="live",
        policy_hash="hash_a",
        config_hash="cfg_a",
        today=date(2026, 3, 1),
        telemetry=breached,
    )
    assert summary["capital_ramp"]["phase_index"] == 1
    assert summary["capital_ramp"]["multiplier"] == 0.50
    assert summary["capital_ramp"]["transition"] == "downgrade_on_breach"


def test_rollout_state_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "rollout_state.json"
    state = RolloutState(
        burn_in_paper_cycles=5,
        burn_in_paper_days=("2026-03-01",),
        burn_in_policy_hash="policy",
        burn_in_config_hash="config",
        ramp_phase_index=1,
        ramp_phase_cycles=7,
        ramp_multiplier=0.5,
    )
    save_rollout_state(path, state)
    loaded = load_rollout_state(path)
    assert loaded.burn_in_paper_cycles == 5
    assert loaded.burn_in_policy_hash == "policy"
    assert loaded.ramp_phase_index == 1
    assert loaded.ramp_multiplier == 0.5

