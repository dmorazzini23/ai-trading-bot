from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

from ai_trading.core.netting_symbol_prelude import prepare_netting_symbol_prelude


def _base_kwargs() -> dict[str, Any]:
    bar_ts = datetime(2026, 4, 19, 15, 29, tzinfo=UTC)
    return {
        "state": SimpleNamespace(current_regime="sideways"),
        "symbol": "AAPL",
        "now": datetime(2026, 4, 19, 15, 30, tzinfo=UTC),
        "current_shares": 5,
        "delta_shares": 10,
        "price": 100.0,
        "net_target": SimpleNamespace(
            bar_ts=bar_ts,
            proposals=[SimpleNamespace(sleeve="day", score=0.6, confidence=0.8)],
        ),
        "policy_disabled_sleeves": set(),
        "policy_rollback_disabled_slices": [],
        "sleeve_configs_map": {
            "day": SimpleNamespace(
                entry_threshold=0.2,
                exit_threshold=0.1,
                flip_threshold=0.3,
                reentry_threshold=0.6,
                deadband_dollars=50.0,
                cost_k=1.5,
                edge_scale_bps=20.0,
                turnover_cap_dollars=0.0,
            )
        },
        "candidate_expected_net_edge": {"AAPL": 8.0},
        "alpha_time_stop_enabled": False,
        "alpha_time_stop_sec": 0.0,
        "alpha_time_stop_max_expected_edge_bps": 0.0,
        "opportunity_quality_enabled": False,
        "opportunity_allowed_symbols": set(),
        "opportunity_openings_only": False,
        "opportunity_quality_by_symbol": {},
        "opportunity_quality_gate": {},
        "opportunity_top_quantile": 0.2,
        "alpha_time_decay_enabled": False,
        "alpha_stale_signal_sec": 0.0,
        "live_execution_mode": True,
        "burn_in_live_ready": True,
        "burn_in_live_reason": "",
        "ramp_live_multiplier": 1.0,
        "ramp_summary": {},
        "gates": [],
        "position_opened_at_func": lambda state, symbol: None,
        "exit_policy_pressure_context_func": lambda *args, **kwargs: {},
    }


def test_prepare_netting_symbol_prelude_blocks_disabled_sleeve() -> None:
    kwargs = _base_kwargs()
    kwargs["policy_disabled_sleeves"] = {"day"}
    kwargs["policy_rollback_disabled_slices"] = ["day"]

    result = prepare_netting_symbol_prelude(**cast(Any, kwargs))

    assert result.blocked_reason == "POLICY_ABLATION_SLEEVE_BLOCK"
    assert result.gates_added == ("POLICY_ABLATION_SLEEVE_BLOCK",)
    assert result.snapshot_updates["policy_rollback"]["disabled_sleeves_for_symbol"] == ["day"]


def test_prepare_netting_symbol_prelude_scales_capital_ramp() -> None:
    kwargs = _base_kwargs()
    kwargs["current_shares"] = 0
    kwargs["delta_shares"] = 10
    kwargs["ramp_live_multiplier"] = 0.4
    kwargs["ramp_summary"] = {"phase_index": 1, "phase_cycles": 5, "breached": False, "transition": "steady"}

    result = prepare_netting_symbol_prelude(**cast(Any, kwargs))

    assert result.blocked_reason is None
    assert result.delta_shares == 4
    assert result.target_shares == 4
    assert result.target_dollars == 400.0
    assert "CAPITAL_RAMP_SCALE" in result.gates_added
    assert result.snapshot_updates["capital_ramp"]["multiplier"] == 0.4
