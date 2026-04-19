from __future__ import annotations

from types import SimpleNamespace

from ai_trading.core.netting_symbol_adjustments import apply_symbol_adjustments


class _LoggerStub:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object] | None]] = []

    def info(self, event: str, *, extra: dict[str, object] | None = None) -> None:
        self.events.append((event, extra))


def test_apply_symbol_adjustments_blocks_when_adaptive_scale_zeroes_qty() -> None:
    result = apply_symbol_adjustments(
        symbol="AAPL",
        state=SimpleNamespace(degraded_providers=set()),
        cfg=SimpleNamespace(global_max_symbol_dollars=25_000.0, execution_min_qty=1, execution_min_notional=1.0),
        current_shares=0,
        delta_shares=1,
        price=100.0,
        expanding_exposure=True,
        initial_requested_delta_shares=1,
        symbol_adaptive_profiles={"AAPL": {"scale": 0.05}},
        slo_derisk_details={},
        primary_feed_derisk={},
        penalty_overlap_coordination_enabled=False,
        penalty_overlap_weight_dampen=0.5,
        penalty_overlap_min_scale_floor=0.55,
        uncertainty_capital_state={},
        safe_float=lambda value: float(value) if value is not None else None,
        resolve_uncertainty_capital_auto_controls_func=lambda **kwargs: {},
        clip_delta_to_symbol_notional_cap_func=lambda **kwargs: (kwargs["delta_shares"], None),
        logger=_LoggerStub(),
    )

    assert result.blocked_reason == "SYMBOL_EXPECTANCY_SLIPPAGE_BLOCK"
    assert result.blocked_metrics == {"symbol_adaptive_sizing": {"scale": 0.05}}
    assert "SYMBOL_EXPECTANCY_SLIPPAGE_BLOCK" in result.gates_added


def test_apply_symbol_adjustments_scales_uncertainty_and_clips_symbol_cap() -> None:
    result = apply_symbol_adjustments(
        symbol="AAPL",
        state=SimpleNamespace(degraded_providers={"primary"}),
        cfg=SimpleNamespace(global_max_symbol_dollars=25_000.0, execution_min_qty=1, execution_min_notional=1.0),
        current_shares=0,
        delta_shares=100,
        price=100.0,
        expanding_exposure=True,
        initial_requested_delta_shares=100,
        symbol_adaptive_profiles={},
        slo_derisk_details={
            "calibration_ece": 0.15,
            "calibration_brier": 0.35,
            "execution_drift_bps": 35.0,
            "drift_psi": 0.30,
            "label_drift_psi": 0.30,
            "residual_drift_psi": 0.30,
            "calibration_ece_samples": 10,
            "calibration_brier_samples": 10,
            "drift_samples": 10,
        },
        primary_feed_derisk={"triggered": True},
        penalty_overlap_coordination_enabled=True,
        penalty_overlap_weight_dampen=0.5,
        penalty_overlap_min_scale_floor=0.55,
        uncertainty_capital_state={},
        safe_float=lambda value: float(value) if value is not None else None,
        resolve_uncertainty_capital_auto_controls_func=lambda **kwargs: {
            "weight": kwargs["base_weight"],
            "min_scale": kwargs["base_min_scale"],
            "effective_score": kwargs["raw_score"],
        },
        clip_delta_to_symbol_notional_cap_func=lambda **kwargs: (
            40,
            {"max_symbol_notional": 4000.0, "price": 100.0},
        ),
        logger=_LoggerStub(),
    )

    assert result.blocked_reason is None
    assert result.delta_shares == 40
    assert "UNCERTAINTY_CAPITAL_SCALE" in result.gates_added
    assert "RISK_CAP_SYMBOL_CLIP" in result.gates_added
    assert result.snapshot_updates["symbol_cap_clip"]["max_symbol_notional"] == 4000.0
    assert result.uncertainty_event is not None
    assert result.uncertainty_event["scaled"] is True
