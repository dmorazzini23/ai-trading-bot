from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from ai_trading.core.netting_symbol_cycle import (
    NettingSymbolProcessor,
    process_netting_symbol,
)
from ai_trading.risk.liquidity_regime import LiquidityFeatures


class _DummyIntent:
    def to_contract(self) -> dict[str, Any]:
        return {"intent": "dummy"}


def _make_net_target(*, bar_ts: datetime, target_dollars: float = 100.0) -> Any:
    return SimpleNamespace(
        symbol="AAPL",
        bar_ts=bar_ts,
        target_dollars=float(target_dollars),
        target_shares=0,
        reasons=[],
        proposals=[
            SimpleNamespace(
                sleeve="main",
                reason_code=None,
                score=0.9,
                confidence=0.9,
            )
        ],
    )


def _make_processor(**overrides: Any) -> tuple[NettingSymbolProcessor, list[dict[str, Any]]]:
    now = datetime(2026, 4, 19, tzinfo=UTC)
    records: list[dict[str, Any]] = []

    def record_decision_func(**kwargs: Any) -> dict[str, Any]:
        records.append(kwargs)
        return kwargs

    def safe_float(value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except Exception:
            return None

    def prepare_symbol_prelude_func(**kwargs: Any) -> Any:
        current_shares = int(kwargs["current_shares"])
        delta_shares = int(kwargs["delta_shares"])
        price = float(kwargs["price"])
        return SimpleNamespace(
            delta_shares=delta_shares,
            target_shares=current_shares + delta_shares,
            target_dollars=float((current_shares + delta_shares) * price),
            gates_added=(),
            snapshot_updates={},
            blocked_reason=None,
        )

    def apply_symbol_adjustments_func(**kwargs: Any) -> Any:
        current_shares = int(kwargs["current_shares"])
        delta_shares = int(kwargs["delta_shares"])
        price = float(kwargs["price"])
        return SimpleNamespace(
            delta_shares=delta_shares,
            target_shares=current_shares + delta_shares,
            target_dollars=float((current_shares + delta_shares) * price),
            gates_added=(),
            snapshot_updates={},
            blocked_reason=None,
            blocked_metrics=None,
            uncertainty_event=None,
        )

    def prepare_symbol_approval_func(**kwargs: Any) -> Any:
        current_shares = int(kwargs["current_shares"])
        delta_shares = int(kwargs["delta_shares"])
        price = float(kwargs["price"])
        return SimpleNamespace(
            delta_shares=delta_shares,
            target_shares=current_shares + delta_shares,
            target_dollars=float((current_shares + delta_shares) * price),
            side="buy" if delta_shares > 0 else "sell",
            opening_trade=current_shares == 0 and delta_shares != 0,
            gates_added=(),
            snapshot_updates={},
            blocked_reason=None,
            blocked_metrics=None,
            approval=SimpleNamespace(expected_net_edge_bps=3.0),
            approval_context={"approved": True},
        )

    def prepare_submit_prelude_func(**kwargs: Any) -> Any:
        return SimpleNamespace(
            execution_intent_context=SimpleNamespace(
                client_order_id="cid-1",
                pretrade_intent=_DummyIntent(),
                order_lineage_metadata={"lineage": "x"},
                order_annotations={"annotation": "x"},
                decision_trace_id="trace-1",
            ),
            submit_quote_source="nbbo",
            submit_bid_at_arrival=9.9,
            submit_ask_at_arrival=10.1,
            submit_mid_at_arrival=10.0,
            submit_arrival_price=10.0,
            gates_added=(),
            snapshot_updates={},
            blocked_reason=None,
            blocked_metrics=None,
            blocked_order_intent=None,
        )

    def execute_submission_func(**kwargs: Any) -> Any:
        return SimpleNamespace(
            status="submitted",
            gates_added=("OK_TRADE",),
            attempted_increment=1,
            submitted_increment=1,
            metrics={"fill_prob": 0.8},
            tca_record={"arrival_price": 10.0},
            order_payload={"client_order_id": "cid-1"},
            decision_trace_id="trace-1",
            order_intent_contract={"intent": "dummy"},
        )

    processor = NettingSymbolProcessor(
        state=SimpleNamespace(
            halt_trading=False,
            halt_reason=None,
            stop_lock={},
            last_order_bar_ts={},
            degraded_providers=set(),
        ),
        runtime=SimpleNamespace(),
        cfg=SimpleNamespace(seed="seed"),
        now=now,
        logger=SimpleNamespace(debug=lambda *a, **k: None),
        decision_snapshot_template={"liquidity_regime": "normal"},
        latest_price={"AAPL": 10.0},
        latest_liquidity={"AAPL": LiquidityFeatures(rolling_volume=1000.0, spread_bps=2.0, volatility_proxy=0.01)},
        positions={},
        skip_reasons={},
        kill_switch=False,
        policy_disabled_sleeves=set(),
        policy_rollback_disabled_slices=(),
        sleeve_configs_map={"main": SimpleNamespace(reentry_threshold=0.6)},
        candidate_expected_net_edge={"AAPL": 5.0},
        candidate_expected_capture={"AAPL": 4.0},
        alpha_time_stop_enabled=False,
        alpha_time_stop_sec=0.0,
        alpha_time_stop_max_expected_edge_bps=0.0,
        opportunity_quality_enabled=False,
        opportunity_allowed_symbols=set(),
        opportunity_openings_only=False,
        opportunity_quality_by_symbol={},
        opportunity_quality_gate={},
        opportunity_top_quantile=0.2,
        alpha_time_decay_enabled=False,
        alpha_stale_signal_sec=0.0,
        live_execution_mode=False,
        burn_in_live_ready=True,
        burn_in_live_reason="",
        ramp_live_multiplier=1.0,
        ramp_summary={},
        liq_regime_enabled=False,
        thin_spread_bps=10.0,
        thin_vol_mult=2.0,
        primary_feed_derisk={},
        quarantine_enabled=False,
        quarantine_manager=None,
        quarantine_apply_sleeve=False,
        quarantine_apply_symbol=False,
        quarantine_mode="block",
        event_blackout_enabled=False,
        event_blackout_days=0,
        event_blackout_cache={},
        alpha_decay_deweight_enabled=False,
        alpha_decay_qty_step=0.0,
        alpha_decay_qty_max_deweight=0.0,
        capacity_throttle_enabled=False,
        capacity_spread_soft_bps=0.0,
        capacity_spread_hard_bps=0.0,
        capacity_volume_soft_participation=0.0,
        capacity_volume_hard_participation=0.0,
        capacity_min_scale=1.0,
        slo_derisk_scale=1.0,
        slo_derisk_details={},
        execution_model_lineage={},
        exec_engine=None,
        effective_policy=SimpleNamespace(),
        edge_realism_rank_factor_by_symbol={},
        edge_realism_apply_to_approval_enabled=False,
        portfolio_current_gross=0.0,
        sector_gross={},
        max_new_orders_per_cycle=None,
        portfolio_optimizer_enabled=False,
        portfolio_optimizer=None,
        portfolio_optimizer_openings_only=False,
        portfolio_optimizer_market_data={},
        portfolio_optimizer_context={},
        ledger=None,
        rate_limiter=None,
        breakers=SimpleNamespace(),
        symbol_adaptive_profiles={},
        uncertainty_capital_state={},
        uncertainty_cycle_events=[],
        penalty_overlap_coordination_enabled=False,
        penalty_overlap_weight_dampen=0.0,
        penalty_overlap_min_scale_floor=0.0,
        ineffective_gate_blocklist=set(),
        gate_root_cause_func=lambda gate: gate,
        position_opened_at_func=lambda *a, **k: None,
        exit_policy_pressure_context_func=lambda *a, **k: {},
        is_near_event_func=lambda *a, **k: False,
        enforce_participation_cap_func=lambda **k: (True, k["order_qty"], None),
        alpha_decay_entry_guard_func=lambda *a, **k: {},
        safe_float=safe_float,
        resolve_uncertainty_capital_auto_controls_func=lambda **k: {},
        clip_delta_to_symbol_notional_cap_func=lambda **k: (int(k["delta_shares"]), None),
        clip_sell_qty_to_available_position_func=lambda **k: (int(k["requested_qty"]), None),
        percentile_linear_func=lambda values, p: None,
        slippage_setting_bps_func=lambda: 0.0,
        get_sector_func=lambda symbol: None,
        evaluate_execution_approval_func=lambda **k: None,
        approve_execution_candidate_func=lambda **k: None,
        gate_name_is_halt_noise_func=lambda gate: False,
        resolve_order_quote_basis_func=lambda *a, **k: ("nbbo", 9.9, 10.1, 10.0, 10.0, None),
        portfolio_optimizer_allows_trade_func=lambda **k: (True, {}),
        auth_forbidden_cooldown_remaining_seconds_func=lambda **k: 0.0,
        safe_validate_pretrade_func=lambda *a, **k: (True, "OK", {}),
        extract_order_value_func=lambda *a, **k: None,
        extract_order_fill_timestamp_func=lambda order: None,
        normalize_order_status_token_func=lambda status: "new",
        has_persistable_fill_func=lambda **k: False,
        normalize_submitted_order_func=lambda **k: SimpleNamespace(),
        record_successful_submission_func=lambda **k: None,
        build_order_metrics_and_tca_func=lambda **k: ({}, None),
        submit_order_func=lambda *a, **k: object(),
        classify_exception_func=lambda *a, **k: None,
        handle_error_func=lambda *a, **k: None,
        trigger_quarantine_func=lambda *a, **k: None,
        cancel_all_open_orders_oms_func=lambda runtime: None,
        resolve_submit_none_reason_func=lambda runtime: "SUBMIT_NONE",
        record_auth_forbidden_cooldown_func=lambda *a, **k: None,
        get_regime_signal_profile_func=lambda: "unknown",
        normalize_quote_source_token_func=lambda token: token,
        resolve_quote_proxy_source_func=lambda **k: None,
        resolved_tca_path_func=lambda: None,
        write_tca_record_func=lambda *a, **k: None,
        session_bucket_from_ts_func=lambda ts: "regular",
        compute_attribution_metrics_func=lambda **k: {},
        record_decision_func=record_decision_func,
        prepare_symbol_prelude_func=prepare_symbol_prelude_func,
        apply_symbol_adjustments_func=apply_symbol_adjustments_func,
        prepare_symbol_approval_func=prepare_symbol_approval_func,
        prepare_submit_prelude_func=prepare_submit_prelude_func,
        execute_submission_func=execute_submission_func,
    )
    for key, value in overrides.items():
        setattr(processor, key, value)
    return processor, records


def test_process_netting_symbol_halt_records_block() -> None:
    processor, records = _make_processor()
    processor.state.halt_trading = True
    processor.state.halt_reason = "HALT_TRADING"
    processor.prepare_symbol_prelude_func = lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not run"))

    result = process_netting_symbol(
        processor=processor,
        symbol="AAPL",
        net_target=_make_net_target(bar_ts=processor.now),
        orders_submitted=0,
    )

    assert result.attempted_increment == 0
    assert result.submitted_increment == 0
    assert len(records) == 1
    assert records[0]["gates"] == ["HALT_TRADING"]


def test_process_netting_symbol_suppresses_short_opening_in_long_only(monkeypatch) -> None:
    monkeypatch.setenv("TRADING__ALLOW_SHORTS", "0")
    processor, records = _make_processor(
        cfg=SimpleNamespace(seed="seed", launch_profile="paper_trade", shorts_allowed=False)
    )
    processor.prepare_symbol_prelude_func = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("short opening should be suppressed before prelude")
    )

    result = process_netting_symbol(
        processor=processor,
        symbol="AAPL",
        net_target=_make_net_target(bar_ts=processor.now, target_dollars=-100.0),
        orders_submitted=0,
    )

    assert result.attempted_increment == 0
    assert result.submitted_increment == 0
    assert len(records) == 1
    assert records[0]["net_target"].target_shares == 0
    assert records[0]["net_target"].target_dollars == 0.0
    assert "LONG_ONLY_SHORT_SUPPRESSED" in records[0]["gates"]


def test_process_netting_symbol_submitted_order_records_and_counts() -> None:
    seen_orders_submitted: list[int] = []

    def prepare_symbol_approval_func(**kwargs: Any) -> Any:
        seen_orders_submitted.append(int(kwargs["orders_submitted"]))
        return SimpleNamespace(
            delta_shares=int(kwargs["delta_shares"]),
            target_shares=int(kwargs["current_shares"] + kwargs["delta_shares"]),
            target_dollars=float((kwargs["current_shares"] + kwargs["delta_shares"]) * kwargs["price"]),
            side="buy",
            opening_trade=True,
            gates_added=(),
            snapshot_updates={},
            blocked_reason=None,
            blocked_metrics=None,
            approval=SimpleNamespace(expected_net_edge_bps=3.0),
            approval_context={"approved": True},
        )

    processor, records = _make_processor(prepare_symbol_approval_func=prepare_symbol_approval_func)

    result = process_netting_symbol(
        processor=processor,
        symbol="AAPL",
        net_target=_make_net_target(bar_ts=processor.now),
        orders_submitted=2,
    )

    assert seen_orders_submitted == [2]
    assert result.attempted_increment == 1
    assert result.submitted_increment == 1
    assert len(records) == 1
    assert records[0]["order"] == {"client_order_id": "cid-1"}
    assert records[0]["decision_trace_id"] == "trace-1"
    assert "OK_TRADE" in records[0]["gates"]
