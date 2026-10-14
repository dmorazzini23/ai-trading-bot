from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

from ai_trading.core.netting_submit_prelude import prepare_netting_submit_prelude
from ai_trading.risk.liquidity_regime import LiquidityFeatures, LiquidityRegime


def _base_kwargs() -> dict[str, Any]:
    return {
        "state": SimpleNamespace(halt_trading=False, halt_reason=""),
        "runtime": SimpleNamespace(),
        "cfg": SimpleNamespace(
            seed="seed",
            execution_require_realtime_nbbo=True,
            execution_mode="paper",
            paper_sampling_enabled=True,
            paper_sampling_passive_only=True,
            paper_sampling_allowed_symbols=("AAPL", "AMZN", "MSFT"),
        ),
        "now": datetime(2026, 4, 19, 15, 0, tzinfo=UTC),
        "symbol": "AAPL",
        "side": "buy",
        "price": 100.0,
        "delta_shares": 10,
        "current_shares": 0.0,
        "bar_ts": datetime(2026, 4, 19, 14, 59, tzinfo=UTC),
        "liq_features": LiquidityFeatures(rolling_volume=1000.0, spread_bps=5.0, volatility_proxy=1.0),
        "liq_regime": LiquidityRegime.NORMAL,
        "net_target": SimpleNamespace(proposals=[]),
        "slo_derisk_details": {"rolling_volume": 1000.0},
        "symbol_snapshot": {"effective_policy_hash": "policy"},
        "execution_model_lineage": {"model_id": "m1"},
        "correlation_id": "opp_aapl_canonical",
        "event_risk_near": False,
        "opening_trade": True,
        "portfolio_optimizer_enabled": False,
        "portfolio_optimizer": None,
        "portfolio_optimizer_openings_only": False,
        "positions": {},
        "portfolio_optimizer_market_data": {},
        "portfolio_optimizer_context": {"enabled": False},
        "ledger": None,
        "rate_limiter": object(),
        "breakers": SimpleNamespace(
            allow=lambda dep: True,
            open_reason=lambda dep: None,
        ),
        "kill_switch_active": False,
        "gate_name_is_halt_noise_func": lambda gate: False,
        "resolve_order_quote_basis_func": lambda runtime, symbol, side, fallback_price: (
            "nbbo",
            99.5,
            100.5,
            100.0,
            100.0,
            None,
        ),
        "portfolio_optimizer_allows_trade_func": lambda **kwargs: (True, {"decision": "allow"}),
        "auth_forbidden_cooldown_remaining_seconds_func": lambda *args, **kwargs: 0.0,
        "safe_validate_pretrade_func": lambda intent, **kwargs: (True, "OK", {}),
        "get_sector_func": lambda symbol: "TECH",
    }


def test_prepare_netting_submit_prelude_blocks_on_portfolio_optimizer() -> None:
    kwargs = _base_kwargs()
    kwargs["portfolio_optimizer_enabled"] = True
    kwargs["portfolio_optimizer"] = object()
    kwargs["portfolio_optimizer_allows_trade_func"] = lambda **kwargs: (
        False,
        {"decision": "reject", "why": "cluster_cap"},
    )

    result = prepare_netting_submit_prelude(**cast(Any, kwargs))

    assert result.blocked_reason == "PORTFOLIO_OPTIMIZER_REJECT"
    assert result.blocked_metrics == {"portfolio_optimizer": {"decision": "reject", "why": "cluster_cap"}}
    assert result.snapshot_updates["portfolio_optimizer"]["why"] == "cluster_cap"


def test_prepare_netting_submit_prelude_blocks_on_optimizer_init_failure() -> None:
    kwargs = _base_kwargs()
    kwargs["portfolio_optimizer_enabled"] = True
    kwargs["portfolio_optimizer"] = None
    kwargs["portfolio_optimizer_context"] = {
        "enabled": True,
        "active": False,
        "init_failed": True,
        "init_fail_open": False,
        "error_type": "RuntimeError",
    }

    result = prepare_netting_submit_prelude(**cast(Any, kwargs))

    assert result.blocked_reason == "PORTFOLIO_OPTIMIZER_INIT_FAILED"
    assert result.blocked_metrics == {
        "portfolio_optimizer": kwargs["portfolio_optimizer_context"]
    }
    assert result.snapshot_updates["portfolio_optimizer"]["init_failed"] is True


def test_prepare_netting_submit_prelude_allows_explicit_optimizer_init_fail_open() -> None:
    kwargs = _base_kwargs()
    kwargs["portfolio_optimizer_enabled"] = True
    kwargs["portfolio_optimizer"] = None
    kwargs["portfolio_optimizer_context"] = {
        "enabled": False,
        "active": False,
        "init_failed": True,
        "init_fail_open": True,
        "fail_open_applied": True,
    }

    result = prepare_netting_submit_prelude(**cast(Any, kwargs))

    assert result.blocked_reason is None
    assert result.execution_intent_context is not None
    assert result.snapshot_updates["portfolio_optimizer"]["fail_open_applied"] is True


def test_prepare_netting_submit_prelude_blocks_opening_optimizer_decision_error() -> None:
    kwargs = _base_kwargs()
    kwargs["portfolio_optimizer_enabled"] = True
    kwargs["portfolio_optimizer"] = object()
    kwargs["portfolio_optimizer_allows_trade_func"] = lambda **_kwargs: (_ for _ in ()).throw(
        RuntimeError("optimizer down")
    )

    result = prepare_netting_submit_prelude(**cast(Any, kwargs))

    assert result.blocked_reason == "PORTFOLIO_OPTIMIZER_DECISION_ERROR"
    assert result.blocked_metrics is not None
    assert result.blocked_metrics["portfolio_optimizer"]["fail_closed"] is True
    assert result.snapshot_updates["portfolio_optimizer"]["error_type"] == "RuntimeError"


def test_prepare_netting_submit_prelude_blocks_live_optimizer_decision_error() -> None:
    kwargs = _base_kwargs()
    kwargs["opening_trade"] = False
    kwargs["cfg"] = SimpleNamespace(
        seed="seed",
        execution_require_realtime_nbbo=True,
        execution_mode="live",
    )
    kwargs["portfolio_optimizer_enabled"] = True
    kwargs["portfolio_optimizer"] = object()
    kwargs["portfolio_optimizer_allows_trade_func"] = lambda **_kwargs: (_ for _ in ()).throw(
        ValueError("optimizer bad input")
    )

    result = prepare_netting_submit_prelude(**cast(Any, kwargs))

    assert result.blocked_reason == "PORTFOLIO_OPTIMIZER_DECISION_ERROR"
    assert result.blocked_metrics is not None
    assert result.blocked_metrics["portfolio_optimizer"]["fail_closed"] is True


def test_prepare_netting_submit_prelude_blocks_on_pretrade_and_exposes_order_intent() -> None:
    kwargs = _base_kwargs()
    kwargs["safe_validate_pretrade_func"] = lambda intent, **kwargs: (
        False,
        "PRICE_COLLAR_BLOCK",
        {"reason": "collar"},
    )

    result = prepare_netting_submit_prelude(**cast(Any, kwargs))

    assert result.blocked_reason == "PRICE_COLLAR_BLOCK"
    assert result.blocked_metrics == {"pretrade": {"reason": "collar"}}
    assert result.blocked_order_intent is not None
    assert result.blocked_order_intent.symbol == "AAPL"
    assert result.blocked_order_intent.correlation_id == "opp_aapl_canonical"
    assert result.blocked_order_intent.metadata["order_type"] == "limit"
    assert (
        result.blocked_order_intent.metadata["execution_profile"]
        == "paper_sampling_passive"
    )


def test_prepare_netting_submit_prelude_builds_final_intent_with_nbbo_quote() -> None:
    kwargs = _base_kwargs()

    result = prepare_netting_submit_prelude(**cast(Any, kwargs))

    assert result.blocked_reason is None
    assert result.execution_intent_context is not None
    assert result.submit_quote_source == "nbbo"
    assert result.execution_intent_context.pretrade_intent.client_order_id
    assert result.execution_intent_context.correlation_id == "opp_aapl_canonical"
    assert result.execution_intent_context.order_lineage_metadata["order_type"] == "limit"
    assert (
        result.execution_intent_context.order_lineage_metadata["execution_profile"]
        == "paper_sampling_passive"
    )
    assert result.execution_intent_context.order_annotations["quote"]["midpoint"] == 100.0


def test_prelude_decision_time_follows_arrival_quote_not_cycle_start() -> None:
    kwargs = _base_kwargs()
    cycle_now = datetime.now(UTC) - timedelta(seconds=8)
    arrival_quote_ts = datetime.now(UTC) - timedelta(milliseconds=200)
    kwargs["now"] = cycle_now
    kwargs["resolve_order_quote_basis_func"] = lambda runtime, symbol, side, fallback_price: (
        "broker_nbbo", 99.5, 100.5, 100.0, 100.5, arrival_quote_ts,
    )

    result = prepare_netting_submit_prelude(**cast(Any, kwargs))

    assert result.execution_intent_context is not None
    lineage = result.execution_intent_context.order_lineage_metadata
    decision_ts = datetime.fromisoformat(lineage["decision_ts"])
    # Causal: the decision is made after the quote it uses was observed.
    assert 0.0 <= (decision_ts - arrival_quote_ts).total_seconds() <= 1.0
    assert decision_ts > cycle_now
    evidence = result.decision_quote_evidence
    assert evidence is not None
    assert evidence["quote_observed_at"] == lineage["decision_ts"]
    assert evidence["quote_timestamp"] == arrival_quote_ts.isoformat()
    assert evidence["raw_quote_bid"] == 99.5
    assert evidence["raw_quote_ask"] == 100.5
    assert evidence["quote_source"] == "broker_nbbo"
    assert 200.0 <= evidence["quote_age_ms"] < 1000.0
    # The pretrade gate input is intentionally unchanged (still cycle-relative).
    assert result.execution_intent_context.pretrade_intent.quote_age_ms == 0.0


def test_prelude_pretrade_block_carries_arrival_quote_evidence() -> None:
    kwargs = _base_kwargs()
    kwargs["safe_validate_pretrade_func"] = lambda intent, **kwargs: (False, "PRICE_COLLAR_BLOCK", {})

    result = prepare_netting_submit_prelude(**cast(Any, kwargs))

    assert result.blocked_reason == "PRICE_COLLAR_BLOCK"
    assert result.decision_quote_evidence is not None
    assert result.decision_quote_evidence["raw_quote_bid"] == 99.5
    assert "quote_timestamp" not in result.decision_quote_evidence

