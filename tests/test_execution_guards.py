from __future__ import annotations

from types import SimpleNamespace

from ai_trading.core import execution_guards
from ai_trading.policy.compiler import ExecutionApproval


class _Proposal:
    def __init__(self, confidence: float, target_dollars: float) -> None:
        self.confidence = confidence
        self.target_dollars = target_dollars


def test_evaluate_execution_approval_builds_candidate_and_context(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_approve(_policy, candidate):
        captured["candidate"] = candidate
        return ExecutionApproval(
            allowed=True,
            adjusted_delta_shares=7,
            expected_net_edge_bps=12.5,
            reasons=("OK_EDGE",),
        )

    result = execution_guards.evaluate_execution_approval(
        effective_policy=object(),
        symbol="AAPL",
        side="buy",
        delta_shares=5,
        current_shares=2.0,
        price=100.0,
        expected_edge_total=15.0,
        expected_cost_total=4.0,
        proposals=[_Proposal(0.8, 1200.0), _Proposal(0.6, 800.0)],
        spread_bps=10.0,
        rolling_volume=25000.0,
        pending_oldest_age_sec=12.0,
        calibration_ok=True,
        reject_rate_pct=1.5,
        portfolio_current_gross=5000.0,
        sector_gross={"TECH": 2000.0},
        sector_name="TECH",
        max_new_orders_per_cycle=3,
        orders_submitted=1,
        engine_cycle_new_orders_submitted=2,
        safety_tier_raw="normal",
        approval_func=_fake_approve,
    )

    candidate = captured["candidate"]
    assert getattr(candidate, "symbol") == "AAPL"
    assert getattr(candidate, "pacing_headroom") == 1
    assert getattr(candidate, "confidence") == 0.8
    assert result.adjusted_delta_shares == 7
    assert result.adjusted_side == "buy"
    assert result.stale_orders_present is True
    assert result.sector_name == "TECH"


def test_build_portfolio_optimizer_positions_filters_invalid_values() -> None:
    positions = {
        "AAPL": 1,
        "MSFT": "3.5",
        "BAD": "x",
        "NAN": float("nan"),
    }

    result = execution_guards.build_portfolio_optimizer_positions(
        positions,
        symbol="AAPL",
        current_shares=9.0,
    )

    assert result == {"AAPL": 9.0, "MSFT": 3.5}


def test_build_pretrade_validation_cfg_for_thin_liquidity(monkeypatch) -> None:
    monkeypatch.setenv("AI_TRADING_LIQ_THIN_COLLAR_MULT", "0.5")
    cfg = SimpleNamespace(
        max_order_dollars=10000.0,
        max_order_shares=50,
        price_collar_pct=0.04,
    )

    pretrade_cfg, effective_collar_pct, collar_mult = execution_guards.build_pretrade_validation_cfg(
        cfg,
        thin_liquidity=True,
    )

    assert collar_mult == 0.5
    assert effective_collar_pct == 0.02
    assert getattr(pretrade_cfg, "price_collar_pct") == 0.02


def test_build_pretrade_validation_cfg_preserves_non_collar_limits(monkeypatch) -> None:
    monkeypatch.setenv("AI_TRADING_LIQ_THIN_COLLAR_MULT", "0.5")
    cfg = SimpleNamespace(
        max_order_dollars=10000.0,
        max_order_shares=50,
        max_symbol_notional=25000.0,
        quote_max_age_ms=800,
        rth_only=True,
        allow_extended=False,
        price_collar_pct=0.04,
    )

    pretrade_cfg, effective_collar_pct, _collar_mult = execution_guards.build_pretrade_validation_cfg(
        cfg,
        thin_liquidity=True,
    )

    assert effective_collar_pct == 0.02
    assert getattr(pretrade_cfg, "max_symbol_notional") == 25000.0
    assert getattr(pretrade_cfg, "quote_max_age_ms") == 800
    assert getattr(pretrade_cfg, "rth_only") is True
