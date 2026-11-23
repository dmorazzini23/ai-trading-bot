from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from ai_trading import rebalancer


@pytest.fixture
def portfolio_settings(monkeypatch) -> SimpleNamespace:
    settings = SimpleNamespace(
        ENABLE_PORTFOLIO_FEATURES=False,
        portfolio_drift_threshold=0.05,
    )
    monkeypatch.setattr(rebalancer, "get_settings", lambda: settings)
    return settings


def _tax_rebalancer(portfolio_settings) -> rebalancer.TaxAwareRebalancer:
    portfolio_settings.ENABLE_PORTFOLIO_FEATURES = False
    return rebalancer.TaxAwareRebalancer(tax_rate_short=0.37, tax_rate_long=0.20)






def test_lazy_optimizer_and_regime_detector_properties(monkeypatch, portfolio_settings) -> None:
    tax = _tax_rebalancer(portfolio_settings)
    optimizer = object()
    detector = object()
    monkeypatch.setattr(rebalancer, "create_portfolio_optimizer", lambda: optimizer)
    monkeypatch.setattr(rebalancer, "create_regime_detector", lambda: detector)

    assert tax.portfolio_optimizer is optimizer
    assert tax.portfolio_optimizer is optimizer
    assert tax.regime_detector is detector
    assert tax.regime_detector is detector


def test_tax_impact_missing_data_and_long_term_loss(portfolio_settings) -> None:
    tax = _tax_rebalancer(portfolio_settings)

    assert tax.calculate_tax_impact({"entry_price": 100.0}, 90.0) == {
        "error": "Missing position data"
    }

    impact = tax.calculate_tax_impact(
        {
            "entry_price": 100.0,
            "quantity": 10,
            "entry_date": datetime.now(UTC) - timedelta(days=400),
        },
        80.0,
    )

    assert impact["total_gain_loss"] == -200.0
    assert impact["is_long_term"] is True
    assert impact["tax_liability"] == 0
    assert impact["tax_efficiency_score"] == 1.0


def test_loss_harvesting_skips_wash_sale_and_sorts_by_priority(portfolio_settings) -> None:
    tax = _tax_rebalancer(portfolio_settings)
    old_entry = datetime.now(UTC) - timedelta(days=100)
    positions = {
        "LOW": {
            "entry_price": 100.0,
            "quantity": 10,
            "entry_date": old_entry,
        },
        "HIGH": {
            "entry_price": 100.0,
            "quantity": 100,
            "entry_date": old_entry,
        },
        "WASH": {
            "entry_price": 100.0,
            "quantity": 100,
            "entry_date": old_entry,
            "last_sale_date": datetime.now(UTC) - timedelta(days=3),
        },
    }

    opportunities = tax.identify_loss_harvesting_opportunities(
        positions,
        {"LOW": 90.0, "HIGH": 80.0, "WASH": 50.0},
    )

    assert [item["symbol"] for item in opportunities] == ["HIGH", "LOW"]
    assert opportunities[0]["tax_benefit"] > opportunities[1]["tax_benefit"]


def test_calculate_optimal_rebalance_flags_near_long_term_sale(portfolio_settings) -> None:
    tax = _tax_rebalancer(portfolio_settings)
    positions = {
        "AAPL": {
            "entry_price": 80.0,
            "quantity": 100,
            "entry_date": datetime.now(UTC) - timedelta(days=330),
        },
        "MSFT": {
            "entry_price": 50.0,
            "quantity": 20,
            "entry_date": datetime.now(UTC) - timedelta(days=40),
        },
        "BAD": {"entry_price": 10.0, "quantity": 5, "entry_date": datetime.now(UTC)},
    }

    plan = tax.calculate_optimal_rebalance(
        positions,
        {"AAPL": 0.10, "MSFT": 0.60, "GOOG": 0.30},
        {"AAPL": 100.0, "MSFT": 50.0, "GOOG": 25.0, "BAD": 0.0},
        account_equity=10_000.0,
    )

    trades = {trade["symbol"]: trade for trade in plan["rebalance_trades"]}
    assert "BAD" not in trades
    assert trades["AAPL"]["trade_quantity"] < 0
    assert trades["AAPL"]["tax_impact"]["is_optimal_timing"] is False
    assert trades["AAPL"]["tax_impact"]["delay_recommendation"] > 0
    assert trades["MSFT"]["trade_quantity"] > 0
    assert trades["GOOG"]["trade_quantity"] > 0
    assert plan["portfolio_drift"] > 0
    assert any("AAPL" in rec for rec in plan["recommendations"])


def test_calculate_optimal_rebalance_preserves_short_direction(portfolio_settings) -> None:
    tax = _tax_rebalancer(portfolio_settings)
    positions = {
        "SHORT_COVER": {
            "entry_price": 12.0,
            "quantity": -100,
            "entry_date": datetime.now(UTC) - timedelta(days=40),
        },
        "SHORT_MORE": {
            "entry_price": 12.0,
            "quantity": -100,
            "entry_date": datetime.now(UTC) - timedelta(days=40),
        },
    }

    plan = tax.calculate_optimal_rebalance(
        positions,
        {"SHORT_COVER": -0.25, "SHORT_MORE": -0.75},
        {"SHORT_COVER": 10.0, "SHORT_MORE": 10.0},
        account_equity=2_000.0,
    )

    trades = {trade["symbol"]: trade for trade in plan["rebalance_trades"]}
    assert plan["current_weights"]["SHORT_COVER"] == pytest.approx(-0.5)
    assert plan["current_weights"]["SHORT_MORE"] == pytest.approx(-0.5)
    assert trades["SHORT_COVER"]["trade_quantity"] == 50
    assert trades["SHORT_COVER"]["side"] == "buy_to_cover"
    assert trades["SHORT_MORE"]["trade_quantity"] == -50
    assert trades["SHORT_MORE"]["side"] == "sell_short"


def test_calculate_optimal_rebalance_weights_use_account_equity(portfolio_settings) -> None:
    tax = _tax_rebalancer(portfolio_settings)

    plan = tax.calculate_optimal_rebalance(
        {
            "LONG": {
                "entry_price": 10.0,
                "quantity": 10,
                "entry_date": datetime.now(UTC) - timedelta(days=40),
            },
            "SHORT": {
                "entry_price": 10.0,
                "quantity": -5,
                "entry_date": datetime.now(UTC) - timedelta(days=40),
            },
        },
        {"LONG": 0.10, "SHORT": -0.05},
        {"LONG": 10.0, "SHORT": 10.0},
        account_equity=1_000.0,
    )

    assert plan["current_weights"]["LONG"] == pytest.approx(0.10)
    assert plan["current_weights"]["SHORT"] == pytest.approx(-0.05)
    assert plan["rebalance_trades"] == []


def test_calculate_optimal_rebalance_accepts_qty_payloads(portfolio_settings) -> None:
    tax = _tax_rebalancer(portfolio_settings)

    plan = tax.calculate_optimal_rebalance(
        {
            "DICT_QTY": {
                "entry_price": 12.0,
                "qty": -100,
                "entry_date": datetime.now(UTC) - timedelta(days=40),
            },
            "OBJECT_QTY": SimpleNamespace(
                entry_price=12.0,
                qty=50,
                entry_date=datetime.now(UTC) - timedelta(days=40),
            ),
        },
        {"DICT_QTY": -0.25, "OBJECT_QTY": 0.25},
        {"DICT_QTY": 10.0, "OBJECT_QTY": 20.0},
        account_equity=2_000.0,
    )

    assert plan["current_weights"]["DICT_QTY"] == pytest.approx(-0.5)
    assert plan["current_weights"]["OBJECT_QTY"] == pytest.approx(0.5)


def test_calculate_optimal_rebalance_skips_missing_tax_basis_sells(portfolio_settings) -> None:
    tax = _tax_rebalancer(portfolio_settings)

    plan = tax.calculate_optimal_rebalance(
        {"AAPL": {"quantity": 100}},
        {"AAPL": 0.10},
        {"AAPL": 100.0},
        account_equity=10_000.0,
    )

    assert plan["rebalance_trades"] == []


def test_tax_efficiency_priority_and_recommendation_helpers(portfolio_settings) -> None:
    tax = _tax_rebalancer(portfolio_settings)

    assert tax._calculate_tax_efficiency(500.0, True) == pytest.approx(0.75)  # noqa: SLF001
    assert tax._calculate_rebalance_priority(0.20, {"tax_liability": 50, "is_optimal_timing": False}) == pytest.approx(-35.0)  # noqa: SLF001
    assert tax._calculate_portfolio_drift({"AAPL": 0.7}, {"AAPL": 0.5, "MSFT": 0.5}) == pytest.approx(0.35)  # noqa: SLF001
    assert tax._calculate_overall_tax_efficiency([]) == 1.0  # noqa: SLF001

    efficiency = tax._calculate_overall_tax_efficiency(  # noqa: SLF001
        [
            {"trade_value": 1000.0, "tax_impact": {"tax_liability": 0, "is_optimal_timing": True}},
            {"trade_value": 1000.0, "tax_impact": {"tax_liability": 100, "is_optimal_timing": False}},
        ]
    )
    assert efficiency == pytest.approx(0.8)

    recommendations = tax._generate_rebalance_recommendations(  # noqa: SLF001
        [
            {
                "symbol": "AAPL",
                "trade_quantity": -1,
                "trade_value": -100.0,
                "tax_impact": {
                    "is_optimal_timing": False,
                    "delay_recommendation": 12,
                    "tax_liability": 50.0,
                },
            },
            *[
                {
                    "symbol": f"S{i}",
                    "trade_quantity": -1,
                    "trade_value": -1000.0,
                    "tax_impact": {"tax_liability": 0.0},
                }
                for i in range(6)
            ],
        ]
    )
    assert any("delaying sale of AAPL" in rec for rec in recommendations)
    assert any("High tax impact for AAPL" in rec for rec in recommendations)
    assert any("spreading sales" in rec for rec in recommendations)


def test_rebalance_portfolio_attaches_tax_aware_plan(monkeypatch, portfolio_settings) -> None:
    portfolio_settings.ENABLE_PORTFOLIO_FEATURES = True
    expected_plan = {"portfolio_drift": 0.12, "total_tax_impact": 5.0, "rebalance_trades": []}

    class _TaxRebalancer:
        def calculate_optimal_rebalance(self, current_positions, target_weights, current_prices, account_equity):
            assert current_positions == {"AAPL": {"quantity": 5}}
            assert target_weights == {"AAPL": 0.5}
            assert current_prices == {"AAPL": 100.0}
            assert account_equity == 1000.0
            return expected_plan

    monkeypatch.setattr(rebalancer, "TaxAwareRebalancer", _TaxRebalancer)
    ctx = SimpleNamespace(
        account_equity=1000.0,
        current_positions={"AAPL": {"quantity": 5}},
        target_weights={"AAPL": 0.5},
        current_prices={"AAPL": 100.0},
    )

    rebalancer.rebalance_portfolio(ctx)

    assert ctx.rebalance_plan is expected_plan


















def test_maybe_rebalance_uses_drift_threshold(monkeypatch, portfolio_settings) -> None:
    rebalancer._last_rebalance = datetime.now(UTC) - timedelta(minutes=10)
    monkeypatch.setattr(rebalancer, "rebalance_interval_min", lambda: 1)
    monkeypatch.setattr(rebalancer, "compute_portfolio_weights", lambda _ctx, _symbols: {"AAPL": 0.60})
    calls: list[object] = []
    monkeypatch.setattr(rebalancer, "rebalance_portfolio", lambda ctx: calls.append(ctx))
    ctx = SimpleNamespace(portfolio_weights={"AAPL": 0.50})

    rebalancer.maybe_rebalance(ctx)

    assert calls == [ctx]
