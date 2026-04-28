from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai_trading.risk import pre_trade_validation as ptv


def test_liquidity_validation_uses_absolute_quantity_for_sell_orders() -> None:
    validator = ptv.LiquidityValidator()

    result = validator.validate_liquidity(
        "AAPL",
        -100_000,
        {
            "avg_volume": 1_000_000,
            "current_volume": 1_000_000,
            "bid_ask_spread": 0.01,
            "bid_size": 200_000,
            "ask_size": 200_000,
            "last_price": 100.0,
        },
    )

    assert result.details["trade_quantity"] == pytest.approx(100_000.0)
    assert result.details["participation_rate"] == pytest.approx(0.10)


def test_position_risk_projects_signed_flip_to_resulting_gross(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ptv,
        "safe_settings",
        lambda: SimpleNamespace(ENABLE_PORTFOLIO_FEATURES=False),
    )
    validator = ptv.RiskValidator()

    result = validator.validate_position_risk(
        "AAPL",
        quantity=400,
        price=100.0,
        account_equity=100_000.0,
        current_positions={"AAPL": {"notional_value": -20_000.0}},
    )

    assert result.status is ptv.ValidationStatus.WARNING
    assert result.details["order_value"] == pytest.approx(40_000.0)
    assert result.details["position_value"] == pytest.approx(20_000.0)
    assert result.details["position_percentage"] == pytest.approx(0.20)


def test_position_risk_uses_side_for_short_additions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ptv,
        "safe_settings",
        lambda: SimpleNamespace(ENABLE_PORTFOLIO_FEATURES=False),
    )
    validator = ptv.RiskValidator()

    result = validator.validate_position_risk(
        "AAPL",
        quantity=100,
        price=100.0,
        account_equity=100_000.0,
        current_positions={"AAPL": {"notional_value": -20_000.0}},
        side="sell_short",
    )

    assert result.details["order_value"] == pytest.approx(-10_000.0)
    assert result.details["position_value"] == pytest.approx(-30_000.0)
    assert result.details["position_percentage"] == pytest.approx(0.30)
    assert result.status is ptv.ValidationStatus.REJECTED


def test_correlation_exposure_uses_projected_gross_for_short_book(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ptv,
        "safe_settings",
        lambda: SimpleNamespace(ENABLE_PORTFOLIO_FEATURES=True),
    )
    validator = ptv.RiskValidator()

    exposure = validator._calculate_correlation_exposure(
        "AAPL",
        -10_000.0,
        {"MSFT": {"notional_value": -10_000.0}},
        {"AAPL_MSFT": 0.9},
    )

    assert exposure == pytest.approx(0.45)


def test_portfolio_risk_concentration_uses_projected_flip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ptv,
        "safe_settings",
        lambda: SimpleNamespace(ENABLE_PORTFOLIO_FEATURES=True),
    )
    validator = ptv.RiskValidator()

    result = validator.validate_portfolio_risk(
        {"symbol": "AAPL", "quantity": 400, "price": 100.0, "side": "buy"},
        {
            "account_equity": 100_000.0,
            "current_positions": {"AAPL": {"notional_value": -20_000.0}},
            "correlations": {},
        },
    )

    assert result.details["position_count"] == 1
    assert result.details["concentration_risk"] == pytest.approx(0.04)
