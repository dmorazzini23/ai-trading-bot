from __future__ import annotations

import logging
import threading
from datetime import datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest

np = pytest.importorskip("numpy")

from ai_trading.risk import engine as risk_engine
from ai_trading.risk import pre_trade_validation as ptv


def _bare_risk_engine(**config_updates: Any) -> risk_engine.RiskEngine:
    engine = object.__new__(risk_engine.RiskEngine)
    config = {
        "position_size_min_usd": 250.0,
        "max_symbol_exposure": 0.10,
        "min_order_value": 100.0,
        "max_order_value": 1_000.0,
    }
    config.update(config_updates)
    engine.config = SimpleNamespace(**config)
    engine.global_limit = 0.50
    engine.exposure = {}
    engine.strategy_exposure = {}
    engine._returns = []
    engine._drawdowns = []
    engine.asset_limits = {}
    engine.strategy_limits = {}
    engine._invalid_min_size_logged = False
    engine._lock = threading.Lock()
    engine.current_trades = 0
    engine.max_trades = 2
    return cast(risk_engine.RiskEngine, engine)


def test_minimum_quantity_uses_config_then_fallback_and_logs_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    engine = _bare_risk_engine(position_size_min_usd="bad")
    monkeypatch.setattr(risk_engine, "get_position_size_min_usd", lambda: 300.0)

    with caplog.at_level(logging.WARNING):
        first = risk_engine._derive_minimum_quantity(engine, price=100.0)
        second = risk_engine._derive_minimum_quantity(engine, price=100.0)

    assert first == 3
    assert second == 3
    assert sum("Invalid position_size_min_usd" in rec.message for rec in caplog.records) == 1


@pytest.mark.parametrize(
    ("raw_qty", "expected"),
    [
        (float("nan"), 2),
        (0.0, 2),
        (-5.0, 0),
        (7.9, 7),
    ],
)
def test_calculate_position_size_normalizes_edge_quantities(raw_qty: float, expected: int) -> None:
    engine = _bare_risk_engine(position_size_min_usd=200.0)
    signal = SimpleNamespace(symbol="AAPL")

    assert risk_engine._calculate_position_size(engine, raw_qty, 100.0, signal) == expected


def test_apply_weight_limits_clamps_to_most_constrained_capacity() -> None:
    engine = _bare_risk_engine()
    engine.global_limit = 0.50
    engine.asset_limits = {"equity": 0.30}
    engine.strategy_limits = {"momentum": 0.25}
    engine.exposure = {"equity": 0.20}
    engine.strategy_exposure = {"momentum": 0.10}
    signal = SimpleNamespace(
        symbol="AAPL",
        asset_class="equity",
        strategy="momentum",
        weight=0.80,
        confidence=0.9,
    )

    assert engine._apply_weight_limits(signal) == pytest.approx(0.1)


def test_compute_volatility_returns_stable_metrics_for_finite_returns() -> None:
    engine = _bare_risk_engine()

    result = engine.compute_volatility(np.array([0.01, -0.02, 0.03, -0.01]))

    assert result["volatility"] > 0
    assert result["std_vol"] == pytest.approx(result["volatility"])
    assert result["mad_scaled"] == pytest.approx(result["mad"] * 1.4826)
    assert result["garch_vol"] >= 0


def test_dynamic_stop_price_picks_tightest_stop_by_direction() -> None:
    assert risk_engine.dynamic_stop_price(
        100.0,
        atr=2.0,
        percent=0.04,
        direction="long",
    ) == pytest.approx(97.0)
    assert risk_engine.dynamic_stop_price(
        100.0,
        atr=2.0,
        percent=0.04,
        direction="short",
    ) == pytest.approx(103.0)


def test_market_hours_validator_static_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ptv, "load_pandas_market_calendars", lambda: None)
    validator = ptv.MarketHoursValidator()

    open_result = validator.validate_market_hours(datetime(2026, 4, 21, 15, 0, tzinfo=ptv.UTC))
    extended_result = validator.validate_market_hours(datetime(2026, 4, 21, 13, 0, tzinfo=ptv.UTC))
    closed_result = validator.validate_market_hours(datetime(2026, 4, 21, 2, 0, tzinfo=ptv.UTC))

    assert open_result.status is ptv.ValidationStatus.APPROVED
    assert extended_result.status is ptv.ValidationStatus.WARNING
    assert closed_result.status is ptv.ValidationStatus.REJECTED


def test_liquidity_validator_error_and_low_liquidity_paths() -> None:
    validator = ptv.LiquidityValidator()

    bad = validator.validate_liquidity("AAPL", "not-a-number", {})
    thin = validator.validate_liquidity(
        "AAPL",
        100,
        {"avg_volume": 10, "current_volume": 10, "last_price": 100.0},
    )

    assert bad.status is ptv.ValidationStatus.REJECTED
    assert "error" in bad.details
    assert thin.status is ptv.ValidationStatus.REJECTED
    assert thin.score == 0.0


def test_portfolio_risk_warns_on_high_correlation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ptv,
        "safe_settings",
        lambda: SimpleNamespace(ENABLE_PORTFOLIO_FEATURES=True),
    )
    validator = ptv.RiskValidator()

    result = validator.validate_portfolio_risk(
        {"symbol": "AAPL", "quantity": 100, "price": 100.0},
        {
            "account_equity": 100_000.0,
            "current_positions": {"MSFT": {"notional_value": 10_000.0}},
            "correlations": {"AAPL_MSFT": 0.90},
        },
    )

    assert result.status is ptv.ValidationStatus.WARNING
    assert result.category is ptv.ValidationCategory.CORRELATION
    assert result.details["correlation_exposure"] == pytest.approx(0.45)


def test_pretrade_validator_overall_result_status_matrix() -> None:
    validator = object.__new__(ptv.PreTradeValidator)
    validator.min_overall_score = 0.60
    validator.warning_threshold = 0.80

    approved = ptv.ValidationResult(
        ptv.ValidationCategory.SYSTEM_HEALTH,
        ptv.ValidationStatus.APPROVED,
        "ok",
        {},
        0.95,
        [],
    )
    warning = ptv.ValidationResult(
        ptv.ValidationCategory.LIQUIDITY,
        ptv.ValidationStatus.WARNING,
        "watch",
        {},
        0.75,
        [],
    )
    rejected = ptv.ValidationResult(
        ptv.ValidationCategory.RISK_LIMITS,
        ptv.ValidationStatus.REJECTED,
        "no",
        {},
        1.0,
        [],
    )

    assert validator._calculate_overall_result([]) == (ptv.ValidationStatus.REJECTED, 0.0)
    assert validator._calculate_overall_result([approved]) == (
        ptv.ValidationStatus.APPROVED,
        pytest.approx(0.95),
    )
    assert validator._calculate_overall_result([approved, warning])[0] is ptv.ValidationStatus.WARNING
    assert validator._calculate_overall_result([approved, rejected]) == (
        ptv.ValidationStatus.REJECTED,
        0.0,
    )
