from __future__ import annotations

import logging
import threading
from types import SimpleNamespace
from typing import Any, cast

import pytest

np = pytest.importorskip("numpy")

from ai_trading.risk import engine as risk_engine


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
    engine.asset_limits = {}
    engine.strategy_limits = {}
    engine._invalid_min_size_logged = False
    engine._lock = threading.Lock()
    engine.current_trades = 0
    engine.max_trades = 2
    return cast(risk_engine.RiskEngine, engine)


def test_minimum_quantity_rejects_invalid_config_and_logs_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    engine = _bare_risk_engine(position_size_min_usd="bad")

    with caplog.at_level(logging.WARNING):
        first = risk_engine._derive_minimum_quantity(engine, price=100.0)
        second = risk_engine._derive_minimum_quantity(engine, price=100.0)

    assert first == 0
    assert second == 0
    assert sum("Invalid position_size_min_usd" in rec.message for rec in caplog.records) == 1


@pytest.mark.parametrize(
    ("raw_qty", "expected"),
    [
        (float("nan"), 0),
        (0.0, 0),
        (-5.0, 0),
        (7.9, 7),
    ],
)
def test_calculate_position_size_normalizes_edge_quantities(raw_qty: float, expected: int) -> None:
    engine = _bare_risk_engine(position_size_min_usd=200.0)
    signal = SimpleNamespace(symbol="AAPL")

    assert risk_engine._calculate_position_size(engine, raw_qty, 100.0, signal) == expected


def test_final_position_size_uses_buying_power_for_short_cash_cap() -> None:
    engine = _bare_risk_engine(max_symbol_exposure=1.0)
    signal = SimpleNamespace(symbol="MSFT", side="sell_short", asset_class="equity", strategy="momentum")

    capped = risk_engine._cap_final_position_size(
        engine,
        qty=20,
        price=100.0,
        signal=signal,
        cash=50.0,
        total_equity=10_000.0,
        buying_power=1_500.0,
    )

    assert capped == 15


def test_final_position_size_skips_cash_cap_for_short_without_buying_power() -> None:
    engine = _bare_risk_engine(max_symbol_exposure=1.0)
    signal = SimpleNamespace(symbol="MSFT", side="sell_short", asset_class="equity", strategy="momentum")

    capped = risk_engine._cap_final_position_size(
        engine,
        qty=20,
        price=100.0,
        signal=signal,
        cash=50.0,
        total_equity=10_000.0,
        buying_power=None,
    )

    assert capped == 20


def test_final_position_size_enforces_symbol_exposure_cap() -> None:
    engine = _bare_risk_engine(max_symbol_exposure=0.10)
    engine.symbol_exposure = {"AAPL": 0.08}
    signal = SimpleNamespace(symbol="AAPL", side="buy", asset_class="equity", strategy="momentum")

    capped = risk_engine._cap_final_position_size(
        engine,
        qty=10,
        price=100.0,
        signal=signal,
        cash=10_000.0,
        total_equity=10_000.0,
    )

    assert capped == 2


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


def test_apply_weight_limits_does_not_round_weak_weight_upward() -> None:
    engine = _bare_risk_engine()
    engine.global_limit = 0.50
    signal = SimpleNamespace(
        symbol="AAPL",
        asset_class="equity",
        strategy="momentum",
        weight=0.04,
        confidence=0.9,
    )

    assert engine._apply_weight_limits(signal) == pytest.approx(0.04)


def test_compute_volatility_returns_stable_metrics_for_finite_returns() -> None:
    engine = _bare_risk_engine()

    result = engine.compute_volatility(np.array([0.01, -0.02, 0.03, -0.01]))

    assert result["volatility"] > 0
    assert result["std_vol"] == pytest.approx(result["volatility"])
    assert result["mad_scaled"] == pytest.approx(result["mad"] * 1.4826)
    assert result["garch_vol"] >= 0














