from __future__ import annotations

import importlib
import logging
import random
import types
from typing import Any, cast

import pytest

np = pytest.importorskip("numpy")

from ai_trading.risk.engine import RiskEngine, TradeSignal


def _signal(**updates: Any) -> Any:
    base = {
        "symbol": "AAPL",
        "side": "buy",
        "confidence": 1.0,
        "strategy": "runtime",
        "weight": 0.10,
        "asset_class": "equity",
    }
    base.update(updates)
    return TradeSignal(**base)


def _engine() -> RiskEngine:
    engine = RiskEngine()
    engine.config = cast(
        Any,
        types.SimpleNamespace(
            atr_multiplier=1.0,
            position_size_min_usd=100.0,
            volatility_lookback_days=10,
            max_concurrent_orders=3,
        ),
    )
    engine.asset_limits["equity"] = 1.0
    engine.strategy_limits["runtime"] = 1.0
    return engine


def test_current_volatility_fallback_logs_once(caplog: pytest.LogCaptureFixture) -> None:
    engine = _engine()

    with caplog.at_level(logging.INFO):
        first = engine._current_volatility()
        second = engine._current_volatility()

    assert first == second
    assert first > 0
    assert sum(1 for rec in caplog.records if rec.message == "PORTFOLIO_VOLATILITY_FALLBACK") == 1


def test_adaptive_global_cap_scales_up_and_down() -> None:
    engine = _engine()
    engine.global_limit = 0.5
    engine._returns = [0.02, 0.018, 0.021, 0.019]
    bullish_cap = engine._adaptive_global_cap()

    engine._returns = [-0.20, -0.05, -0.04, -0.03]
    defensive_cap = engine._adaptive_global_cap()

    assert bullish_cap > engine.global_limit
    assert defensive_cap < engine.global_limit


def test_available_exposure_clamps_invalid_cap() -> None:
    engine = _engine()
    engine.global_limit = 0.8
    engine.exposure = {"equity": 0.25}

    assert engine.available_exposure(cap=cast(Any, "bad")) == pytest.approx(0.55)


def test_refresh_positions_rebuilds_exposure_from_broker_positions() -> None:
    engine = _engine()

    class _Api:
        @staticmethod
        def get_account() -> types.SimpleNamespace:
            return types.SimpleNamespace(equity="10000")

        @staticmethod
        def list_positions() -> list[types.SimpleNamespace]:
            return [
                types.SimpleNamespace(asset_class="equity", qty="10", avg_entry_price="100", current_price="150"),
                types.SimpleNamespace(asset_class="crypto", qty="2", avg_entry_price="500", market_value="1200"),
            ]

    engine.refresh_positions(_Api())

    assert engine.exposure["equity"] == pytest.approx(0.15)
    assert engine.exposure["crypto"] == pytest.approx(0.12)


def test_position_exists_handles_match_and_missing_api() -> None:
    engine = _engine()
    api = types.SimpleNamespace(
        list_positions=lambda: [
            types.SimpleNamespace(symbol="MSFT"),
            types.SimpleNamespace(symbol="AAPL"),
        ],
    )

    assert engine.position_exists(api, "AAPL") is True
    assert engine.position_exists(object(), "AAPL") is False


def test_can_trade_rejects_invalid_signal_and_bad_weight(caplog: pytest.LogCaptureFixture) -> None:
    engine = _engine()
    bad_weight = _signal(weight="not-a-number")

    with caplog.at_level(logging.WARNING):
        assert engine.can_trade(object()) is False
        assert engine.can_trade(bad_weight) is True

    assert any("invalid signal type" in rec.message for rec in caplog.records)
    assert any("Invalid signal.weight" in rec.message for rec in caplog.records)


def test_can_trade_force_continue_does_not_override_exposure_breach(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    engine = _engine()
    engine.global_limit = 0.1
    engine.asset_limits["equity"] = 0.1
    engine.exposure["equity"] = 0.2
    monkeypatch.setenv("FORCE_CONTINUE_ON_EXPOSURE", "true")

    with caplog.at_level(logging.WARNING):
        assert engine.can_trade(_signal(weight=0.2)) is False

    assert any("FORCE_CONTINUE_ON_EXPOSURE ignored" in rec.message for rec in caplog.records)


def test_can_trade_rejects_negative_weight_short_exposure_breach() -> None:
    engine = _engine()
    engine.asset_limits["equity"] = 0.5
    engine.strategy_limits["runtime"] = 0.5
    engine.exposure["equity"] = 0.4

    assert engine.can_trade(_signal(side="sell_short", weight=-0.2)) is False
    assert engine.can_trade(_signal(side="short", weight=-0.05)) is True


def test_can_trade_projects_strategy_exposure_with_pending_weight() -> None:
    engine = _engine()
    engine.asset_limits["equity"] = 1.0
    engine.strategy_limits["runtime"] = 0.20
    engine.strategy_exposure["runtime"] = 0.12

    assert engine.can_trade(_signal(weight=0.05), pending=0.02) is True
    assert engine.can_trade(_signal(weight=0.05), pending=0.04) is False


def test_risk_engine_import_does_not_reseed_global_rngs() -> None:
    import ai_trading.risk.engine as risk_engine_module

    random.seed(987654)
    np.random.seed(987654)
    expected_py = random.random()
    expected_np = np.random.random()

    random.seed(987654)
    np.random.seed(987654)
    importlib.reload(risk_engine_module)

    assert random.random() == expected_py
    assert np.random.random() == expected_np


def test_register_fill_invalid_and_negative_sell_paths(caplog: pytest.LogCaptureFixture) -> None:
    engine = _engine()
    sell = _signal(side="sell", weight=0.20)

    with caplog.at_level(logging.WARNING):
        engine.register_fill(object())
        engine.register_fill(sell)

    assert engine.exposure["equity"] == 0.0
    assert any("invalid signal type" in rec.message for rec in caplog.records)
    assert any(rec.message == "EXPOSURE_NEGATIVE_PREVENTED" for rec in caplog.records)


def test_trade_slot_acquire_and_release_limits() -> None:
    engine = _engine()
    engine.max_trades = 2

    assert engine.acquire_trade_slot() is True
    assert engine.acquire_trade_slot() is True
    assert engine.acquire_trade_slot() is False
    engine.release_trade_slot()
    assert engine.current_trades == 1
    engine.release_trade_slot()
    engine.release_trade_slot()
    assert engine.current_trades == 0


def test_update_exposure_requires_context_and_swallows_refresh_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    engine = _engine()

    with pytest.raises(RuntimeError):
        engine.update_exposure()

    setattr(
        cast(Any, engine),
        "refresh_positions",
        lambda _api: (_ for _ in ()).throw(ValueError("bad positions")),
    )
    with caplog.at_level(logging.WARNING):
        engine.update_exposure(types.SimpleNamespace(api=object()))

    assert any("Failed to update exposure" in rec.message for rec in caplog.records)


def test_hard_stop_waits_for_recovered_metrics_after_cooldown() -> None:
    engine = _engine()
    engine.hard_stop = True
    engine._hard_stop_until = 0.0

    engine._maybe_lift_hard_stop()

    assert engine.hard_stop is True

    engine._check_drawdown_and_update_stop(0.01)

    assert engine.hard_stop is False
    assert engine._hard_stop_until is None


@pytest.mark.parametrize(
    ("cash", "price"),
    [
        (0.0, 100.0),
        (1_000.0, 0.0),
    ],
)
def test_position_size_rejects_invalid_cash_or_price(cash: float, price: float) -> None:
    engine = _engine()

    assert engine.position_size(_signal(), cash=cash, price=price) == 0


def test_position_size_falls_back_when_account_equity_invalid(caplog: pytest.LogCaptureFixture) -> None:
    engine = _engine()
    setattr(cast(Any, engine), "_get_atr_data", lambda _symbol: None)
    setattr(cast(Any, engine), "check_max_drawdown", lambda _api: True)
    api = types.SimpleNamespace(get_account=lambda: types.SimpleNamespace(equity="bad"))

    with caplog.at_level(logging.WARNING):
        qty = engine.position_size(_signal(weight=0.2), cash=1_000.0, price=50.0, api=api)

    assert qty > 0
    assert any("Error getting account equity" in rec.message for rec in caplog.records)


def test_compute_volatility_rejects_nan_values(caplog: pytest.LogCaptureFixture) -> None:
    engine = _engine()

    with caplog.at_level(logging.ERROR):
        result = engine.compute_volatility(np.array([0.01, np.nan, 0.02]))

    assert result == {"volatility": 0.0, "mad": 0.0, "garch_vol": 0.0}
    assert any("invalid values" in rec.message for rec in caplog.records)
