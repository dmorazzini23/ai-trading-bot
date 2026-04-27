from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from ai_trading import algorithm_optimizer as opt
from ai_trading.algorithm_optimizer import (
    AlgorithmOptimizer,
    MarketConditions,
    MarketRegime,
    OptimizedParameters,
    TradingPhase,
)


def _conditions(regime: MarketRegime, *, volatility: float = 0.2, phase: TradingPhase = TradingPhase.MID_DAY) -> MarketConditions:
    return MarketConditions(
        regime=regime,
        volatility=volatility,
        trend_strength=1.2,
        volume_profile=1.0,
        correlation_to_market=0.5,
        sector_rotation=0.0,
        vix_level=20.0,
        time_of_day=phase,
    )


def _price_frame(rows: int = 40, *, start: float = 100.0, stop: float = 140.0) -> pd.DataFrame:
    close = np.linspace(start, stop, rows)
    return pd.DataFrame(
        {
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1_000, 5_000, rows),
        }
    )


def test_market_regime_detection_defaults_and_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    optimizer = AlgorithmOptimizer()
    monkeypatch.setattr(optimizer, "_get_trading_phase", lambda: TradingPhase.MARKET_OPEN)

    short = optimizer.detect_market_regime(_price_frame(5))
    assert short.regime is MarketRegime.SIDEWAYS
    assert short.time_of_day is TradingPhase.MARKET_OPEN

    detected = optimizer.detect_market_regime(
        _price_frame(40, start=100, stop=180),
        volume_data=pd.DataFrame({"volume": np.linspace(1_000, 4_000, 40)}),
        market_data=_price_frame(40, start=50, stop=55),
    )

    assert detected.regime in {
        MarketRegime.TRENDING,
        MarketRegime.BULL_MARKET,
        MarketRegime.LOW_VOLATILITY,
        MarketRegime.SIDEWAYS,
    }
    assert optimizer.market_regimes[-1] is detected
    assert optimizer._classify_regime(0.4, 0.1, pd.Series([0.0])) is MarketRegime.VOLATILE  # noqa: SLF001
    assert optimizer._classify_regime(0.05, 0.1, pd.Series([0.0])) is MarketRegime.LOW_VOLATILITY  # noqa: SLF001
    assert optimizer._classify_regime(0.2, 2.0, pd.Series([0.02] * 6)) is MarketRegime.BULL_MARKET  # noqa: SLF001
    assert optimizer._classify_regime(0.2, 2.0, pd.Series([-0.02] * 6)) is MarketRegime.BEAR_MARKET  # noqa: SLF001
    assert optimizer._classify_regime(0.2, 2.0, pd.Series([0.0] * 6)) is MarketRegime.TRENDING  # noqa: SLF001


def test_trading_phase_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    optimizer = AlgorithmOptimizer()

    class FakeDateTime:
        @classmethod
        def now(cls, _tz):
            return SimpleNamespace(time=lambda: FakeDateTime.current)

    monkeypatch.setattr(opt, "datetime", FakeDateTime)
    cases = [
        (time(8, 0), TradingPhase.PRE_MARKET),
        (time(9, 45), TradingPhase.MARKET_OPEN),
        (time(14, 59), TradingPhase.MID_DAY),
        (time(15, 30), TradingPhase.LATE_DAY),
        (time(16, 1), TradingPhase.AFTER_HOURS),
    ]
    for current, expected in cases:
        FakeDateTime.current = current
        assert optimizer._get_trading_phase() is expected  # noqa: SLF001


def test_parameter_optimization_adjusts_regime_time_performance_and_bounds() -> None:
    optimizer = AlgorithmOptimizer()
    optimizer.last_optimization = datetime.now(UTC)
    unchanged = optimizer.optimize_parameters(_conditions(MarketRegime.SIDEWAYS), [], force_optimization=False)
    assert unchanged is optimizer.current_parameters

    optimized = optimizer.optimize_parameters(
        _conditions(MarketRegime.BEAR_MARKET, volatility=0.4, phase=TradingPhase.PRE_MARKET),
        [-0.03, -0.02, -0.01, -0.04, 0.01, -0.05],
        force_optimization=True,
    )

    assert optimized.position_size_multiplier < 1.0
    assert optimized.stop_loss_multiplier < 1.0
    assert optimized.rsi_overbought == 65.0
    assert optimized.volatility_lookback == 10
    assert optimizer.parameter_history

    low_vol = optimizer.optimize_parameters(
        _conditions(MarketRegime.LOW_VOLATILITY, volatility=0.05, phase=TradingPhase.LATE_DAY),
        [0.01] * 8 + [-0.01],
        force_optimization=True,
    )
    assert low_vol.volatility_lookback == 30
    assert low_vol.take_profit_multiplier > 1.0

    bounded = optimizer._enforce_bounds(  # noqa: SLF001
        OptimizedParameters(
            position_size_multiplier=99.0,
            stop_loss_multiplier=0.1,
            take_profit_multiplier=99.0,
            rsi_oversold=1.0,
            rsi_overbought=99.0,
            moving_average_period=1,
            volume_threshold=99.0,
            volatility_lookback=1,
            correlation_threshold=99.0,
        )
    )
    assert bounded.position_size_multiplier == 3.0
    assert bounded.stop_loss_multiplier == 0.5
    assert bounded.moving_average_period == 5


def test_position_sizing_kelly_stops_take_profit_and_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    optimizer = AlgorithmOptimizer()
    assert optimizer.calculate_optimal_position_size("AAPL", 0.0, 10_000.0, 0.2, _conditions(MarketRegime.SIDEWAYS)) == 0

    for value in [0.02, -0.01] * 6:
        optimizer.performance_history.append({"symbol": "AAPL", "return": value})
    shares = optimizer.calculate_optimal_position_size(
        "AAPL",
        price=100.0,
        account_value=100_000.0,
        volatility=0.2,
        market_conditions=_conditions(MarketRegime.BULL_MARKET),
    )
    assert 1 <= shares <= 100
    assert optimizer._calculate_kelly_fraction("AAPL") > 0.0  # noqa: SLF001
    assert optimizer._get_regime_multiplier(MarketRegime.VOLATILE) == 0.7  # noqa: SLF001
    assert optimizer._get_time_multiplier(TradingPhase.AFTER_HOURS) == 0.4  # noqa: SLF001

    buy_stop = optimizer.calculate_stop_loss(100.0, "BUY", 0.02, 2.0)
    sell_stop = optimizer.calculate_stop_loss(100.0, "SELL", 0.02, 2.0)
    assert buy_stop < 100.0
    assert sell_stop > 100.0
    assert optimizer.calculate_take_profit(100.0, "BUY", buy_stop) > 100.0
    assert optimizer.calculate_take_profit(100.0, "SELL", sell_stop) < 100.0

    validation = optimizer.validate_mathematical_models()
    assert validation["tests_failed"] == 0
    assert set(validation["tests_run"]) == {
        "position_sizing",
        "stop_loss_calculation",
        "take_profit_calculation",
        "parameter_optimization",
    }

    optimizer.regime_performance[MarketRegime.BULL_MARKET].extend([0.01, -0.02, 0.03])
    report = optimizer.get_optimization_report()
    assert report["parameter_changes"] >= 1
    assert report["regime_performance"]["bull_market"]["count"] == 3

    monkeypatch.setattr(opt, "_algorithm_optimizer", None)
    assert opt.get_algorithm_optimizer() is opt.get_algorithm_optimizer()
    assert opt.initialize_algorithm_optimizer() is opt.get_algorithm_optimizer()
