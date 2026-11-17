from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from ai_trading import capital_scaling


@pytest.fixture(autouse=True)
def _reload_capital_scaling() -> None:
    importlib.reload(capital_scaling)


def test_capital_scaling_engine_updates_baseline_and_current_scale() -> None:
    engine = capital_scaling.CapitalScalingEngine({}, initial_equity=100_000.0)

    assert engine.compression_factor(100_000.0) == pytest.approx(1.0)
    assert engine.compression_factor(200_000.0) == pytest.approx(2.0)
    assert engine.compression_factor(1_000.0) == pytest.approx(0.1)

    engine.update(None, 120_000.0)
    assert engine._base == 120_000.0
    assert engine.current_scale() == pytest.approx(1.0)
    engine.update_baseline(130_000.0)
    assert engine._base == 130_000.0


def test_capital_scaling_engine_computes_and_caps_position_size() -> None:
    engine = capital_scaling.CapitalScalingEngine({}, initial_equity=100_000.0)

    assert engine.compute_position_size(0.0, volatility=0.1, drawdown=0.1) == 0.0
    low_risk = engine.compute_position_size(100_000.0, volatility=0.0, drawdown=0.0)
    high_risk = engine.compute_position_size(100_000.0, volatility=0.5, drawdown=0.5)
    assert low_risk > high_risk
    assert engine.scale_position(500.0, equity=200_000.0, volatility=0.0, drawdown=0.0) == 500.0
    assert engine.scale_position(10_000.0, equity=100_000.0, volatility=0.0, drawdown=0.0) == 2_000.0


def test_update_if_present_handles_result_current_scale_and_failure_paths() -> None:
    runtime = SimpleNamespace()

    class ResultScaler:
        def update(self, _runtime, _equity):
            return 0.75

    runtime.capital_scaler = ResultScaler()
    assert capital_scaling.update_if_present(runtime, 100_000.0) == 0.75

    class ObjectScaler:
        def update(self, _runtime, _equity):
            return object()

    runtime.capital_scaler = ObjectScaler()
    assert capital_scaling.update_if_present(runtime, 100_000.0) == 1.0

    class BadCurrent:
        def current_scale(self):
            raise ValueError("bad")

    runtime.capital_scaler = BadCurrent()
    assert capital_scaling.capital_scale(runtime) == 1.0


