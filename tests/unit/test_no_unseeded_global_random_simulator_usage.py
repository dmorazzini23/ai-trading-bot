from __future__ import annotations

import pytest

from ai_trading.core.enums import OrderSide, OrderType
from ai_trading.execution.simulator import FillSimulator, SlippageModel


def test_slippage_model_requires_seed_without_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AI_TRADING_ALLOW_NONDETERMINISTIC_SIM", raising=False)
    with pytest.raises(RuntimeError, match="requires deterministic RNG"):
        SlippageModel()


def test_fill_simulator_requires_seed_without_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AI_TRADING_ALLOW_NONDETERMINISTIC_SIM", raising=False)
    with pytest.raises(RuntimeError, match="requires deterministic RNG"):
        FillSimulator()


def test_unseeded_simulators_allow_explicit_nondeterministic_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_ALLOW_NONDETERMINISTIC_SIM", "1")
    slippage = SlippageModel()
    simulator = FillSimulator(slippage_model=slippage)
    result = simulator.simulate_fill(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=5,
        price=190.0,
        order_type=OrderType.LIMIT,
    )
    assert isinstance(result, dict)
    assert "filled" in result


def _simulate_fill_snapshot(seed: int) -> dict:
    simulator = FillSimulator(slippage_model=SlippageModel(seed=seed), seed=seed)
    return simulator.simulate_fill(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        price=190.0,
        order_type=OrderType.LIMIT,
    )


def test_seeded_fill_simulator_is_reproducible() -> None:
    first = _simulate_fill_snapshot(seed=123)
    second = _simulate_fill_snapshot(seed=123)
    assert first == second
