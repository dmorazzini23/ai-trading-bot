"""Tests for RSI helpers accepting non-pandas sequences."""

from __future__ import annotations

from typing import Iterable

import pytest

from ai_trading.position.profit_taking import ProfitTakingEngine
from ai_trading.position.technical_analyzer import TechnicalSignalAnalyzer
from ai_trading.position.trailing_stops import TrailingStopManager


class MockSequence:
    """Simple sequence without pandas methods like ``diff``."""

    def __init__(self, values: Iterable[float]) -> None:
        self._values = list(values)

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return len(self._values)

    def __iter__(self):  # pragma: no cover - simple delegation
        return iter(self._values)

    def __getitem__(self, index):  # pragma: no cover - simple delegation
        return self._values[index]


@pytest.mark.parametrize("factory", (list, MockSequence))
def test_technical_analyzer_rsi_accepts_sequences(factory) -> None:
    analyzer = TechnicalSignalAnalyzer()
    prices = factory(range(1, 40))

    value = analyzer._calculate_rsi(prices, 14)

    assert isinstance(value, float)
    assert 0.0 <= value <= 100.0


@pytest.mark.parametrize("factory", (list, MockSequence))
def test_trailing_stop_rsi_accepts_sequences(factory) -> None:
    manager = TrailingStopManager()
    prices = factory(range(1, 40))

    value = manager._calculate_rsi(prices, 14)

    assert isinstance(value, float)
    assert 0.0 <= value <= 100.0


@pytest.mark.parametrize("factory", (list, MockSequence))
def test_profit_taking_rsi_accepts_sequences(factory) -> None:
    engine = ProfitTakingEngine()
    prices = factory(range(1, 40))

    value = engine._calculate_rsi(prices, 14)

    assert isinstance(value, float)
    assert 0.0 <= value <= 100.0


@pytest.mark.parametrize("factory", (list, MockSequence))
def test_feature_builder_rsi_accepts_sequences(factory) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn.pipeline")

    from ai_trading.pipeline import FeatureBuilder

    builder = FeatureBuilder()
    close = factory(range(1, 40))

    result = builder._calculate_rsi(close, 14)

    assert hasattr(result, "iloc")
    assert len(result) == len(close)
