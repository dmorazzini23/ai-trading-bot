"""Tests for RSI helpers accepting non-pandas sequences."""

from __future__ import annotations

from typing import Iterable

import pytest


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
def test_feature_builder_rsi_accepts_sequences(factory) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn.pipeline")

    from ai_trading.pipeline import FeatureBuilder

    builder = FeatureBuilder()
    close = factory(range(1, 40))

    result = builder._calculate_rsi(close, 14)

    assert hasattr(result, "iloc")
    assert len(result) == len(close)
