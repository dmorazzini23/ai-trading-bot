"""Regression tests for ExecutionResult surface properties."""

from types import SimpleNamespace

import pytest

from ai_trading.execution.engine import ExecutionResult


@pytest.mark.parametrize(
    "raw_side,expected",
    [
        ("buy", "buy"),
        ("sell", "sell"),
        ("sell_short", "sell"),
        ("cover", "buy"),
    ],
)
def test_execution_result_side_and_symbol(raw_side: str, expected: str) -> None:
    """ExecutionResult should expose normalized side and symbol."""

    order = SimpleNamespace(id="ord-1", side=raw_side, symbol="AAPL")
    result = ExecutionResult(order, "accepted", 0, 10, None)

    assert result.side == expected
    assert result.symbol == "AAPL"
