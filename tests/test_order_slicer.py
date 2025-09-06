"""Tests for order_slicer utilities."""
from ai_trading.execution.order_slicer import apply_fill_ratio


def test_apply_fill_ratio_handles_none():
    """Ensure None values default to zero."""
    assert apply_fill_ratio({}, 0.5) == 0
    assert apply_fill_ratio({"slice_qty": None}, 0.5) == 0
    assert apply_fill_ratio({"slice_qty": 10}, None) == 0


def test_apply_fill_ratio_basic():
    """Verify normal multiplication works."""
    assert apply_fill_ratio({"slice_qty": 10}, 0.5) == 5
