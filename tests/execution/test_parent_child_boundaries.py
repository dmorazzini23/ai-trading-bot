"""Parent execution must preserve quantity and reject ambiguous order intent."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from ai_trading.execution.engine import ExecutionEngine


@pytest.fixture
def sliced(monkeypatch):
    engine = ExecutionEngine()
    submissions = []
    def submit(symbol, side, quantity, order_type, **kwargs):
        submissions.append((symbol, side, quantity, order_type, kwargs))
        return SimpleNamespace(filled_quantity=quantity,
                               order=SimpleNamespace(average_fill_price=100.1, expected_price=100))
    monkeypatch.setattr(engine, "execute_order", submit)
    monkeypatch.setattr(engine, "_resolve_execution_audit_store", lambda: None)
    monkeypatch.setattr("ai_trading.execution.engine.time.sleep", lambda seconds: None)
    return engine, submissions


@pytest.mark.parametrize("slices", [None, [], [2, 3], [12, 4], [0, 0],
    [0.2, 0.8], [{"ratio": 0.2}, {"ratio": 0.8}],
    [{"percent": 30}, {"percent": 70}], [{"quantity": 2}, {"qty": 8}],
    [{"qty": -1}, {"qty": 15}],
])
def test_slice_planning_conserves_parent_quantity(sliced, slices):
    engine, submissions = sliced
    original = deepcopy(slices)
    results = engine.execute_sliced(slices, symbol="AAPL", side="buy", quantity=10,
                                    parent_order_id="parent-test", slice_interval_seconds=0)
    assert sum(row[2] for row in submissions) == 10
    assert all(row[2] > 0 for row in submissions)
    assert len(results) == len(submissions)
    assert slices == original
    summary = engine.last_parent_execution_summary
    assert summary["filled_quantity"] == 10
    assert summary["requested_quantity"] == 10
    assert summary["failed_slices"] == 0


@pytest.mark.parametrize("order_type", ["unsupported", "", None, "LIMIT"])
def test_unknown_order_type_does_not_become_a_market_order(sliced, order_type):
    engine, submissions = sliced
    if order_type == "LIMIT":
        engine.execute_sliced([10], symbol="AAPL", side="buy", quantity=10,
                             order_type=order_type, limit_price=100)
        assert submissions[0][3].value == "limit"
    else:
        with pytest.raises(ValueError, match="Unsupported order type"):
            engine.execute_sliced([10], symbol="AAPL", side="buy", quantity=10,
                                 order_type=order_type)
        assert submissions == []


@pytest.mark.parametrize("side", [None, "hold", ""])
def test_invalid_side_never_submits_children(sliced, side):
    engine, submissions = sliced
    with pytest.raises(ValueError, match="valid 'side'"):
        engine.execute_sliced([10], symbol="AAPL", side=side, quantity=10)
    assert submissions == []


def test_failed_child_is_reported_without_fabricating_fills(sliced, monkeypatch):
    engine, _ = sliced
    calls = []
    def unavailable(*args, **kwargs):
        calls.append(args)
        return None
    monkeypatch.setattr(engine, "execute_order", unavailable)
    results = engine.execute_sliced([5, 5], symbol="AAPL", side="buy", quantity=10,
        slice_retry_attempts=2, slice_retry_backoff_seconds=0, slice_interval_seconds=0)
    assert results == [None, None]
    assert len(calls) == 4
    summary = engine.last_parent_execution_summary
    assert summary["failed_slices"] == 2
    assert summary["filled_quantity"] == 0
    assert summary["retry_count"] == 2
