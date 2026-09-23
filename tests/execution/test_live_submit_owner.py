"""Live broker submission paths require the shared OMS owner."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai_trading.execution.live_trading import ExecutionEngine


def _engine(mode: str, store: object | None) -> ExecutionEngine:
    engine = ExecutionEngine.__new__(ExecutionEngine)
    engine.execution_mode = mode
    engine.order_manager = SimpleNamespace(_intent_store=store)
    engine.trading_client = SimpleNamespace(
        submit_order=lambda **_kwargs: pytest.fail("broker submit reached")
    )
    return engine


@pytest.mark.parametrize("mode", ["live", "production"])
def test_live_raw_submit_paths_fail_before_broker_without_owner(mode: str) -> None:
    engine = _engine(mode, None)
    engine._position_quantity = lambda _symbol: -2

    with pytest.raises(RuntimeError, match="OMS_SUBMIT_OWNER_UNAVAILABLE"):
        engine.safe_submit_order(order_data=object())
    with pytest.raises(RuntimeError, match="OMS_SUBMIT_OWNER_UNAVAILABLE"):
        engine._submit_order_to_alpaca({"symbol": "AAPL", "side": "buy"})
    assert engine._submit_cover_order("AAPL", 1) is False


def test_paper_and_owned_live_paths_preserve_owner_check_boundary() -> None:
    checked: list[str] = []
    store = SimpleNamespace(assert_submit_owner=lambda: checked.append("checked"))
    live_engine = _engine("live", store)
    live_engine._assert_submit_owner()
    assert checked == ["checked"]

    paper_engine = _engine("paper", None)
    paper_engine._assert_submit_owner()
