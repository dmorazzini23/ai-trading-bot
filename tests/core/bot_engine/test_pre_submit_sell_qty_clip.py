from __future__ import annotations

from ai_trading.core import bot_engine


class _DummyExecEngine:
    def __init__(self, *, open_sell_qty: float) -> None:
        self._open_sell_qty = float(open_sell_qty)

    def open_order_totals(self, symbol: str) -> tuple[float, float]:
        _ = symbol
        return (0.0, self._open_sell_qty)


def test_pre_submit_sell_qty_clip_accounts_for_open_sell_reservations() -> None:
    adjusted_qty, context = bot_engine._clip_sell_qty_to_available_position(
        symbol="AVGO",
        current_shares=3,
        requested_qty=4,
        exec_engine=_DummyExecEngine(open_sell_qty=1.2),
    )

    assert adjusted_qty == 1
    assert context is not None
    assert context["available_qty"] == 1
    assert context["reserved_shares"] == 2


def test_pre_submit_sell_qty_clip_is_noop_without_long_inventory() -> None:
    adjusted_qty, context = bot_engine._clip_sell_qty_to_available_position(
        symbol="AVGO",
        current_shares=0,
        requested_qty=4,
        exec_engine=_DummyExecEngine(open_sell_qty=0.0),
    )

    assert adjusted_qty == 4
    assert context is None
