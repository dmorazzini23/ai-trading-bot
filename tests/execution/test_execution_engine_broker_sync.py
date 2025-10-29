"""Unit tests for broker synchronization helpers."""

from types import SimpleNamespace

from ai_trading.execution.engine import ExecutionEngine
from ai_trading.execution.live_trading import LiveTradingExecutionEngine


def test_update_broker_snapshot_tracks_open_quantities() -> None:
    """Engine should accumulate open order quantities per symbol/side."""

    engine = ExecutionEngine()
    open_orders = [
        {"symbol": "AAPL", "side": "buy", "qty": 10},
        SimpleNamespace(symbol="AAPL", side="sell", quantity=4),
        {"symbol": "MSFT", "side": "sell_short", "quantity": 5},
    ]
    positions = [SimpleNamespace(symbol="AAPL", qty=6)]

    snapshot = engine._update_broker_snapshot(open_orders, positions)

    assert snapshot.open_buy_by_symbol["AAPL"] == 10
    assert snapshot.open_sell_by_symbol["AAPL"] == 4
    assert engine.open_order_totals("AAPL") == (10, 4)
    # Synchronize should return cached snapshot without mutation.
    assert engine.synchronize_broker_state() is snapshot


class _StubTradingClient:
    def get_orders(self, status: str = "open"):
        return [
            SimpleNamespace(symbol="AMD", side="buy", qty=3),
            SimpleNamespace(symbol="AMD", side="sell", qty=1),
        ]

    def get_all_positions(self):
        return [SimpleNamespace(symbol="AMD", qty=2)]


def test_live_engine_fetches_broker_state() -> None:
    """Live engine should fetch broker state and update quantity index."""

    engine = LiveTradingExecutionEngine(ctx=None)
    engine.trading_client = _StubTradingClient()

    snapshot = engine.synchronize_broker_state()

    assert snapshot.open_orders
    assert snapshot.positions
    assert engine.open_order_totals("AMD") == (3, 1)
