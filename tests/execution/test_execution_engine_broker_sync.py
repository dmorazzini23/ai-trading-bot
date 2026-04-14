"""Unit tests for broker synchronization helpers."""

from types import SimpleNamespace

import pytest

from ai_trading.execution.engine import ExecutionEngine
from ai_trading.execution.live_trading import LiveTradingExecutionEngine
from ai_trading.oms.intent_store import IntentStore

pytest.importorskip("sqlalchemy")


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
    assert getattr(engine, "_position_tracker", {}).get("AAPL") == 6
    # Synchronize should return cached snapshot without mutation.
    assert engine.synchronize_broker_state() is snapshot


def test_update_broker_snapshot_preserves_fractional_quantities() -> None:
    """Engine should retain fractional open-order quantities."""

    engine = ExecutionEngine()
    open_orders = [
        {"symbol": "AAPL", "side": "buy", "qty": "0.6"},
        SimpleNamespace(symbol="AAPL", side="sell", quantity="0.4"),
    ]
    snapshot = engine._update_broker_snapshot(open_orders, positions=[])

    assert snapshot.open_buy_by_symbol["AAPL"] == 0.6
    assert snapshot.open_sell_by_symbol["AAPL"] == 0.4
    assert engine.open_order_totals("AAPL") == (0.6, 0.4)


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
    assert getattr(engine, "_position_tracker", {}).get("AMD") == 2


def test_live_engine_preserves_fractional_open_order_quantities() -> None:
    """Live engine broker sync should not truncate fractional quantities."""

    class _FractionalStubTradingClient:
        def get_orders(self, status: str = "open"):
            return [
                SimpleNamespace(symbol="AMD", side="buy", qty="0.6"),
                SimpleNamespace(symbol="AMD", side="sell", qty="0.2"),
            ]

        def get_all_positions(self):
            return [SimpleNamespace(symbol="AMD", qty=2)]

    engine = LiveTradingExecutionEngine(ctx=None)
    engine.trading_client = _FractionalStubTradingClient()

    snapshot = engine.synchronize_broker_state()

    assert snapshot.open_buy_by_symbol["AMD"] == 0.6
    assert snapshot.open_sell_by_symbol["AMD"] == 0.2
    assert engine.open_order_totals("AMD") == (0.6, 0.2)


def test_live_engine_reconciles_terminal_broker_lookup_for_missing_open_intent(
    tmp_path,
) -> None:
    """Live broker sync should close a durable intent only with terminal evidence."""

    store = IntentStore(path=str(tmp_path / "live_engine_reconcile_terminal.db"))
    engine = LiveTradingExecutionEngine(ctx=None)
    engine.order_manager.configure_intent_store(store)

    intent, created = store.create_intent(
        intent_id="intent-live-terminal-lookup",
        idempotency_key="live-terminal-lookup-key",
        symbol="MSFT",
        side="buy",
        quantity=5.0,
        status="SUBMITTED",
    )
    assert created is True
    store.mark_submitted(intent.intent_id, "broker-order-808")

    class _TerminalLookupTradingClient:
        def get_orders(self, status: str = "open"):
            assert status == "open"
            return []

        def get_all_positions(self):
            return []

        def get_order_by_id(self, order_id: str):
            assert order_id == "broker-order-808"
            return SimpleNamespace(
                id=order_id,
                client_order_id=intent.intent_id,
                status="filled",
                symbol="MSFT",
                side="buy",
                qty=5,
            )

    engine.trading_client = _TerminalLookupTradingClient()

    snapshot = engine.synchronize_broker_state()

    assert snapshot.open_orders == ()
    refreshed = store.get_intent(intent.intent_id)
    assert refreshed is not None
    assert refreshed.status == "FILLED"
    open_intent_ids = {record.intent_id for record in store.get_open_intents()}
    assert intent.intent_id not in open_intent_ids


def test_live_engine_reconciles_terminal_broker_lookup_by_client_order_id_fallback(
    tmp_path,
) -> None:
    """Live broker sync should close by client_order_id when broker_order_id is absent."""

    store = IntentStore(path=str(tmp_path / "live_engine_reconcile_client_id.db"))
    engine = LiveTradingExecutionEngine(ctx=None)
    engine.order_manager.configure_intent_store(store)

    intent, created = store.create_intent(
        intent_id="intent-live-client-id-fallback",
        idempotency_key="live-client-id-fallback-key",
        symbol="MSFT",
        side="buy",
        quantity=5.0,
        status="SUBMITTED",
    )
    assert created is True
    assert intent.broker_order_id is None

    class _ClientOrderIdFallbackTradingClient:
        def get_orders(self, status: str = "open"):
            assert status == "open"
            return []

        def get_all_positions(self):
            return []

        def get_order_by_client_order_id(self, client_order_id: str):
            assert client_order_id == intent.intent_id
            return SimpleNamespace(
                id="broker-order-client-fallback",
                client_order_id=client_order_id,
                status="filled",
                symbol="MSFT",
                side="buy",
                qty=5,
            )

    engine.trading_client = _ClientOrderIdFallbackTradingClient()

    snapshot = engine.synchronize_broker_state()

    assert snapshot.open_orders == ()
    refreshed = store.get_intent(intent.intent_id)
    assert refreshed is not None
    assert refreshed.status == "FILLED"
    open_intent_ids = {record.intent_id for record in store.get_open_intents()}
    assert intent.intent_id not in open_intent_ids


def test_live_engine_reconciles_terminal_broker_lookup_for_missing_partially_filled_intent(
    tmp_path,
) -> None:
    """Live broker sync should terminalize a partially filled durable intent."""

    store = IntentStore(path=str(tmp_path / "live_engine_reconcile_partial_fill.db"))
    engine = LiveTradingExecutionEngine(ctx=None)
    engine.order_manager.configure_intent_store(store)

    intent, created = store.create_intent(
        intent_id="intent-live-partial-terminal-lookup",
        idempotency_key="live-partial-terminal-lookup-key",
        symbol="MSFT",
        side="buy",
        quantity=5.0,
        status="SUBMITTED",
    )
    assert created is True
    store.mark_submitted(intent.intent_id, "broker-order-909")
    store.record_fill(intent.intent_id, fill_qty=1.0, fill_price=190.25)

    class _PartialFillTerminalLookupTradingClient:
        def get_orders(self, status: str = "open"):
            assert status == "open"
            return []

        def get_all_positions(self):
            return []

        def get_order_by_id(self, order_id: str):
            assert order_id == "broker-order-909"
            return SimpleNamespace(
                id=order_id,
                client_order_id=intent.intent_id,
                status="filled",
                symbol="MSFT",
                side="buy",
                qty=5,
            )

    engine.trading_client = _PartialFillTerminalLookupTradingClient()

    snapshot = engine.synchronize_broker_state()

    assert snapshot.open_orders == ()
    refreshed = store.get_intent(intent.intent_id)
    assert refreshed is not None
    assert refreshed.status == "FILLED"
    open_intent_ids = {record.intent_id for record in store.get_open_intents()}
    assert intent.intent_id not in open_intent_ids
