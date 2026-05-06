"""Unit tests for broker synchronization helpers."""

import json
from types import SimpleNamespace

import pytest

from ai_trading.execution.engine import ExecutionEngine
from ai_trading.execution.live_trading import LiveTradingExecutionEngine
from ai_trading.oms.event_store import EventStore
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


def test_update_broker_snapshot_prefers_remaining_open_quantity() -> None:
    """Open-order exposure should use unfilled quantity, not original order size."""

    engine = ExecutionEngine()
    open_orders = [
        {"symbol": "AAPL", "side": "buy", "qty": "10", "filled_qty": "4"},
        {"symbol": "AAPL", "side": "sell", "qty": "7", "filled_qty": "2"},
        {"symbol": "MSFT", "side": "buy_to_cover", "qty": "9", "unfilled_qty": "3"},
    ]

    snapshot = engine._update_broker_snapshot(open_orders, positions=[])

    assert snapshot.open_buy_by_symbol["AAPL"] == 6
    assert snapshot.open_sell_by_symbol["AAPL"] == 5
    assert snapshot.open_buy_by_symbol["MSFT"] == 3
    assert engine.open_order_totals("AAPL") == (6, 5)


def test_update_broker_snapshot_persists_runtime_position_and_risk_snapshots(
    monkeypatch,
    tmp_path,
) -> None:
    """Broker sync should persist immutable position and risk snapshots when enabled."""

    db_path = tmp_path / "runtime_snapshot_events.db"
    monkeypatch.setenv("AI_TRADING_OMS_RUNTIME_SNAPSHOT_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_OMS_INTENT_STORE_PATH", str(db_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("AI_TRADING_MODEL_ID", "ml-main")
    monkeypatch.setenv("AI_TRADING_MODEL_VERSION", "v1")
    monkeypatch.setenv("AI_TRADING_DATASET_HASH", "ds-1")
    monkeypatch.setenv("AI_TRADING_FEATURE_VERSION", "fv-1")
    monkeypatch.setenv("AI_TRADING_MODEL_ARTIFACT_HASH", "artifact-1")
    monkeypatch.setenv("AI_TRADING_CONFIG_SNAPSHOT_HASH", "cfg-1")
    monkeypatch.setenv("AI_TRADING_EFFECTIVE_POLICY_HASH", "policy-1")

    engine = ExecutionEngine()
    open_orders = [
        {"symbol": "AAPL", "side": "buy", "qty": 1.5},
        {"symbol": "AAPL", "side": "sell", "qty": 0.5},
    ]
    positions = [SimpleNamespace(symbol="AAPL", qty=3, side="long", market_price=190.25)]

    engine._update_broker_snapshot(open_orders, positions)

    store = EventStore(url=f"sqlite:///{db_path}")
    position_rows = store.list_position_snapshots(symbol="AAPL")
    risk_rows = store.list_risk_snapshots(source="executionengine")
    assert len(position_rows) >= 1
    assert len(risk_rows) >= 1
    latest_position = position_rows[-1]
    latest_risk = risk_rows[-1]
    assert latest_position["symbol"] == "AAPL"
    assert float(latest_position["quantity"]) == pytest.approx(3.0)
    assert int(latest_risk["positions_count"]) == 1
    assert int(latest_risk["open_orders_count"]) == 2
    assert str(latest_position["policy_hash"]) == "policy-1"
    assert str(latest_position["model_hash"]) == "artifact-1"
    assert str(latest_risk["policy_hash"]) == "policy-1"
    assert str(latest_risk["model_hash"]) == "artifact-1"
    assert str(latest_risk["config_hash"]) == "cfg-1"
    position_payload = json.loads(str(latest_position["payload_json"]))
    risk_payload = json.loads(str(latest_risk["payload_json"]))
    position_lineage = position_payload.get("lineage", {})
    risk_lineage = risk_payload.get("lineage", {})
    assert position_lineage["dataset_hash"] == "ds-1"
    assert position_lineage["feature_version"] == "fv-1"
    assert risk_lineage["model_id"] == "ml-main"
    assert risk_lineage["model_version"] == "v1"


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


def test_live_engine_broker_sync_fail_closes_on_open_order_fetch_failure(
    monkeypatch,
) -> None:
    """Broker sync must not reconcile durable state against an unknown open-order list."""

    class _OpenOrderFailureClient:
        def get_orders(self, **_kwargs):
            raise TimeoutError("open orders unavailable")

        def get_all_positions(self):
            return [SimpleNamespace(symbol="AMD", qty=2)]

    engine = LiveTradingExecutionEngine(ctx=None)
    preserved = engine._update_broker_snapshot(
        [SimpleNamespace(symbol="AAPL", side="buy", qty=2)],
        positions=[],
    )
    engine.trading_client = _OpenOrderFailureClient()
    reconcile_calls: list[object] = []
    pending_calls: list[object] = []
    monkeypatch.setattr(
        engine,
        "_reconcile_durable_intents",
        lambda **kwargs: reconcile_calls.append(kwargs),
    )
    monkeypatch.setattr(
        engine,
        "_reconcile_pending_order_runtime_artifacts",
        lambda **kwargs: pending_calls.append(kwargs),
    )

    snapshot = engine.synchronize_broker_state()

    assert snapshot is preserved
    assert reconcile_calls == []
    assert pending_calls == []
    assert engine.open_order_totals("AAPL") == (2, 0)


def test_live_engine_fetches_native_alpaca_orders_with_filter_request() -> None:
    """Broker sync should use alpaca-py's get_orders(filter=...) contract."""

    class _NativeAlpacaClient:
        def __init__(self) -> None:
            self.filter = None

        def get_orders(self, *, filter):
            self.filter = filter
            return [SimpleNamespace(symbol="AMD", side="buy", qty=3)]

        def get_all_positions(self):
            return [SimpleNamespace(symbol="AMD", qty=2)]

    client = _NativeAlpacaClient()
    engine = LiveTradingExecutionEngine(ctx=None)
    engine.trading_client = client

    snapshot = engine.synchronize_broker_state()

    assert snapshot.open_orders
    assert client.filter is not None
    assert getattr(client.filter, "status", None).value == "open"
    assert engine.open_order_totals("AMD") == (3, 0)


def test_pending_policy_snapshot_uses_native_alpaca_order_filter() -> None:
    """Pending-order maintenance must use the native order request filter."""

    class _NativeAlpacaClient:
        def __init__(self) -> None:
            self.filter = None

        def get_account(self):
            return SimpleNamespace(status="ACTIVE")

        def get_orders(self, *, filter):
            self.filter = filter
            return [SimpleNamespace(id="ord-1", symbol="AMD", status="open")]

        def get_all_positions(self):
            return []

    client = _NativeAlpacaClient()
    engine = LiveTradingExecutionEngine(ctx=None)
    engine.trading_client = client

    orders = engine._list_open_orders_snapshot()

    assert len(orders) == 1
    assert client.filter is not None
    assert getattr(client.filter, "status", None).value == "open"


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


def test_live_engine_snapshot_prefers_unfilled_quantity() -> None:
    """Live broker sync should aggregate leaves/unfilled quantities first."""

    engine = LiveTradingExecutionEngine(ctx=None)
    open_orders = [
        SimpleNamespace(symbol="AMD", side="buy", qty="10", filled_qty="4"),
        SimpleNamespace(symbol="AMD", side="sell", qty="5", remaining_qty="2"),
    ]

    snapshot = engine._update_broker_snapshot(open_orders, positions=[])

    assert snapshot.open_buy_by_symbol["AMD"] == 6
    assert snapshot.open_sell_by_symbol["AMD"] == 2
    assert engine.open_order_totals("AMD") == (6, 2)


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


def test_live_engine_reconciles_terminal_broker_lookup_by_client_id(
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

    class _ClientOrderIdTradingClient:
        def get_orders(self, status: str = "open"):
            assert status == "open"
            return []

        def get_all_positions(self):
            return []

        def get_order_by_client_id(self, client_order_id: str):
            assert client_order_id == intent.intent_id
            return SimpleNamespace(
                id="broker-order-client-fallback",
                client_order_id=client_order_id,
                status="filled",
                symbol="MSFT",
                side="buy",
                qty=5,
            )

    engine.trading_client = _ClientOrderIdTradingClient()

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


def test_order_manager_reconcile_matches_tradier_tag_client_order_id(
    tmp_path,
) -> None:
    """Tradier tag should satisfy client_order_id reconciliation fallback."""

    store = IntentStore(path=str(tmp_path / "tradier_tag_client_order_id.db"))
    engine = ExecutionEngine()
    engine.order_manager.configure_intent_store(store)

    intent, created = store.create_intent(
        intent_id="intent-tradier-tag",
        idempotency_key="tradier-tag-key",
        symbol="AAPL",
        side="buy",
        quantity=3.0,
        status="SUBMITTED",
    )
    assert created is True

    summary = engine.order_manager.reconcile_open_intents(
        broker_orders=[
            {
                "id": "tradier-order-1",
                "tag": intent.intent_id,
                "symbol": "AAPL",
                "side": "buy",
                "quantity": "3",
                "status": "open",
            }
        ],
    )

    refreshed = store.get_intent(intent.intent_id)
    assert refreshed is not None
    assert refreshed.broker_order_id == "tradier-order-1"
    assert summary["matched_open_orders"] == 1
    assert summary["marked_submitted"] == 1


def test_order_manager_reconcile_records_tradier_partial_cancel_aliases(
    tmp_path,
) -> None:
    """Tradier exec_quantity/avg_fill_price should feed partial terminal fills."""

    store = IntentStore(path=str(tmp_path / "tradier_partial_cancel_aliases.db"))
    engine = ExecutionEngine()
    engine.order_manager.configure_intent_store(store)

    intent, created = store.create_intent(
        intent_id="intent-tradier-partial-cancel",
        idempotency_key="tradier-partial-cancel-key",
        symbol="MSFT",
        side="buy",
        quantity=10.0,
        status="SUBMITTED",
    )
    assert created is True
    store.mark_submitted(intent.intent_id, "tradier-order-partial")

    summary = engine.order_manager.reconcile_open_intents(
        broker_orders=[],
        get_order_by_id_fn=lambda _order_id: {
            "id": "tradier-order-partial",
            "tag": intent.intent_id,
            "symbol": "MSFT",
            "side": "buy",
            "quantity": "10",
            "status": "canceled",
            "exec_quantity": "4",
            "avg_fill_price": "249.5",
        },
    )

    refreshed = store.get_intent(intent.intent_id)
    fills = store.list_fills(intent.intent_id)
    assert refreshed is not None
    assert refreshed.status == "CANCELED"
    assert len(fills) == 1
    assert fills[0].fill_qty == pytest.approx(4.0)
    assert fills[0].fill_price == pytest.approx(249.5)
    assert summary["intents_checked"] == 1


def test_order_manager_reconcile_treats_broker_error_as_rejected(
    tmp_path,
) -> None:
    """Broker error status is terminal rejection evidence."""

    store = IntentStore(path=str(tmp_path / "broker_error_terminal.db"))
    engine = ExecutionEngine()
    engine.order_manager.configure_intent_store(store)

    intent, created = store.create_intent(
        intent_id="intent-broker-error",
        idempotency_key="broker-error-key",
        symbol="MSFT",
        side="buy",
        quantity=2.0,
        status="SUBMITTED",
    )
    assert created is True
    store.mark_submitted(intent.intent_id, "broker-error-order")

    engine.order_manager.reconcile_open_intents(
        broker_orders=[],
        get_order_by_id_fn=lambda _order_id: {
            "id": "broker-error-order",
            "client_order_id": intent.intent_id,
            "symbol": "MSFT",
            "side": "buy",
            "quantity": "2",
            "status": "error",
        },
    )

    refreshed = store.get_intent(intent.intent_id)
    assert refreshed is not None
    assert refreshed.status == "REJECTED"
