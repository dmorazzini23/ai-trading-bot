from __future__ import annotations

from datetime import UTC, datetime

from ai_trading.contracts import (
    Bar,
    BrokerOrderSnapshot,
    ExecutionResult,
    PositionSnapshot,
    Quote,
)


def test_quote_from_mapping_normalizes_shape() -> None:
    ts = datetime.now(UTC)

    quote = Quote.from_mapping(
        {
            "symbol": "msft",
            "timestamp": ts.isoformat(),
            "bid": 300.1,
            "ask": 300.3,
            "mid": 300.2,
            "last": 300.25,
            "provider": "alpaca",
            "feed": "iex",
        }
    )

    payload = quote.to_dict()

    assert payload["symbol"] == "MSFT"
    assert payload["provider"] == "alpaca"
    assert payload["feed"] == "iex"
    assert payload["mid"] == 300.2


def test_broker_order_snapshot_from_mapping_normalizes_aliases() -> None:
    ts = datetime.now(UTC)

    snapshot = BrokerOrderSnapshot.from_mapping(
        {
            "client_order_id": "coid-1",
            "order_id": "broker-1",
            "side": "buy",
            "qty": 10,
            "filled_qty": 4,
            "price": 101.5,
            "fill_price": 101.55,
            "status": "partially_filled",
            "venue": "NASDAQ",
            "timestamp": ts.isoformat(),
        }
    )

    payload = snapshot.to_dict()

    assert payload["broker_order_id"] == "broker-1"
    assert payload["limit_price"] == 101.5
    assert payload["filled_qty"] == 4.0
    assert payload["venue"] == "NASDAQ"


def test_execution_result_from_mapping_rehydrates_nested_broker_order() -> None:
    payload = {
        "submitted": True,
        "accepted": True,
        "status": "filled",
        "provider": "alpaca",
        "venue": "NASDAQ",
        "fill_count": 1,
        "filled_qty": 10,
        "realized_slippage_bps": 1.5,
        "fees": 0.25,
        "broker_order": {
            "client_order_id": "coid-2",
            "broker_order_id": "broker-2",
            "side": "sell",
            "qty": 10,
            "filled_qty": 10,
            "limit_price": 99.5,
            "fill_price": 99.45,
            "status": "filled",
            "venue": "NASDAQ",
        },
    }

    result = ExecutionResult.from_mapping(payload).to_dict()

    assert result["submitted"] is True
    assert result["broker_order"]["client_order_id"] == "coid-2"
    assert result["broker_order"]["fill_price"] == 99.45


def test_position_snapshot_from_mapping_normalizes_symbol_and_qty() -> None:
    ts = datetime.now(UTC)

    snapshot = PositionSnapshot.from_mapping(
        {
            "symbol": "nvda",
            "qty": "7",
            "market_value": 700.0,
            "avg_entry_price": 100.0,
            "provider": "alpaca",
            "timestamp": ts.isoformat(),
        }
    )

    payload = snapshot.to_dict()

    assert payload["symbol"] == "NVDA"
    assert payload["qty"] == 7.0
    assert payload["market_value"] == 700.0


def test_bar_from_mapping_preserves_original_metadata() -> None:
    ts = datetime.now(UTC)

    bar = Bar.from_mapping(
        {
            "symbol": "aapl",
            "timestamp": ts.isoformat(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1500,
            "provider": "alpaca",
            "feed": "sip",
            "session": "regular",
        }
    ).to_dict()

    assert bar["symbol"] == "AAPL"
    assert bar["feed"] == "sip"
    assert bar["metadata"]["session"] == "regular"
