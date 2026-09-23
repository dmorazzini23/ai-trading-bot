"""Live broker submission paths require the shared OMS owner."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import ai_trading.execution.live_trading as lt
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

    with pytest.raises(RuntimeError, match="DIRECT_LIVE_SUBMIT_REQUIRES_CANONICAL_PRETRADE"):
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
    with pytest.raises(RuntimeError, match="DIRECT_LIVE_SUBMIT_REQUIRES_CANONICAL_PRETRADE"):
        live_engine.safe_submit_order(order_data=object())

    paper_engine = _engine("paper", None)
    paper_engine._assert_submit_owner()


def test_live_low_level_submit_requires_claimed_canonical_intent() -> None:
    intent = SimpleNamespace(
        intent_id="claimed-id", status="SUBMITTING", symbol="AAPL", side="buy", quantity=2.0,
        metadata_json='{"source":"live_execution_engine"}',
    )
    store = SimpleNamespace(
        assert_submit_owner=lambda: None,
        get_intent=lambda _intent_id: intent,
    )
    engine = _engine("live", store)
    payload = {
        "symbol": "AAPL", "side": "buy", "quantity": 2,
        "client_order_id": "claimed-id", "type": "market",
    }

    intent.status = "PENDING_SUBMIT"
    with pytest.raises(RuntimeError, match="LIVE_SUBMIT_DURABLE_PRETRADE_REQUIRED"):
        engine._submit_order_to_alpaca(dict(payload))

    intent.status = "SUBMITTING"
    intent.quantity = 1.0
    with pytest.raises(RuntimeError, match="LIVE_SUBMIT_DURABLE_PRETRADE_REQUIRED"):
        engine._submit_order_to_alpaca(dict(payload))

    intent.quantity = 2.0
    intent.metadata_json = '{}'
    with pytest.raises(RuntimeError, match="LIVE_SUBMIT_DURABLE_PRETRADE_REQUIRED"):
        engine._submit_order_to_alpaca(dict(payload))

    intent.metadata_json = '{"source":"live_execution_engine"}'
    intent.quantity = float("nan")
    with pytest.raises(RuntimeError, match="LIVE_SUBMIT_DURABLE_PRETRADE_REQUIRED"):
        engine._submit_order_to_alpaca(dict(payload))

    intent.quantity = 2.0
    with pytest.raises(RuntimeError, match="LIVE_SUBMIT_DURABLE_PRETRADE_REQUIRED"):
        engine._submit_order_to_alpaca({**payload, "quantity": float("nan")})


def test_live_low_level_submit_accepts_claimed_intent_identity(monkeypatch) -> None:
    intent = SimpleNamespace(
        intent_id="claimed-id", status="SUBMITTING", symbol="AAPL", side="buy", quantity=2.0,
        metadata_json='{"source":"live_execution_engine"}',
    )
    store = SimpleNamespace(
        assert_submit_owner=lambda: None,
        get_intent=lambda _intent_id: intent,
    )
    engine = _engine("live", store)
    requests: list[object] = []

    class Request:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    engine.trading_client = SimpleNamespace(
        submit_order=lambda *, order_data: requests.append(order_data) or SimpleNamespace(
            id="broker-order", client_order_id=order_data.client_order_id, status="new"
        )
    )
    engine._should_suppress_duplicate_client_order_id = lambda **_kwargs: False
    engine._resolve_time_in_force = lambda _value: "day"
    engine._record_client_order_id_submission = lambda _client_id: None
    monkeypatch.setattr(
        lt, "_ensure_request_models",
        lambda: (Request, Request, SimpleNamespace(BUY="buy"), SimpleNamespace(DAY="day")),
    )

    result = engine._submit_order_to_alpaca({
        "symbol": "AAPL", "side": "buy", "quantity": 2,
        "client_order_id": "claimed-id", "type": "market",
    })

    assert getattr(result, "id") == "broker-order"
    assert len(requests) == 1
    assert getattr(requests[0], "client_order_id") == "claimed-id"


def test_live_timeout_does_not_blindly_retry_submission() -> None:
    engine = _engine("live", SimpleNamespace(assert_submit_owner=lambda: None))
    engine._acquire_submit_rate_limit_permit = lambda **_kwargs: (True, {})
    attempts: list[dict[str, object]] = []

    def _submit_order_to_alpaca(order_data: dict[str, object]) -> None:
        attempts.append(order_data)
        raise TimeoutError("response lost after broker acceptance")

    with pytest.raises(TimeoutError):
        engine._execute_with_retry(
            _submit_order_to_alpaca,
            {"symbol": "AAPL", "side": "buy", "type": "market"},
        )
    assert len(attempts) == 1


def test_live_replacement_never_cancels_before_new_intent_is_supported() -> None:
    engine = _engine("live", SimpleNamespace(assert_submit_owner=lambda: None))
    engine._cancel_order_alpaca = lambda _order_id: pytest.fail("original order canceled")
    assert engine._replace_limit_order_with_marketable(
        symbol="AAPL", side="buy", qty=2, existing_order_id="broker-original",
        client_order_id="client-original",
        order_data_snapshot={"symbol": "AAPL", "side": "buy", "quantity": 2},
        limit_price=100.0,
    ) is None
