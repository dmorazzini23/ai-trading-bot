from decimal import Decimal
from types import SimpleNamespace

import pytest

from ai_trading.execution import live_trading


def test_extract_api_error_metadata_handles_sparse_exception():
    class CustomError(Exception):
        pass

    err = CustomError("broken")

    metadata = live_trading._extract_api_error_metadata(err)

    assert metadata["detail"] == "broken"
    assert metadata["error_type"] == "CustomError"


def test_submit_limit_order_handles_timeout_without_unboundlocalerror(monkeypatch, caplog):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setattr(live_trading, "get_alpaca_creds", lambda: ("key", "secret"))
    monkeypatch.setattr(live_trading, "get_tick_size", lambda symbol: Decimal("0.01"))
    monkeypatch.setattr(
        live_trading,
        "_call_preflight_capacity",
        lambda *args, **kwargs: SimpleNamespace(can_submit=True, suggested_qty=args[3]),
    )
    monkeypatch.setattr(live_trading, "_safe_mode_guard", lambda *args, **kwargs: False)
    monkeypatch.setattr(live_trading.ExecutionEngine, "_refresh_settings", lambda self: None)
    monkeypatch.setattr(live_trading.ExecutionEngine, "_get_account_snapshot", lambda self: {})
    monkeypatch.setattr(
        live_trading.ExecutionEngine,
        "_should_skip_for_pdt",
        lambda self, *a, **k: (False, "", {}),
    )

    engine = live_trading.ExecutionEngine()
    engine.is_initialized = True
    engine.trading_client = object()
    engine._ensure_initialized = lambda: True
    engine._pre_execution_checks = lambda: True

    def _raise_timeout(*args, **kwargs):
        raise TimeoutError("provider timeout")

    engine._execute_with_retry = _raise_timeout  # type: ignore[assignment]

    with caplog.at_level("ERROR"):
        result = engine.submit_limit_order("AAPL", "buy", 1, limit_price=123.45)

    assert result is None
    assert any(record.message == "ORDER_SUBMIT_RETRIES_EXHAUSTED" for record in caplog.records)


def test_normalize_order_payload_preserves_fractional_fill_qty():
    payload = {"id": "order-1", "status": "partially_filled", "filled_qty": "0.6", "qty": "1"}

    _, status, filled_qty, requested_qty, order_id, client_order_id = live_trading._normalize_order_payload(
        payload,
        qty_fallback=1,
    )

    assert status == "partially_filled"
    assert filled_qty == pytest.approx(0.6)
    assert requested_qty == 1
    assert order_id == "order-1"
    assert client_order_id is None
