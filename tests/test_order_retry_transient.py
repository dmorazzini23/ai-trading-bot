import os

os.environ.setdefault("ENV_IMPORT_GUARD", "0")

import pytest

from ai_trading.execution.live_trading import CapacityCheck, ExecutionEngine


class _DummySettings:
    mode = "paper"
    shadow_mode = False
    order_timeout_seconds = 30
    slippage_limit_bps = 5
    price_provider_order = ()
    data_feed_intraday = "iex"


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch):
    """Provide deterministic execution settings for tests."""

    from ai_trading.execution import live_trading

    monkeypatch.setattr(live_trading, "get_execution_settings", lambda: _DummySettings)
    monkeypatch.setattr(live_trading, "get_alpaca_creds", lambda: ("key", "secret"))
    monkeypatch.setattr(live_trading, "get_alpaca_base_url", lambda: "https://paper-api")
    monkeypatch.setattr(live_trading, "alpaca_credential_status", lambda: (True, True))
    monkeypatch.setattr(
        live_trading,
        "preflight_capacity",
        lambda symbol, side, price, qty, broker: CapacityCheck(True, int(qty), None),
    )


def test_transient_retry_uses_stable_client_id(caplog):
    """Ensure transient failures retry once with the same client order id."""

    engine = ExecutionEngine(shadow_mode=False)
    engine.is_initialized = True
    engine.trading_client = object()

    attempts: list[str] = []

    def fake_submit(order_data):
        attempts.append(order_data["client_order_id"])
        if len(attempts) == 1:
            raise TimeoutError("simulated timeout")
        return {
            "id": "order-ok",
            "client_order_id": order_data["client_order_id"],
            "status": "accepted",
            "symbol": order_data["symbol"],
            "qty": order_data["quantity"],
        }

    engine._submit_order_to_alpaca = fake_submit  # type: ignore[assignment]

    caplog.set_level("WARNING")
    result = engine.submit_market_order("AAPL", "buy", 5)

    assert len(attempts) == 2
    assert attempts[0] == attempts[1]

    prefix = attempts[0].split("-")
    assert prefix[0] == "AAPL"
    assert prefix[1] == "buy"
    assert prefix[2].isdigit()

    assert result["client_order_id"] == attempts[0]
    assert engine.stats["retry_count"] == 1

    scheduled = [rec for rec in caplog.records if rec.message == "ORDER_RETRY_SCHEDULED"]
    assert len(scheduled) == 1
    assert scheduled[0].attempt == 2
    assert scheduled[0].reason == "timeout"

    assert all(rec.message != "ORDER_RETRY_GAVE_UP" for rec in caplog.records)
