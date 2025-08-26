import builtins
import importlib

import pytest
from tests.optdeps import require

from ai_trading.broker import alpaca_credentials

@pytest.mark.unit
def test_pending_orders_lock_exists_and_is_lock():
    alpaca_api = require("alpaca_api")
    assert hasattr(alpaca_api, "_pending_orders_lock")
    lock = alpaca_api._pending_orders_lock
    # Check that it has threading lock behavior
    lock_type = type(lock).__name__
    assert lock_type in ["RLock", "Lock"], f"Expected RLock or Lock, got {lock_type}"
    assert hasattr(alpaca_api, "_pending_orders")
    assert isinstance(alpaca_api._pending_orders, dict)

@pytest.mark.unit
@pytest.mark.skipif(
    importlib.util.find_spec("alpaca_api") is None,
    reason="alpaca_api not installed",
)
def test_submit_order_uses_client_and_returns(dummy_alpaca_client, monkeypatch):
    alpaca_api = require("alpaca_api")
    submit = getattr(alpaca_api, "submit_order", None)
    if submit is None:
        pytest.skip("submit_order not available")

    # Mock the DRY_RUN setting to False so the actual client is used
    monkeypatch.setattr(alpaca_api, "DRY_RUN", False)
    monkeypatch.setattr(alpaca_api, "SHADOW_MODE", False)

    # Create a simple order request object
    class OrderReq:
        def __init__(self):
            self.symbol = "META"
            self.qty = 1
            self.side = "buy"
            self.time_in_force = "day"

        def __getattr__(self, name):
            if name == '_test_scenario':
                return False
            return None

    order_req = OrderReq()
    res = submit(dummy_alpaca_client, order_req)
    assert res is not None
    assert getattr(res, "id", None) is not None
    assert dummy_alpaca_client.calls, "Client submit_order should be called"


@pytest.mark.unit
def test_initialize_raises_when_sdk_missing(monkeypatch):
    """initialize should raise if the Alpaca SDK is absent."""

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "alpaca_trade_api":
            raise ModuleNotFoundError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError):
        alpaca_credentials.initialize(shadow=False)

    placeholder = alpaca_credentials.initialize(shadow=True)
    assert placeholder.__class__ is object
