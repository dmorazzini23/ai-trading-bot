import pytest

alpaca_api = pytest.importorskip("alpaca_api", reason="alpaca_api module not found")

@pytest.mark.unit
def test_pending_orders_lock_exists_and_is_lock():
    assert hasattr(alpaca_api, "_pending_orders_lock")
    lock = alpaca_api._pending_orders_lock
    # Check that it has threading lock behavior
    lock_type = type(lock).__name__
    assert lock_type in ["RLock", "Lock"], f"Expected RLock or Lock, got {lock_type}"
    assert hasattr(alpaca_api, "_pending_orders")
    assert isinstance(alpaca_api._pending_orders, dict)

@pytest.mark.unit
def test_submit_order_uses_client_and_returns(dummy_alpaca_client, monkeypatch):
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