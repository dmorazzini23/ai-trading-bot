from __future__ import annotations

import types
import time


def test_poll_order_fill_status_returns_on_filled(monkeypatch):
    from ai_trading.core.execution_flow import poll_order_fill_status

    # Fake API that transitions statuses over successive calls
    class _Order:
        def __init__(self, status: str, filled_qty: str = "0"):
            self.status = status
            self.filled_qty = filled_qty

    class _API:
        def __init__(self):
            self._i = 0

        def get_order(self, _order_id):
            self._i += 1
            if self._i == 1:
                return _Order("new")
            if self._i == 2:
                return _Order("partially_filled", "10")
            return _Order("filled", "100")

    class _Ctx:
        def __init__(self):
            self.api = _API()

    ctx = _Ctx()
    t0 = time.time()
    poll_order_fill_status(ctx, "oid-1", timeout=2)
    elapsed = time.time() - t0
    # Should return quickly (well under timeout) once status becomes filled
    assert elapsed < 2

