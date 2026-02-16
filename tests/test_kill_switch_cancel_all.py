from __future__ import annotations

from types import SimpleNamespace

from ai_trading.oms.cancel_all import cancel_all_open_orders


class _Api:
    def __init__(self) -> None:
        self.cancelled: list[str] = []

    def list_orders(self, status: str = "open"):
        assert status == "open"
        return [
            SimpleNamespace(id="oid-1", client_order_id="cid-1"),
            SimpleNamespace(id="oid-2", client_order_id="cid-2"),
        ]

    def cancel_order(self, order_id: str) -> None:
        self.cancelled.append(order_id)


def test_cancel_all_open_orders_cancels_everything() -> None:
    runtime = SimpleNamespace(api=_Api())
    result = cancel_all_open_orders(runtime)
    assert result.reason_code == "CANCEL_ALL_TRIGGERED"
    assert result.total_open == 2
    assert result.cancelled == 2
    assert result.failed == 0
    assert runtime.api.cancelled == ["oid-1", "oid-2"]
