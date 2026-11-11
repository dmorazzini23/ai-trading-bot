from types import SimpleNamespace

from ai_trading.oms.cancel_all import cancel_all_open_orders


class _BulkCancelOnlyAPI:
    def __init__(self) -> None:
        self.bulk_cancels = 0

    def list_orders(self, status: str = "open"):
        return [SimpleNamespace(id="a", client_order_id="ca"), SimpleNamespace(id="b", client_order_id="cb")]

    def cancel_orders(self) -> None:
        self.bulk_cancels += 1


class _PerOrderAPI(_BulkCancelOnlyAPI):
    def __init__(self) -> None:
        super().__init__()
        self.cancelled: list[str] = []

    def cancel_order_by_id(self, order_id: str) -> None:
        self.cancelled.append(order_id)


def test_cancel_all_never_uses_bulk_cancel_for_a_single_order():
    api = _BulkCancelOnlyAPI()

    result = cancel_all_open_orders(SimpleNamespace(api=api))

    assert api.bulk_cancels == 0
    assert result.total_open == 2
    assert result.cancelled == 0
    assert result.failed == 2
    assert all("per-order cancel" in err["error"] for err in result.errors)


def test_cancel_all_cancels_each_order_by_id():
    api = _PerOrderAPI()

    result = cancel_all_open_orders(SimpleNamespace(api=api))

    assert api.cancelled == ["a", "b"]
    assert api.bulk_cancels == 0
    assert result.cancelled == 2
    assert result.failed == 0
