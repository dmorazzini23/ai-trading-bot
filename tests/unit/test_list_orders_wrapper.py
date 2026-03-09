from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from ai_trading.core import alpaca_client


class _GetOrdersClient:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def get_orders(self, *args: Any, **kwargs: Any) -> list[str]:
        self.kwargs = dict(kwargs)
        return ["ok"]

    def cancel_order_by_id(self, order_id: Any) -> tuple[str, Any]:
        return ("cancelled", order_id)


def test_validate_trading_api_maps_get_orders_to_list_orders() -> None:
    client = _GetOrdersClient()

    assert alpaca_client._validate_trading_api(client) is True
    list_orders = cast(Any, getattr(client, "list_orders", None))
    assert callable(list_orders)
    assert list_orders(status="open") == ["ok"]
    assert client.kwargs is not None
    if "filter" in client.kwargs:
        filter_obj = client.kwargs["filter"]
        statuses = getattr(filter_obj, "statuses", None)
        if statuses is None:
            statuses = getattr(filter_obj, "status", None)
        if statuses is None and isinstance(filter_obj, Mapping):
            statuses = filter_obj.get("statuses") or filter_obj.get("status")
        if statuses is not None and not isinstance(statuses, (list, tuple)):
            statuses = [getattr(statuses, "value", statuses)]
        assert statuses in (["open"], ["OPEN"], ("open",), ("OPEN",))
    else:
        assert client.kwargs == {"status": "open"}
