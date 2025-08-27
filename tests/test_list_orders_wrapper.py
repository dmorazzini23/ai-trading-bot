from ai_trading.core.bot_engine import _validate_trading_api, list_open_orders


class _FakeClient:
    def __init__(self):
        self.kwargs = None

    def get_orders(self, *args, **kwargs):
        self.kwargs = kwargs
        return ["ok"]


def test_list_orders_wrapper_passes_status():
    client = _FakeClient()
    assert _validate_trading_api(client)
    orders = list_open_orders(client)
    assert orders == ["ok"]
    assert client.kwargs == {"status": "open"}
