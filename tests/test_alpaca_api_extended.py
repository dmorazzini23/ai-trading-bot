import types

from ai_trading import alpaca_api


class HTTPError(Exception):
    def __init__(self, status: int):
        self.status = status


def test_submit_order_http_error():
    def submit_order(**_):
        raise HTTPError(500)

    api = types.SimpleNamespace(submit_order=submit_order)
    res = alpaca_api.submit_order(api, symbol="AAPL", qty=1, side="buy")
    assert not res.success
    assert res.status == 500
    assert res.retryable


def test_submit_order_generic_error():
    def submit_order(**_):
        raise Exception("boom")

    api = types.SimpleNamespace(submit_order=submit_order)
    res = alpaca_api.submit_order(api, symbol="AAPL", qty=1, side="buy")
    assert not res.success
    assert res.status == 0
    assert not res.retryable
