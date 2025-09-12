import types

import pytest

from ai_trading import alpaca_api


class HTTPError(Exception):
    def __init__(self, status: int):
        self.status = status


def test_submit_order_http_error():
    def submit_order(**_):
        raise HTTPError(500)

    api = types.SimpleNamespace(submit_order=submit_order)
    with pytest.raises(HTTPError) as e:
        alpaca_api.submit_order("AAPL", "buy", qty=1, client=api)
    assert e.value.status == 500


def test_submit_order_generic_error():
    def submit_order(**_):
        raise Exception("boom")

    api = types.SimpleNamespace(submit_order=submit_order)
    with pytest.raises(Exception):
        alpaca_api.submit_order("AAPL", "buy", qty=1, client=api)
