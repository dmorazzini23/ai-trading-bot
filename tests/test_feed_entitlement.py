from ai_trading.data import bars


class _Account:
    def __init__(self, feeds):
        self.market_data_subscription = feeds


class _Client:
    def __init__(self, feeds):
        self._feeds = feeds

    def get_account(self):
        return _Account(self._feeds)


def test_ensure_entitled_feed_switches():
    bars._ENTITLE_CACHE.clear()
    client = _Client(['iex'])
    assert bars._ensure_entitled_feed(client, 'sip') == 'iex'


def test_ensure_entitled_feed_keeps():
    bars._ENTITLE_CACHE.clear()
    client = _Client(['sip'])
    assert bars._ensure_entitled_feed(client, 'sip') == 'sip'
