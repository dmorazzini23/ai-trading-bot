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


def test_ensure_entitled_feed_upgrades_cached_entitlement():
    bars._ENTITLE_CACHE.clear()
    client = _Client(['iex'])
    assert bars._ensure_entitled_feed(client, 'sip') == 'iex'
    client._feeds = ['sip']
    assert bars._ensure_entitled_feed(client, 'sip') == 'sip'


def test_ensure_entitled_feed_downgrades_cached_entitlement():
    bars._ENTITLE_CACHE.clear()
    client = _Client(['sip'])
    assert bars._ensure_entitled_feed(client, 'sip') == 'sip'
    client._feeds = ['iex']
    assert bars._ensure_entitled_feed(client, 'sip') == 'iex'


def test_ensure_entitled_feed_keeps():
    bars._ENTITLE_CACHE.clear()
    client = _Client(['sip'])
    assert bars._ensure_entitled_feed(client, 'sip') == 'sip'


def test_ensure_entitled_feed_keeps_when_env_unset(monkeypatch):
    bars._ENTITLE_CACHE.clear()
    for key in ("ALPACA_ALLOW_SIP", "ALPACA_SIP_ENTITLED", "ALPACA_HAS_SIP"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ALPACA_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET", "test-secret")
    client = _Client(['sip'])
    assert bars._ensure_entitled_feed(client, 'sip') == 'sip'


def test_ensure_entitled_feed_keeps_when_account_advertises_sip(monkeypatch):
    bars._ENTITLE_CACHE.clear()
    for key in ("ALPACA_ALLOW_SIP", "ALPACA_SIP_ENTITLED"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ALPACA_HAS_SIP", "1")
    monkeypatch.setenv("ALPACA_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET", "test-secret")
    client = _Client(['sip'])
    assert bars._ensure_entitled_feed(client, 'sip') == 'sip'


def test_ensure_entitled_feed_keeps_when_account_reports_sip_without_creds(monkeypatch):
    bars._ENTITLE_CACHE.clear()
    for key in ("ALPACA_ALLOW_SIP", "ALPACA_SIP_ENTITLED", "ALPACA_HAS_SIP"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("ALPACA_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET", raising=False)
    client = _Client(['sip'])
    assert bars._ensure_entitled_feed(client, 'sip') == 'sip'


def test_ensure_entitled_feed_downgrades_when_allow_flag_disables(monkeypatch):
    bars._ENTITLE_CACHE.clear()
    monkeypatch.setenv("ALPACA_ALLOW_SIP", "0")
    for key in ("ALPACA_SIP_ENTITLED", "ALPACA_HAS_SIP"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("ALPACA_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET", raising=False)
    client = _Client(['sip', 'iex'])
    assert bars._ensure_entitled_feed(client, 'sip') == 'iex'
