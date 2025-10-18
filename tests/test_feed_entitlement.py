from datetime import UTC, datetime, timedelta

from types import SimpleNamespace

from ai_trading.data import bars
from ai_trading.core import bot_engine


class _Account:
    def __init__(self, feeds, generation=None):
        self.market_data_subscription = feeds
        if generation is not None:
            self.updated_at = generation


class _Client:
    def __init__(self, feeds, generation=None):
        self._feeds = feeds
        self._generation = generation

    def get_account(self):
        return _Account(self._feeds, self._generation)


def test_ensure_entitled_feed_switches():
    bars._ENTITLE_CACHE.clear()
    client = _Client(['iex'])
    assert bars._ensure_entitled_feed(client, 'sip') == 'iex'


def test_get_entitled_feeds_refreshes_cache_on_upgrade():
    bars._ENTITLE_CACHE.clear()
    generation = datetime(2024, 1, 1, tzinfo=UTC)
    client = _Client(['iex'], generation)
    feeds_initial = bars._get_entitled_feeds(client)
    assert feeds_initial == {'iex'}
    cache_key = id(client)
    first_entry = bars._ENTITLE_CACHE[cache_key]
    client._feeds = ['sip']
    client._generation = generation + timedelta(minutes=1)
    feeds_upgraded = bars._get_entitled_feeds(client)
    assert feeds_upgraded == {'sip'}
    second_entry = bars._ENTITLE_CACHE[cache_key]
    assert second_entry is not first_entry
    assert second_entry.generation > first_entry.generation


def test_get_entitled_feeds_refreshes_cache_on_downgrade():
    bars._ENTITLE_CACHE.clear()
    generation = datetime(2024, 1, 1, tzinfo=UTC)
    client = _Client(['sip'], generation)
    feeds_initial = bars._get_entitled_feeds(client)
    assert feeds_initial == {'sip'}
    cache_key = id(client)
    first_entry = bars._ENTITLE_CACHE[cache_key]
    client._feeds = ['iex']
    client._generation = generation + timedelta(minutes=2)
    feeds_downgraded = bars._get_entitled_feeds(client)
    assert feeds_downgraded == {'iex'}
    second_entry = bars._ENTITLE_CACHE[cache_key]
    assert second_entry is not first_entry
    assert second_entry.generation > first_entry.generation


def test_get_entitled_feeds_refreshes_cache_on_generation_change():
    bars._ENTITLE_CACHE.clear()
    generation = datetime(2024, 1, 1, tzinfo=UTC)
    client = _Client(['sip'], generation)
    feeds_initial = bars._get_entitled_feeds(client)
    assert feeds_initial == {'sip'}
    cache_key = id(client)
    first_entry = bars._ENTITLE_CACHE[cache_key]
    client._generation = generation + timedelta(minutes=3)
    feeds_next = bars._get_entitled_feeds(client)
    assert feeds_next == {'sip'}
    second_entry = bars._ENTITLE_CACHE[cache_key]
    assert second_entry is not first_entry
    assert second_entry.generation > first_entry.generation


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


def test_ensure_entitled_feed_normalizes_entitlement_case(monkeypatch):
    bars._ENTITLE_CACHE.clear()
    for key in ("ALPACA_ALLOW_SIP", "ALPACA_SIP_ENTITLED", "ALPACA_HAS_SIP"):
        monkeypatch.delenv(key, raising=False)
    client = _Client(['iex'])
    client.entitlements = ['SIP']
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


def test_ensure_entitled_feed_blocks_direct_sip_when_allow_flag_disables(monkeypatch):
    bars._ENTITLE_CACHE.clear()
    monkeypatch.setenv("ALPACA_ALLOW_SIP", "0")
    for key in ("ALPACA_SIP_ENTITLED", "ALPACA_HAS_SIP"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("ALPACA_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET", raising=False)
    client = _Client(['sip'])
    assert bars._ensure_entitled_feed(client, 'sip') == 'iex'


def test_ensure_entitled_feed_trusts_account_when_sip_disallowed_advisory(monkeypatch):
    bars._ENTITLE_CACHE.clear()
    for key in ("ALPACA_ALLOW_SIP", "ALPACA_SIP_ENTITLED", "ALPACA_HAS_SIP"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ALPACA_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET", "test-secret")
    monkeypatch.setattr(bars, "sip_disallowed", lambda: True)
    client = _Client(['sip'])
    assert bars._ensure_entitled_feed(client, 'sip') == 'sip'


def test_ensure_entitled_feed_respects_explicit_entitlement_false(monkeypatch):
    bars._ENTITLE_CACHE.clear()
    monkeypatch.delenv("ALPACA_ALLOW_SIP", raising=False)
    monkeypatch.setenv("ALPACA_SIP_ENTITLED", "0")
    monkeypatch.setenv("ALPACA_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET", "test-secret")
    monkeypatch.setattr(bars, "sip_disallowed", lambda: False)
    client = _Client(['sip', 'iex'])
    assert bars._ensure_entitled_feed(client, 'sip') == 'iex'


def test_ensure_entitled_feed_blocks_direct_sip_when_explicit_entitlement_false(monkeypatch):
    bars._ENTITLE_CACHE.clear()
    monkeypatch.delenv("ALPACA_ALLOW_SIP", raising=False)
    monkeypatch.setenv("ALPACA_SIP_ENTITLED", "0")
    monkeypatch.setenv("ALPACA_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET", "test-secret")
    monkeypatch.setattr(bars, "sip_disallowed", lambda: False)
    client = _Client(['sip'])
    assert bars._ensure_entitled_feed(client, 'sip') == 'iex'


def test_sip_authorized_rejects_failover_without_entitlement(monkeypatch):
    monkeypatch.delenv("ALPACA_ALLOW_SIP", raising=False)
    monkeypatch.delenv("ALPACA_HAS_SIP", raising=False)
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SIP_UNAUTHORIZED", raising=False)

    monkeypatch.setattr(
        bot_engine,
        "CFG",
        SimpleNamespace(alpaca_feed_failover=("sip",)),
        raising=False,
    )
    monkeypatch.setattr(
        bot_engine,
        "get_trading_config",
        lambda: SimpleNamespace(
            alpaca_allow_sip=False,
            alpaca_has_sip=False,
            alpaca_api_key=None,
            alpaca_secret_key=None,
        ),
    )
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "_sip_allowed",
        lambda: False,
    )
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "_SIP_UNAUTHORIZED",
        False,
        raising=False,
    )

    assert bot_engine._sip_authorized() is False


def test_sip_allowed_helper_refreshes_on_env_toggle(monkeypatch):
    from ai_trading.data import fetch as data_fetch

    for key in ("ALPACA_ALLOW_SIP", "ALPACA_HAS_SIP"):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("ALPACA_ALLOW_SIP", "0")
    assert data_fetch._sip_allowed() is False

    monkeypatch.setenv("ALPACA_ALLOW_SIP", "1")
    assert data_fetch._sip_allowed() is True

    monkeypatch.setenv("ALPACA_ALLOW_SIP", "0")
    assert data_fetch._sip_allowed() is False
