from collections import deque

from ai_trading.data import fetch as data_fetcher
from ai_trading.data import provider_monitor as pm


def test_iex_ignores_sip_unauthorized(monkeypatch) -> None:
    monitor = pm.ProviderMonitor()
    monkeypatch.setattr(pm, "provider_monitor", monitor)
    monkeypatch.setattr(data_fetcher, "provider_monitor", monitor)
    monkeypatch.setattr(pm, "_SAFE_MODE_ACTIVE", False, raising=False)
    monkeypatch.setattr(pm, "_SAFE_MODE_REASON", None, raising=False)
    monkeypatch.setattr(pm, "_sip_auth_events", deque(), raising=False)

    monkeypatch.setenv("DATA_FEED_INTRADAY", "iex")
    monkeypatch.setenv("ALPACA_ALLOW_SIP", "1")
    monkeypatch.setenv("ALPACA_HAS_SIP", "1")
    monkeypatch.setenv("ALPACA_SIP_UNAUTHORIZED", "1")
    monkeypatch.setattr(data_fetcher, "_ALLOW_SIP", True, raising=False)
    monkeypatch.setattr(data_fetcher, "_SIP_PRECHECK_DONE", False, raising=False)
    monkeypatch.setattr(data_fetcher, "_SIP_DISALLOWED_WARNED", False, raising=False)
    monkeypatch.setattr(data_fetcher, "_SIP_UNAUTHORIZED", True, raising=False)
    monkeypatch.setattr(data_fetcher, "_SIP_UNAUTHORIZED_UNTIL", None, raising=False)
    monkeypatch.setattr(data_fetcher, "_alpaca_disabled_until", None, raising=False)

    class _DummyResponse:
        status_code = 401

    class _DummySession:
        def get(self, *args, **kwargs):
            return _DummyResponse()

    allowed = data_fetcher._sip_fallback_allowed(_DummySession(), {}, "1Min")
    assert allowed is False
    assert "alpaca" not in monitor.fail_counts
    assert data_fetcher.is_primary_provider_enabled() is True
    assert pm.is_safe_mode_active() is False
