from types import SimpleNamespace

from ai_trading.data.fetch import _sip_fallback_allowed
from ai_trading.data.fetch.sip_disallowed import sip_disallowed


def test_sip_disallowed_when_flag_disabled(monkeypatch):
    monkeypatch.setenv("ALPACA_ALLOW_SIP", "0")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.delenv("ALPACA_HAS_SIP", raising=False)
    assert sip_disallowed() is True


def test_sip_disallowed_without_credentials(monkeypatch):
    monkeypatch.setenv("ALPACA_ALLOW_SIP", "1")
    monkeypatch.setenv("ALPACA_HAS_SIP", "1")
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    assert sip_disallowed() is False


def test_sip_allowed_with_credentials(monkeypatch):
    monkeypatch.setenv("ALPACA_ALLOW_SIP", "1")
    monkeypatch.setenv("ALPACA_HAS_SIP", "1")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    assert sip_disallowed() is False


def test_sip_disallowed_when_entitlement_missing(monkeypatch):
    monkeypatch.setenv("ALPACA_ALLOW_SIP", "1")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_HAS_SIP", "0")
    assert sip_disallowed() is True


def test_no_unauthorized_log_when_sip_disabled(monkeypatch, caplog):
    monkeypatch.setenv("ALPACA_ALLOW_SIP", "0")
    monkeypatch.setattr("ai_trading.data.fetch._ALLOW_SIP", False)
    monkeypatch.setattr("ai_trading.data.fetch._SIP_DISALLOWED_WARNED", False)
    caplog.set_level("INFO")
    session = SimpleNamespace(get=lambda *a, **k: SimpleNamespace(status_code=401))
    allowed = _sip_fallback_allowed(session, {}, "1Min")
    assert allowed is False
    messages = [record.message for record in caplog.records]
    assert "UNAUTHORIZED_SIP" not in messages


def test_sip_disallowed_with_explicit_entitlement_false(monkeypatch):
    for key in ("ALPACA_ALLOW_SIP", "ALPACA_HAS_SIP"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ALPACA_SIP_ENTITLED", "0")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    assert sip_disallowed() is True


def test_sip_failover_does_not_call_fetch_when_unentitled(monkeypatch):
    class _Session:
        def __init__(self):
            self.called = False

        def get(self, *args, **kwargs):  # pragma: no cover - defensive
            self.called = True
            return SimpleNamespace(status_code=200)

    session = _Session()
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setattr("ai_trading.data.fetch._sip_allowed", lambda: False)
    allowed = _sip_fallback_allowed(session, {}, "1Min")
    assert allowed is False
    assert session.called is False
