from ai_trading.data.fetch.sip_disallowed import sip_disallowed


def test_sip_disallowed_when_flag_disabled(monkeypatch):
    monkeypatch.setenv("ALPACA_ALLOW_SIP", "0")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.delenv("ALPACA_HAS_SIP", raising=False)
    assert sip_disallowed() is True


def test_sip_disallowed_without_credentials(monkeypatch):
    monkeypatch.setenv("ALPACA_ALLOW_SIP", "1")
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("ALPACA_HAS_SIP", raising=False)
    assert sip_disallowed() is True


def test_sip_allowed_with_credentials(monkeypatch):
    monkeypatch.setenv("ALPACA_ALLOW_SIP", "1")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.delenv("ALPACA_HAS_SIP", raising=False)
    assert sip_disallowed() is False


def test_sip_disallowed_when_entitlement_missing(monkeypatch):
    monkeypatch.setenv("ALPACA_ALLOW_SIP", "1")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_HAS_SIP", "0")
    assert sip_disallowed() is True

