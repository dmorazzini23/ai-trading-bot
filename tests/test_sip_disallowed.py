from types import SimpleNamespace

from ai_trading.data.fetch import _sip_fallback_allowed, logger
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
    assert sip_disallowed() is False


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


def test_no_unauthorized_log_when_sip_disabled(monkeypatch):
    monkeypatch.setenv("ALPACA_ALLOW_SIP", "0")
    monkeypatch.setattr("ai_trading.data.fetch._ALLOW_SIP", False)
    monkeypatch.setattr("ai_trading.data.fetch._SIP_DISALLOWED_WARNED", False)
    captured: list[str] = []

    def _capture(message: str, *args, **kwargs):
        captured.append(message)

    monkeypatch.setattr(logger, "warning", _capture)
    session = SimpleNamespace(get=lambda *a, **k: SimpleNamespace(status_code=401))
    allowed = _sip_fallback_allowed(session, {}, "1Min")
    assert allowed is False
    assert all(msg != "UNAUTHORIZED_SIP" for msg in captured)


def test_sip_disallowed_with_explicit_entitlement_false(monkeypatch):
    for key in ("ALPACA_ALLOW_SIP", "ALPACA_HAS_SIP"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ALPACA_SIP_ENTITLED", "0")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    assert sip_disallowed() is True

