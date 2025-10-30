from ai_trading import main


def test_http_profile_logging_flag(monkeypatch):
    monkeypatch.delenv("HTTP_PROFILE_LOG_ENABLED", raising=False)
    assert main._http_profile_logging_enabled() is False
    monkeypatch.setenv("HTTP_PROFILE_LOG_ENABLED", "1")
    assert main._http_profile_logging_enabled() is True
