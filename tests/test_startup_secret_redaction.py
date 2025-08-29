import logging

from ai_trading import main


def test_startup_logs_redact_secrets(caplog, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "AK123456789")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "SK987654321")
    monkeypatch.setenv("ALPACA_API_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("ALPACA_DATA_FEED", "iex")
    monkeypatch.setenv("WEBHOOK_SECRET", "HOOK-SECRET")
    monkeypatch.setenv("CAPITAL_CAP", "0.5")
    monkeypatch.setenv("DOLLAR_RISK_LIMIT", "1000")

    caplog.set_level(logging.INFO)
    main._fail_fast_env()
    joined = "\n".join(str(rec.__dict__) for rec in caplog.records)
    assert "AK123456789" not in joined
    assert "SK987654321" not in joined
    assert "HOOK-SECRET" not in joined

