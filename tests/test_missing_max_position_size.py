import logging

from ai_trading.config.management import validate_required_env
import ai_trading.main as m


def test_startup_without_max_position_size(monkeypatch, caplog):
    from ai_trading import position_sizing
    cache = position_sizing._Cache()
    monkeypatch.setattr(position_sizing, "_CACHE", cache)
    monkeypatch.setattr(m, "_CACHE", cache)
    monkeypatch.setattr(m, "_get_equity_from_alpaca", lambda *a, **k: 100000.0)
    monkeypatch.setattr(position_sizing, "_get_equity_from_alpaca", lambda *a, **k: 100000.0)
    env = {
        "ALPACA_API_KEY": "dummy",
        "ALPACA_SECRET_KEY": "dummy",
        "ALPACA_DATA_FEED": "iex",
        "ALPACA_TRADING_BASE_URL": "https://paper-api.alpaca.markets",
        "WEBHOOK_SECRET": "secret",
        "AI_TRADING_CAPITAL_CAP": "0.04",
        "DOLLAR_RISK_LIMIT": "0.05",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("MAX_POSITION_SIZE", raising=False)
    monkeypatch.delenv("AI_TRADING_SIGNAL_MAX_POSITION_SIZE", raising=False)
    monkeypatch.delenv("AI_TRADING_MAX_POSITION_SIZE", raising=False)

    snapshot = validate_required_env()
    assert "AI_TRADING_SIGNAL_MAX_POSITION_SIZE" not in snapshot
    assert snapshot["ALPACA_API_KEY"] == "***"
    assert snapshot["ALPACA_SECRET_KEY"] == "***"

    class DummyCfg:
        alpaca_base_url = "paper"
        paper = True

    class DummySettings:
        max_position_mode = "STATIC"
        trading_mode = "balanced"
        capital_cap = 0.04
        dollar_risk_limit = 0.05
        max_position_size = None

    with caplog.at_level(logging.INFO):
        m._validate_runtime_config(cfg=DummyCfg(), tcfg=DummySettings())

    records = [
        r for r in caplog.records
        if r.name == "ai_trading.position_sizing" and r.msg == "CONFIG_AUTOFIX"
    ]
    assert records, "CONFIG_AUTOFIX log not emitted"
    assert getattr(records[0], "fallback", 0) > 0
