import os
import types
from datetime import UTC, datetime, timedelta

import pytest


def test_get_bars_raises_when_settings_missing(monkeypatch):
    import ai_trading.config.settings as settings_mod
    from ai_trading.data import fetch

    monkeypatch.setattr(settings_mod, "get_settings", lambda: None)

    start = datetime.now(UTC) - timedelta(minutes=1)
    end = datetime.now(UTC)
    with pytest.raises(RuntimeError):
        fetch.get_bars("AAPL", "1Min", start, end)


def test_main_exits_when_env_invalid(monkeypatch):
    import ai_trading.main as m

    def bad_validate():
        raise RuntimeError("missing env")

    monkeypatch.setattr(m, "validate_required_env", lambda *a, **k: bad_validate())
    with pytest.raises(SystemExit) as excinfo:
        m.main([])
    assert excinfo.value.code == 1


def test_fail_fast_env_backfills_positive_risk_defaults(monkeypatch):
    import ai_trading.main as m

    for key in (
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "ALPACA_DATA_FEED",
        "WEBHOOK_SECRET",
        "CAPITAL_CAP",
        "DOLLAR_RISK_LIMIT",
        "ALPACA_API_URL",
        "ALPACA_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)

    class DummyConfig:
        @classmethod
        def from_env(cls, *args, **kwargs):  # noqa: D401, ARG003
            return cls()

    monkeypatch.setattr(m, "TradingConfig", DummyConfig)
    monkeypatch.setattr(m, "reload_env", lambda override=False: None)
    monkeypatch.setattr(m, "validate_required_env", lambda *a, **k: ())
    monkeypatch.setattr(m, "redact_config_env", lambda snapshot: snapshot)
    monkeypatch.setattr(
        m,
        "logger",
        types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, critical=lambda *a, **k: None),
    )
    monkeypatch.setattr(
        m,
        "_resolve_alpaca_env",
        lambda: ("test-key", "test-secret", "https://paper-api.alpaca.markets"),
    )

    m._fail_fast_env()

    assert os.getenv("CAPITAL_CAP") == "0.25"
    assert os.getenv("DOLLAR_RISK_LIMIT") == "0.05"
