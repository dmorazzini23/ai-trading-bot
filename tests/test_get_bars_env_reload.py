from datetime import datetime, timedelta, UTC

import pandas as pd

import ai_trading.config.settings as settings_mod
from ai_trading.data import fetch
from ai_trading.config import management


def test_get_bars_recovers_after_env_reload(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        """
ALPACA_API_KEY=key
ALPACA_SECRET_KEY=secret
ALPACA_API_URL=https://paper-api.alpaca.markets
WEBHOOK_SECRET=wh
CAPITAL_CAP=0.04
DOLLAR_RISK_LIMIT=0.05
""".strip()
    )

    for key in [
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "ALPACA_API_URL",
        "WEBHOOK_SECRET",
        "CAPITAL_CAP",
        "DOLLAR_RISK_LIMIT",
    ]:
        monkeypatch.delenv(key, raising=False)

    management.reload_env(str(env_path))

    real_get_settings = settings_mod.get_settings
    calls = {"n": 0}

    def fake_get_settings():
        calls["n"] += 1
        if calls["n"] == 1:
            return None
        return real_get_settings()

    monkeypatch.setattr(settings_mod, "get_settings", fake_get_settings)
    monkeypatch.setattr(fetch, "_fetch_bars", lambda *a, **k: pd.DataFrame())

    start = datetime.now(UTC) - timedelta(minutes=1)
    end = datetime.now(UTC)

    df = fetch.get_bars("AAPL", "1Min", start, end)
    assert isinstance(df, pd.DataFrame)
    assert calls["n"] == 2
