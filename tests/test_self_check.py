from __future__ import annotations

import types


def test_self_check_avoids_base_url(monkeypatch):
    import ai_trading.scripts.self_check as self_check

    values = {
        "ALPACA_DATA_FEED": "iex",
        "ALPACA_ALLOW_SIP": "0",
        "ALPACA_HAS_SIP": "0",
        "ALPACA_API_KEY": "key",
        "ALPACA_SECRET_KEY": "secret",
    }
    seen: dict[str, object] = {}

    def fake_get_env(key, default=None, cast=None):
        value = values.get(key, default)
        if cast is bool:
            return str(value).strip().lower() in {"1", "true", "yes", "on"}
        if callable(cast):
            try:
                return cast(value)
            except Exception:
                return value
        return value

    class DummyBars:
        def __init__(self):
            self.df = [1, 2, 3]

    class DummyClient:
        def __init__(self, *args, **kwargs):
            seen["init_kwargs"] = dict(kwargs)
            assert "base_url" not in kwargs

        def get_stock_bars(self, *_a, **_k):
            return DummyBars()

    monkeypatch.setattr(self_check, "ensure_dotenv_loaded", lambda *a, **k: None)
    monkeypatch.setattr(self_check, "get_env", fake_get_env)
    monkeypatch.setattr(self_check, "get_data_client_cls", lambda: DummyClient)
    monkeypatch.setattr(self_check, "get_api_error_cls", lambda: Exception)

    self_check.main()

    assert seen["init_kwargs"] == {
        "api_key": "key",
        "secret_key": "secret",
    }
