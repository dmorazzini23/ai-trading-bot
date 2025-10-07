from types import SimpleNamespace


def test_missing_alpaca_warning_skipped_when_keys_present(monkeypatch):
    from ai_trading.data import fetch as fetch_mod

    settings = SimpleNamespace(
        data_provider="alpaca",
        alpaca_api_key="key",
        alpaca_secret_key="secret",
        alpaca_data_feed="iex",
    )

    monkeypatch.setattr(fetch_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(
        fetch_mod,
        "broker_keys",
        lambda _: {"ALPACA_API_KEY": "key", "ALPACA_SECRET_KEY": "secret"},
    )
    monkeypatch.setattr(fetch_mod, "get_data_feed_override", lambda: None)
    monkeypatch.setattr(fetch_mod, "resolve_alpaca_feed", lambda _: "iex")

    should_warn, extra = fetch_mod._missing_alpaca_warning_context()

    assert should_warn is False
    assert "missing_keys" not in extra
