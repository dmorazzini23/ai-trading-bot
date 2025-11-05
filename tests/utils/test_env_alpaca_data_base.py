from __future__ import annotations

from ai_trading.utils.env import get_alpaca_data_base_url, get_alpaca_data_v2_base


def test_data_base_url_rejects_trading_host(monkeypatch):
    monkeypatch.setenv("ALPACA_DATA_BASE_URL", "https://api.alpaca.markets")
    assert get_alpaca_data_base_url() == "https://data.alpaca.markets"


def test_data_base_url_custom_proxy(monkeypatch):
    monkeypatch.setenv("ALPACA_DATA_BASE_URL", "http://proxy.example/v2")
    assert get_alpaca_data_base_url() == "http://proxy.example"


def test_data_v2_base(monkeypatch):
    monkeypatch.setenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets")
    assert get_alpaca_data_v2_base() == "https://data.alpaca.markets/v2"
