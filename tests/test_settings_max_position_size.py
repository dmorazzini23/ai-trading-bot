import types

from ai_trading.config.management import Settings, derive_cap_from_settings


def test_max_position_size_fallback_equity_unknown(monkeypatch):
    monkeypatch.delenv("MAX_POSITION_SIZE", raising=False)
    s = Settings()
    cap = derive_cap_from_settings(s, equity=None, fallback=8000.0, capital_cap=0.04)
    assert cap == 8000.0


def test_max_position_size_derived_from_equity(monkeypatch):
    monkeypatch.delenv("MAX_POSITION_SIZE", raising=False)
    s = Settings()
    cap = derive_cap_from_settings(s, equity=100_000.0, fallback=8000.0, capital_cap=0.04)
    assert cap == 4000.0
