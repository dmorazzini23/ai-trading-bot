import types
from typing import Any, cast

from ai_trading.config.management import Settings, derive_cap_from_settings


def _new_settings(**kwargs: Any) -> Settings:
    return cast(Settings, Settings(**kwargs))


def test_max_position_size_fallback_equity_unknown(monkeypatch):
    monkeypatch.delenv("MAX_POSITION_SIZE", raising=False)
    s = _new_settings()
    cap = derive_cap_from_settings(s, equity=None, fallback=8000.0, capital_cap=0.04)
    assert cap == 8000.0


def test_max_position_size_derived_from_equity(monkeypatch):
    monkeypatch.delenv("MAX_POSITION_SIZE", raising=False)
    s = _new_settings()
    cap = derive_cap_from_settings(s, equity=100_000.0, fallback=8000.0, capital_cap=0.04)
    assert cap == 4000.0
