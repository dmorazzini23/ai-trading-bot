from tests.optdeps import require
require("pandas")
import pandas as pd
from ai_trading.data.universe import load_universe, locate_tickers_csv


def test_env_overrides_packaged(monkeypatch, tmp_path):
    csv = tmp_path / "tick.csv"
    pd.DataFrame({"symbol": ["MSFT", "NVDA", "META"]}).to_csv(csv, index=False)
    monkeypatch.setenv("AI_TRADER_TICKERS_CSV", str(csv))
    path = locate_tickers_csv()
    assert path == str(csv)
    uni = load_universe()
    assert set(uni) == {"MSFT", "NVDA", "META"}


def test_packaged_exists_without_env(monkeypatch):
    monkeypatch.delenv("AI_TRADER_TICKERS_CSV", raising=False)
    p = locate_tickers_csv()
    assert p and p.endswith("ai_trading/data/tickers.csv")
    uni = load_universe()
    assert len(uni) > 3  # S&P-100 default


def test_missing_returns_empty(monkeypatch):
    monkeypatch.setenv("AI_TRADER_TICKERS_CSV", "/nonexistent.csv")
    import ai_trading.data.universe as U
    orig = U.locate_tickers_csv
    U.locate_tickers_csv = lambda: None
    try:
        assert load_universe() == []
    finally:
        U.locate_tickers_csv = orig
