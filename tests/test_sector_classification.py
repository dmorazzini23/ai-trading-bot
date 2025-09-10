from pathlib import Path

from ai_trading.core import bot_engine


def test_get_sector_known_symbols():
    assert bot_engine.get_sector("PLTR") == "Technology"
    assert bot_engine.get_sector("BABA") == "Technology"


def test_all_core_symbols_have_explicit_sector(monkeypatch):
    """All symbols in tickers.csv should resolve without external lookups."""

    monkeypatch.setattr(
        bot_engine,
        "_fetch_sector_via_yf",
        lambda symbol: (_ for _ in ()).throw(AssertionError("unexpected yfinance call")),
    )
    tickers_path = Path(__file__).resolve().parent.parent / "tickers.csv"
    symbols = [s.strip() for s in tickers_path.read_text().splitlines() if s.strip()]
    for symbol in symbols:
        assert bot_engine.get_sector(symbol) != "Unknown"
