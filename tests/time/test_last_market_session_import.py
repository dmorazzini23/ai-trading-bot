import importlib

def test_canonical_import_only():
    m = importlib.import_module("ai_trading.utils.time")
    assert hasattr(m, "last_market_session")
    df = importlib.import_module("ai_trading.data_fetcher")
    assert not hasattr(df, "last_market_session")
