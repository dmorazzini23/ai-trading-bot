import types
from ai_trading import data_fetcher
from ai_trading.utils import http


def test_universe_fetch_pooling(monkeypatch):
    calls = {}

    def fake_map_get(urls, timeout=None, headers=None):
        calls['count'] = calls.get('count', 0) + 1
        calls['len'] = len(urls)
        return [((u, 200, f"BODY{i}".encode()), None) for i, u in enumerate(urls)]

    monkeypatch.setattr(http, "map_get", fake_map_get)
    monkeypatch.setattr(data_fetcher, "_parse_bars", lambda s, c, b: b.decode())

    symbols = ["AAA", "BBB", "CCC"]
    out = data_fetcher.fetch_daily_data_async(symbols, "2024-01-01", "2024-01-02")

    assert calls['count'] == 1
    assert calls['len'] == len(symbols)
    assert [out[s] for s in symbols] == [f"BODY{i}" for i in range(len(symbols))]

    stats = http.pool_stats()
    for key in ("workers", "per_host", "pool_maxsize"):
        assert key in stats
