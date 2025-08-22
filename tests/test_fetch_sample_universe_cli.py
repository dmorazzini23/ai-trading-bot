import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location(
    "ai_trading.tools.fetch_sample_universe",
    pathlib.Path(__file__).resolve().parents[1] / "ai_trading" / "tools" / "fetch_sample_universe.py",
)
fetch_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(fetch_module)  # type: ignore[attr-defined]
run = fetch_module.run


def test_run_success(monkeypatch):
    urls_captured = []
    logged = []

    def fake_build(symbol, start, end):
        return f"https://example.com/{symbol}"

    def fake_map_get(urls, timeout=None):
        urls_captured.extend(urls)
        return [(u, 200, b"OK") for u in urls]

    def fake_info(msg, *args, **kwargs):
        logged.append((msg, kwargs.get("extra", {})))

    monkeypatch.setattr("ai_trading.data_fetcher._build_daily_url", fake_build)
    monkeypatch.setattr("ai_trading.utils.http.map_get", fake_map_get)
    monkeypatch.setattr(
        "ai_trading.utils.http.pool_stats",
        lambda: {"workers": 1, "per_host": 1, "pool_maxsize": 1},
    )
    monkeypatch.setattr("ai_trading.tools.fetch_sample_universe.logger.info", fake_info)

    rc = run(["AAA", "BBB", "CCC"], timeout=1.0)
    assert rc == 0
    assert urls_captured == [
        "https://example.com/AAA",
        "https://example.com/BBB",
        "https://example.com/CCC",
    ]
    timing = [m for m in logged if m[0] == "STAGE_TIMING"]
    assert timing and timing[0][1]["stage"] == "UNIVERSE_FETCH" and timing[0][1]["universe_size"] == 3
    stats = [m for m in logged if m[0] == "HTTP_POOL_STATS"]
    assert stats and {"workers", "per_host", "pool_maxsize"} <= stats[0][1].keys()
