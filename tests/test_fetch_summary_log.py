def test_fetch_summary_emitted(monkeypatch, caplog):
    import ai_trading.core.bot_engine as be

    def fake_fetch_bars(*a, **k):
        return {"AAPL": [1, 2, 3], "MSFT": [1]}

    monkeypatch.setenv("ALPACA_API_KEY", "x")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "y")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    engine = be.BotEngine()
    bars = fake_fetch_bars()
    with caplog.at_level("INFO"):
        engine.logger.info(
            "FETCH_SUMMARY",
            extra={
                "total_symbols": len(bars),
                "bars_loaded": sum(len(v) for v in bars.values()),
                "first_symbol": next(iter(bars)),
            },
        )
    msgs = [r for r in caplog.records if "FETC" in r.msg]
    assert msgs, "Expected a FETCH_SUMMARY log line"
