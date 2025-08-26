import ai_trading.core.bot_engine as be


def test_no_legacy_exports():
    assert not hasattr(be, "get_minute_bars")
    assert not hasattr(be, "get_minute_bars_batch")
    assert not hasattr(be, "warmup_cache")
