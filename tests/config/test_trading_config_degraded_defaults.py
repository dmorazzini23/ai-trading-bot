"""Ensure new trading configuration toggles surface via get_trading_config."""

from ai_trading.config import runtime


def test_degraded_feed_defaults(monkeypatch):
    """Default configuration should expose degraded-feed controls."""

    for key in (
        "TRADING__POST_SUBMIT_BROKER_SYNC",
        "TRADING__MIN_QUOTE_FRESHNESS_MS",
        "TRADING__DEGRADED_FEED_MODE",
        "TRADING__DEGRADED_FEED_LIMIT_WIDEN_BPS",
        "LOG__EXEC_SUMMARY_ENABLED",
    ):
        monkeypatch.delenv(key, raising=False)

    runtime.get_trading_config.cache_clear()
    cfg = runtime.get_trading_config()

    assert cfg.post_submit_broker_sync is True
    assert cfg.min_quote_freshness_ms == 1500
    assert cfg.degraded_feed_mode == "widen"
    assert cfg.degraded_feed_limit_widen_bps == 8
    assert cfg.log_exec_summary_enabled is True
