from __future__ import annotations

import types

import pytest

from ai_trading import main
from ai_trading.config.management import reload_trading_config


class _ContextReached(RuntimeError):
    """Sentinel exception to confirm compute paths were invoked."""


def test_main_skip_compute_policy_respected(monkeypatch, caplog):
    import ai_trading.alpaca_api as alpaca_api
    import ai_trading.core.bot_engine as bot_engine
    from ai_trading.data import fetch as data_fetch_module

    monkeypatch.setenv("SKIP_COMPUTE_WHEN_PROVIDER_DISABLED", "false")
    monkeypatch.setenv("TRADING__DEGRADED_FEED_MODE", "widen")
    reload_trading_config()

    monkeypatch.setattr(main, "should_stop", lambda: False)
    monkeypatch.setattr(main, "_is_market_open_base", lambda: True)
    monkeypatch.setattr(main, "_interruptible_sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(bot_engine, "get_trade_logger", lambda: None)
    monkeypatch.setattr(main, "get_settings", lambda: types.SimpleNamespace(max_position_size=None))
    monkeypatch.setattr(main, "alpaca_credential_status", lambda: (True, True))

    monkeypatch.setattr(alpaca_api, "alpaca_get", lambda *a, **k: None)
    monkeypatch.setattr(alpaca_api, "is_alpaca_service_available", lambda: True)
    monkeypatch.setattr(alpaca_api, "_set_alpaca_service_available", lambda *_a, **_k: None)

    monkeypatch.setattr(
        data_fetch_module,
        "is_primary_provider_enabled",
        lambda: False,
        raising=False,
    )

    def _raise_context(*args, **kwargs):
        raise _ContextReached()

    monkeypatch.setattr(main, "_resolve_cached_context", _raise_context)

    caplog.set_level("INFO")
    with pytest.raises(_ContextReached):
        main.run_cycle()

    assert not any(record.message == "PRIMARY_PROVIDER_DISABLED_CYCLE_SKIP" for record in caplog.records)
