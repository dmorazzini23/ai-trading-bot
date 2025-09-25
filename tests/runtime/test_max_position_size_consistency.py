import contextlib
import logging
from math import floor

import pytest

from ai_trading import main


def test_max_position_size_consistency(monkeypatch, caplog):
    """Bot uses resolved max position size consistently at runtime."""
    # Minimal env so TradingConfig.from_env works
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("ALPACA_API_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("ALPACA_DATA_FEED", "iex")
    monkeypatch.setenv("WEBHOOK_SECRET", "wh")
    monkeypatch.setenv("CAPITAL_CAP", "0.04")
    monkeypatch.setenv("DOLLAR_RISK_LIMIT", "0.05")
    monkeypatch.setenv("SCHEDULER_ITERATIONS", "1")
    monkeypatch.delenv("MAX_POSITION_SIZE", raising=False)
    monkeypatch.delenv("AI_TRADING_MAX_POSITION_SIZE", raising=False)

    # Avoid external side effects
    monkeypatch.setattr(main, "_fail_fast_env", lambda: None)
    monkeypatch.setattr(main, "_init_http_session", lambda cfg: True)
    monkeypatch.setattr(main, "start_api_with_signal", lambda ready, err: ready.set())
    monkeypatch.setattr(main, "optimize_memory", lambda: {})
    monkeypatch.setattr(main, "StageTimer", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(main, "_interruptible_sleep", lambda s: None)
    monkeypatch.setattr(main, "_get_equity_from_alpaca", lambda cfg: 100000, raising=False)
    monkeypatch.setattr(
        "ai_trading.core.runtime._get_equity_from_alpaca", lambda cfg, force_refresh=True: 100000
    )
    monkeypatch.setattr("ai_trading.utils.device.get_device", lambda: None)

    from ai_trading.config.management import TradingConfig
    from ai_trading.core.runtime import build_runtime
    from ai_trading.config import get_settings

    captured = {}

    def fake_run_cycle():
        cfg = TradingConfig.from_env()
        S = get_settings()
        mps = getattr(S, "max_position_size", None)
        if mps is not None:
            try:
                object.__setattr__(cfg, "max_position_size", float(mps))
            except Exception:
                try:
                    setattr(cfg, "max_position_size", float(mps))
                except Exception:
                    pass
        runtime = build_runtime(cfg)
        captured["runtime"] = runtime

    monkeypatch.setattr(main, "run_cycle", fake_run_cycle)

    with caplog.at_level(logging.INFO):
        main.main([])

    runtime = captured["runtime"]
    settings = get_settings()
    messages = [r.getMessage() for r in caplog.records + caplog.get_records("teardown")]
    assert any("POSITION_SIZING_RESOLVED" in m for m in messages)
    assert runtime.params["MAX_POSITION_SIZE"] == pytest.approx(settings.max_position_size)


def test_auto_max_position_mode_overrides_provided_size(monkeypatch, caplog):
    capital_cap = 0.05
    equity = 125_432.78
    expected_size = float(floor(equity * capital_cap))

    monkeypatch.setenv("CAPITAL_CAP", str(capital_cap))
    monkeypatch.setenv("MAX_POSITION_MODE", "auto")
    monkeypatch.setenv("MAX_POSITION_SIZE", "99999")
    monkeypatch.setenv("MAX_DRAWDOWN_THRESHOLD", "0.15")

    monkeypatch.setattr(
        "ai_trading.core.runtime._get_equity_from_alpaca",
        lambda cfg, force_refresh=True: equity,
    )
    monkeypatch.setattr(
        "ai_trading.position_sizing._get_equity_from_alpaca",
        lambda cfg, force_refresh=True: equity,
    )

    from ai_trading.config.management import TradingConfig
    from ai_trading.core.runtime import build_runtime

    cfg = TradingConfig.from_env()

    with caplog.at_level(logging.INFO, logger="ai_trading.core.runtime"):
        runtime = build_runtime(cfg)

    assert runtime.params["MAX_POSITION_SIZE"] == expected_size
    assert cfg.max_position_size == expected_size
