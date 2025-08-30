import logging
import contextlib
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
    record = next(r for r in caplog.records if r.msg in {"POSITION_SIZING_RESOLVED", "POSITION_SIZING_FALLBACK"})
    assert runtime.params["MAX_POSITION_SIZE"] == pytest.approx(record.resolved)
