import sys
from types import SimpleNamespace

import pytest


def test_runner_uses_trading_config_for_dollar_risk_limit(monkeypatch):
    """runtime.params should reflect TradingConfig.from_env()."""
    monkeypatch.setenv("DOLLAR_RISK_LIMIT", "0.2")

    from ai_trading import runner

    captured = {}

    def worker(state, runtime):
        captured["runtime"] = runtime

    class DummyState:
        pass

    monkeypatch.setattr(runner, "_load_engine", lambda: (worker, DummyState))
    dummy_bot_engine = SimpleNamespace(
        get_ctx=lambda: SimpleNamespace(),
        _maybe_warm_cache=lambda ctx: None,
    )
    monkeypatch.setitem(sys.modules, "ai_trading.core.bot_engine", dummy_bot_engine)
    monkeypatch.setitem(sys.modules, "ai_trading.data_fetcher", SimpleNamespace(DataFetchError=Exception))
    monkeypatch.setattr(
        "ai_trading.core.runtime.enhance_runtime_with_context", lambda rt, ctx: rt
    )

    runner.run_cycle()

    from ai_trading.config.management import TradingConfig

    cfg = TradingConfig.from_env()
    assert captured["runtime"].params["DOLLAR_RISK_LIMIT"] == pytest.approx(
        cfg.dollar_risk_limit
    )
