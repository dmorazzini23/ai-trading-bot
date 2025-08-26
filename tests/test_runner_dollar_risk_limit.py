import sys
from types import SimpleNamespace

import pytest


def test_runner_prefers_config_for_dollar_risk_limit(monkeypatch):
    """Environment override should propagate to runtime.params."""
    monkeypatch.setenv("DOLLAR_RISK_LIMIT", "0.2")

    from ai_trading.core import runtime as runtime_mod

    orig_build = runtime_mod.build_runtime

    def fake_build(cfg):
        rt = orig_build(cfg)
        rt.params["DOLLAR_RISK_LIMIT"] = 0.05
        return rt

    monkeypatch.setattr(runtime_mod, "build_runtime", fake_build)

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
    monkeypatch.setattr("ai_trading.core.runtime.enhance_runtime_with_context", lambda rt, ctx: rt)

    warnings = []
    monkeypatch.setattr(runner.log, "warning", lambda msg, *a, **k: warnings.append(msg))

    runner.run_cycle()

    assert captured["runtime"].params["DOLLAR_RISK_LIMIT"] == pytest.approx(0.2)
    assert "DOLLAR_RISK_LIMIT_MISMATCH" in warnings
