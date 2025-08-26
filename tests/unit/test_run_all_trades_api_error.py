from __future__ import annotations

import types

import ai_trading.core.bot_engine as eng


def test_run_all_trades_handles_api_error(monkeypatch, caplog):
    class DummyAPI:
        def list_orders(self, status: str = "open"):
            return []

    class DummyRiskEngine:
        def wait_for_exposure_update(self, timeout: float) -> None:
            pass

    state = eng.BotState()
    runtime = types.SimpleNamespace(api=DummyAPI(), risk_engine=DummyRiskEngine())

    monkeypatch.setattr(eng, "_ensure_alpaca_classes", lambda: None)
    monkeypatch.setattr(eng, "_init_metrics", lambda: None)
    monkeypatch.setattr(eng, "is_market_open", lambda: True)
    monkeypatch.setattr(eng, "ensure_alpaca_attached", lambda _rt: None)
    monkeypatch.setattr(eng, "check_pdt_rule", lambda _rt: False)
    monkeypatch.setattr(eng, "get_strategies", lambda: [])
    monkeypatch.setattr(eng, "get_verbose_logging", lambda: False)
    monkeypatch.setattr(eng.CFG, "log_market_fetch", False, raising=False)

    class DummyLock:
        def acquire(self, blocking: bool = False) -> bool:
            return True

        def release(self) -> None:
            pass

    monkeypatch.setattr(eng, "run_lock", DummyLock())

    def raise_api_error(*_a, **_k):
        raise eng.APIError("boom")

    monkeypatch.setattr(eng, "_prepare_run", raise_api_error)

    caplog.set_level("WARNING")
    eng.run_all_trades_worker(state, runtime)

    assert any(str(r.msg).startswith("PREP") for r in caplog.records)
    assert state.running is False
