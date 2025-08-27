from __future__ import annotations

import types

import ai_trading.core.bot_engine as eng


def test_run_all_trades_handles_api_error(monkeypatch, caplog):
    class DummyAPI:
        def __init__(self):
            self.called_with: dict | None = None

        def get_orders(self, **kwargs):
            self.called_with = kwargs
            return []

    class DummyRiskEngine:
        def wait_for_exposure_update(self, timeout: float) -> None:
            pass

    state = eng.BotState()
    api = DummyAPI()
    runtime = types.SimpleNamespace(api=api, risk_engine=DummyRiskEngine())

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
    assert api.called_with is not None
    assert "statuses" in api.called_with
