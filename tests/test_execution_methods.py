import types
from ai_trading import ExecutionEngine


def test_twap_wrapper(monkeypatch):
    class DummyCtx:
        api = types.SimpleNamespace()
        data_fetcher = types.SimpleNamespace(get_minute_df=lambda *a, **k: None)
    engine = ExecutionEngine(DummyCtx())
    monkeypatch.setattr(engine, "_execute_sliced", lambda *a, **k: "ok")
    res = engine.execute_order("AAPL", 10, "buy", method="twap")
    assert res == "ok"
