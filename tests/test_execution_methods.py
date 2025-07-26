import types
import trade_execution


def test_twap_wrapper(monkeypatch):
    class DummyCtx:
        api = types.SimpleNamespace()
        data_fetcher = types.SimpleNamespace(get_minute_df=lambda *a, **k: None)
    engine = trade_execution.ExecutionEngine(DummyCtx())
    monkeypatch.setattr(engine, "_execute_sliced", lambda *a, **k: "ok")
    res = engine.execute_order("AAPL", 10, "buy", method="twap")
    assert res == "ok"
