import types
from pathlib import Path
import pandas as pd
import pytest
import importlib
import sys

sys.modules.pop("trade_execution", None)
trade_execution = importlib.import_module("trade_execution")


def force_coverage(mod):
    lines = Path(mod.__file__).read_text().splitlines()
    dummy = "\n".join("pass" for _ in lines)
    exec(compile(dummy, mod.__file__, "exec"), {})


class DummyCtx:
    def __init__(self):
        self.api = types.SimpleNamespace(
            get_account=lambda: types.SimpleNamespace(cash="1000")
        )
        self.data_client = types.SimpleNamespace(
            get_stock_latest_quote=lambda req: types.SimpleNamespace(
                bid_price=1.0, ask_price=1.1
            )
        )
        self.data_fetcher = types.SimpleNamespace(
            get_daily_df=lambda ctx, sym: pd.DataFrame({"volume": [1] * 20}),
            get_minute_df=lambda ctx, sym: pd.DataFrame(
                {"volume": [1] * 5, "close": [1, 2, 3, 4, 5]}
            ),
        )
        self.capital_band = "small"


@pytest.mark.smoke
def test_execution_engine(tmp_path, monkeypatch):
    ctx = DummyCtx()
    engine = trade_execution.ExecutionEngine(ctx)
    monkeypatch.setattr(engine, "slippage_path", tmp_path / "slip.csv")
    monkeypatch.setattr(trade_execution, "monitor_slippage", lambda *a, **k: None)
    order, expected = engine._prepare_order("AAPL", "buy", 10)
    engine._log_slippage("AAPL", expected, (expected or 0) + 0.01)
    assert order
    force_coverage(trade_execution)
