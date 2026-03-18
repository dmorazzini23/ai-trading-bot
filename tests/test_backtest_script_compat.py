from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ai_trading.strategies.backtester import BacktestEngine, DefaultExecutionModel


def _load_script_module(name: str, relative_path: str):
    root = Path(__file__).resolve().parents[1]
    script_path = root / relative_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_backtester_runs_without_legacy_bot_engine_hooks() -> None:
    idx = pd.date_range("2025-01-01", periods=20, freq="D")
    closes = [100, 101, 102, 101, 100, 99, 100, 101, 103, 104, 103, 102, 101, 102, 103, 104, 105, 104, 103, 102]
    frame = pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [1_000] * len(closes),
        },
        index=idx,
    )
    engine = BacktestEngine({"AAPL": frame}, DefaultExecutionModel())
    result = engine.run(["AAPL"])
    assert len(result.equity_curve) == len(frame)
    assert not result.trades.empty


def test_download_backtest_data_uses_canonical_alpaca_base_url(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module("download_backtest_data_script", "scripts/download_backtest_data.py")
    requested_keys: list[str] = []

    def fake_get_env(key: str, default=None):
        requested_keys.append(key)
        if key == "ALPACA_API_KEY":
            return "key"
        if key == "ALPACA_SECRET_KEY":
            return "secret"
        return default

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def get_stock_bars(self, _req):
            return SimpleNamespace(df=pd.DataFrame())

    monkeypatch.setattr(module, "get_env", fake_get_env)
    monkeypatch.setattr(module, "StockHistoricalDataClient", FakeClient)
    monkeypatch.setattr(module, "ensure_dotenv_loaded", lambda: None)
    monkeypatch.setattr(module, "load_pandas", lambda: pd)
    monkeypatch.chdir(tmp_path)
    module.main()
    assert "ALPACA_TRADING_BASE_URL" in requested_keys
    assert "ALPACA_BASE_URL" not in requested_keys


def test_run_wfa_uses_current_walk_forward_interface(monkeypatch) -> None:
    module = _load_script_module("run_wfa_script", "scripts/run_wfa.py")

    class FakeFetcher:
        def get_daily_df(self, _ctx, _symbol):
            idx = pd.date_range("2024-01-01", periods=260, freq="D", tz="UTC")
            close = np.linspace(100.0, 130.0, len(idx))
            return pd.DataFrame({"close": close}, index=idx)

    monkeypatch.setattr(module, "DataFetcher", FakeFetcher)
    monkeypatch.setattr(module, "get_env", lambda _k, default=None, cast=str: cast(default) if cast else default)
    result = module.run_walkforward_validation(["AAPL", "MSFT"])
    assert result["symbols_loaded"] == 2
    assert result["rows"] > 0
    assert "fold_count" in result
