"""Regression tests for :mod:`ai_trading.data.models`."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pandas as pd

try:  # pragma: no cover - exercised when SDK missing
    import alpaca.data.requests  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - ensure vendor stub installed
    from tests.vendor_stubs import alpaca as _vendor_alpaca

    sys.modules.setdefault("alpaca", _vendor_alpaca)
    sys.modules.setdefault("alpaca.data", _vendor_alpaca.data)
    sys.modules.setdefault("alpaca.data.requests", _vendor_alpaca.data.requests)
    sys.modules.setdefault("alpaca.data.timeframe", _vendor_alpaca.data.timeframe)


def _reload(name: str):
    module = importlib.import_module(name)
    return importlib.reload(module)


def test_alpaca_requests_used_with_sim_execution(monkeypatch):
    """Ensure Alpaca request classes are used when provider is Alpaca."""

    config_mod = importlib.import_module("ai_trading.config")
    settings_mod = importlib.import_module("ai_trading.config.settings")

    monkeypatch.setattr(
        config_mod,
        "get_execution_settings",
        lambda: SimpleNamespace(mode="sim"),
    )
    monkeypatch.setattr(
        settings_mod,
        "get_settings",
        lambda: SimpleNamespace(data_provider="alpaca"),
    )

    _reload("ai_trading.data._alpaca_guard")
    models_mod = _reload("ai_trading.data.models")

    from alpaca.data.requests import StockBarsRequest as SDKStockBarsRequest

    request = models_mod.StockBarsRequest(
        symbol_or_symbols="SPY",
        timeframe=models_mod.TimeFrame.Day,
    )

    assert isinstance(request, SDKStockBarsRequest)

    bars_mod = _reload("ai_trading.data.bars")

    class DummyClient:
        def __init__(self) -> None:
            self.calls: list[SDKStockBarsRequest] = []

        def get_stock_bars(self, req: SDKStockBarsRequest):
            self.calls.append(req)
            return pd.DataFrame(
                {
                    "open": [1.0],
                    "high": [1.0],
                    "low": [1.0],
                    "close": [1.0],
                    "volume": [100],
                },
                index=pd.DatetimeIndex(["2024-01-02T00:00:00+00:00"]),
            )

    client = DummyClient()
    frame = bars_mod.safe_get_stock_bars(
        client,
        request,
        symbol="SPY",
    )

    assert client.calls, "client should receive a StockBarsRequest instance"
    assert isinstance(client.calls[-1], SDKStockBarsRequest)
    assert not frame.empty
