from __future__ import annotations

import types

import pytest


def test_get_bars_df_retries_rate_limit(monkeypatch):
    pd = pytest.importorskip("pandas")
    import ai_trading.alpaca_api as api

    class FakeAPIError(Exception):
        def __init__(self, status_code: int):
            super().__init__("rate limited")
            self.status_code = status_code
            self.response = types.SimpleNamespace(text="rate limit")

    class DummyRequest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    attempts = {"count": 0}
    sleeps: list[float] = []

    dummy_tf = types.SimpleNamespace(amount=1, unit="minute")
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-02", tz="UTC")

    def fake_get_stock_bars(_request):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise FakeAPIError(429)
        frame = pd.DataFrame(
            {"close": [1.0]},
            index=pd.date_range("2024-01-01", periods=1, freq="T", tz="UTC"),
        )
        return types.SimpleNamespace(df=frame)

    rest = types.SimpleNamespace(get_stock_bars=fake_get_stock_bars)

    monkeypatch.setattr(api, "get_api_error_cls", lambda: FakeAPIError)
    monkeypatch.setattr(api, "_canon_symbol", lambda s: s)
    monkeypatch.setattr(api, "_require_pandas", lambda _: pd)
    monkeypatch.setattr(api, "_normalize_timeframe_for_tradeapi", lambda tf: ("1Min", dummy_tf))
    monkeypatch.setattr(api, "_bars_time_window", lambda _tf: (start, end))
    monkeypatch.setattr(
        api,
        "_format_start_end_for_tradeapi",
        lambda tf, s, e: (s, e, s.isoformat(), e.isoformat()),
    )
    monkeypatch.setattr(api, "get_stock_bars_request_cls", lambda: DummyRequest)
    monkeypatch.setattr(api, "_get_rest", lambda bars=True: rest)
    monkeypatch.setattr(api, "_alpaca_calls_total", types.SimpleNamespace(inc=lambda: None))
    monkeypatch.setattr(
        api, "_alpaca_call_latency", types.SimpleNamespace(observe=lambda _v: None)
    )
    monkeypatch.setattr(api, "_alpaca_errors_total", types.SimpleNamespace(inc=lambda: None))
    monkeypatch.setattr(api.time, "sleep", lambda value: sleeps.append(value))

    frame = api.get_bars_df("AAPL", timeframe="1Min")

    assert attempts["count"] == 3
    assert len(sleeps) == 2
    assert sleeps[0] < sleeps[1]
    assert not frame.empty
