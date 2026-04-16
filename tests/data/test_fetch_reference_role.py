from __future__ import annotations

from datetime import UTC, datetime, timedelta
import sys
import types

import pandas as pd

from ai_trading.data import fetch as data_fetcher


def _sample_frame() -> pd.DataFrame:
    ts = datetime.now(UTC).replace(second=0, microsecond=0)
    return pd.DataFrame(
        [
            {
                "timestamp": ts,
                "open": 100.0,
                "high": 101.0,
                "low": 99.5,
                "close": 100.5,
                "volume": 1000.0,
            },
        ],
    )


def test_get_bars_reference_role_skips_execution_fetch(monkeypatch) -> None:
    start = datetime.now(UTC) - timedelta(minutes=20)
    end = datetime.now(UTC) - timedelta(minutes=1)
    expected = _sample_frame()
    called_kwargs: dict[str, object] = {}

    def _fake_reference(*args, **kwargs):
        del args
        called_kwargs.clear()
        called_kwargs.update(kwargs)
        return expected.copy()

    monkeypatch.setattr(data_fetcher, "_fetch_reference_bars", _fake_reference)
    monkeypatch.setattr(
        data_fetcher,
        "_fetch_bars",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("_fetch_bars should not run")),
    )

    result = data_fetcher.get_bars(
        "AAPL",
        "1Min",
        start,
        end,
        feed_role="reference",
        return_meta=False,
    )
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert called_kwargs["feed"] == "delayed_sip"

    result_with_meta = data_fetcher.get_bars(
        "AAPL",
        "1Min",
        start,
        end,
        feed_role="reference",
        return_meta=True,
    )
    assert isinstance(result_with_meta, tuple)
    _, meta = result_with_meta
    assert meta["feed_role"] == "reference"
    assert meta["provider"] == "alpaca_reference"


def test_get_minute_df_delayed_feed_routes_reference(monkeypatch) -> None:
    start = datetime.now(UTC) - timedelta(minutes=10)
    end = datetime.now(UTC) - timedelta(minutes=1)
    expected = _sample_frame()
    called_kwargs: dict[str, object] = {}

    def _fake_reference(*args, **kwargs):
        del args
        called_kwargs.clear()
        called_kwargs.update(kwargs)
        return expected.copy()

    monkeypatch.setattr(data_fetcher, "_fetch_reference_bars", _fake_reference)

    result = data_fetcher.get_minute_df("AAPL", start, end, feed="delayed_sip")
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert called_kwargs["feed"] == "delayed_sip"


def test_fetch_reference_bars_uses_bars_supported_effective_feed(monkeypatch) -> None:
    start = datetime.now(UTC) - timedelta(days=5)
    end = datetime.now(UTC)
    captured: dict[str, object] = {}
    in_window_ts = start + timedelta(days=1)

    def _fake_get_bars_df(*args, **kwargs):
        del args
        if kwargs:
            captured.update(kwargs)
        return pd.DataFrame(
            [
                {
                    "timestamp": in_window_ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.5,
                    "close": 100.5,
                    "volume": 1000.0,
                },
            ],
        )

    monkeypatch.setattr(data_fetcher, "get_reference_feed", lambda *_args, **_kwargs: "delayed_sip")
    monkeypatch.setattr(data_fetcher, "get_reference_bars_feed", lambda *_args, **_kwargs: "sip")
    alpaca_api_stub = types.ModuleType("ai_trading.alpaca_api")
    setattr(alpaca_api_stub, "get_bars_df", _fake_get_bars_df)
    monkeypatch.setitem(sys.modules, "ai_trading.alpaca_api", alpaca_api_stub)

    result = data_fetcher._fetch_reference_bars(  # noqa: SLF001 - intentional integration test
        "AAPL",
        start,
        end,
        "1Day",
        feed="delayed_sip",
        adjustment="raw",
    )

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert captured.get("feed") == "sip"
    assert result.attrs.get("reference_feed_requested") == "delayed_sip"
    assert result.attrs.get("reference_feed_effective") == "sip"
