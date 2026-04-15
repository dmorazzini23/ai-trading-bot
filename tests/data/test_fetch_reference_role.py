from __future__ import annotations

from datetime import UTC, datetime, timedelta

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
