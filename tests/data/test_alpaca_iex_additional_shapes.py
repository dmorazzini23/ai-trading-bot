import logging

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.data.fetch import ensure_ohlcv_schema


def _assert_standard_columns(frame: pd.DataFrame) -> None:
    assert list(frame.columns[:6]) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert "timestamp" in frame.index.names or frame.index.name == "timestamp"


def test_compact_minute_bars_are_normalized(caplog: pytest.LogCaptureFixture) -> None:
    rows = [
        {"t": "2024-10-01T13:30:00Z", "o": 10.0, "h": 10.5, "l": 9.8, "c": 10.2, "v": 1500},
        {"t": "2024-10-01T13:31:00Z", "o": 10.2, "h": 10.6, "l": 10.1, "c": 10.5, "v": 1800},
    ]
    df = pd.DataFrame(rows)

    caplog.set_level(logging.INFO)
    normalized = ensure_ohlcv_schema(df, source="alpaca_iex", frequency="1Min")

    _assert_standard_columns(normalized)
    assert "OHLCV_COLUMNS_MISSING" not in caplog.text


def test_realtime_alias_variants_are_mapped(caplog: pytest.LogCaptureFixture) -> None:
    rows = [
        {
            "timestamp": "2024-10-01T13:30:00Z",
            "openIexRealtime": 25.0,
            "highIex": 25.5,
            "lowIex": 24.8,
            "iexClosePrice": 25.2,
            "volume": 2100,
        }
    ]
    df = pd.DataFrame(rows)

    caplog.set_level(logging.INFO)
    normalized = ensure_ohlcv_schema(df, source="alpaca_iex", frequency="1Min")

    _assert_standard_columns(normalized)
    assert "OHLCV_COLUMNS_MISSING" not in caplog.text


def test_nested_barset_payload_is_recovered(caplog: pytest.LogCaptureFixture) -> None:
    payload = {
        "bars": [
            {"t": "2024-10-01T13:30:00Z", "o": 30.0, "h": 30.2, "l": 29.9, "c": 30.1, "v": 900},
            {"t": "2024-10-01T13:31:00Z", "o": 30.1, "h": 30.4, "l": 30.0, "c": 30.3, "v": 950},
        ],
        "symbol": "AAPL",
    }
    df = pd.DataFrame([payload])

    caplog.set_level(logging.INFO)
    normalized = ensure_ohlcv_schema(df, source="alpaca_iex", frequency="1Min")

    assert len(normalized) == 2
    _assert_standard_columns(normalized)
    assert "OHLCV_COLUMNS_MISSING" not in caplog.text
    assert "OHLCV_SCHEMA_RECOVERED" in caplog.text
