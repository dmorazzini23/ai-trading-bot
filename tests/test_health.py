import types

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.core.bot_engine import pre_trade_health_check
import ai_trading.data.fetch as data_fetch
import ai_trading.alpaca_api as alpaca_api


class DummyFetcher:
    def __init__(self, df):
        self.df = df
    def get_daily_df(self, ctx, sym):
        return self.df

class DummyAPI:
    def get_account(self):
        return types.SimpleNamespace()

class DummyCtx:
    def __init__(self, df):
        self.data_fetcher = DummyFetcher(df)
        self.api = DummyAPI()


def test_health_check_empty_dataframe(monkeypatch):
    monkeypatch.setenv("HEALTH_MIN_ROWS", "30")
    ctx = DummyCtx(pd.DataFrame())
    summary = pre_trade_health_check(ctx, ["AAA"], min_rows=30)
    assert summary["failures"] == ["AAA"]


def test_health_check_succeeds(monkeypatch):
    monkeypatch.setenv("HEALTH_MIN_ROWS", "30")
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020", periods=30, tz="UTC"),
            "open": [1] * 30,
            "high": [1] * 30,
            "low": [1] * 30,
            "close": [1] * 30,
            "volume": [1] * 30,
        }
    )
    ctx = DummyCtx(df)
    summary = pre_trade_health_check(ctx, ["AAA"], min_rows=30)
    assert summary["missing_columns"] == []
    assert summary["failures"] == []
    assert summary["checked"] == 1


def test_health_check_accepts_datetime_index(monkeypatch):
    monkeypatch.setenv("HEALTH_MIN_ROWS", "5")
    df = pd.DataFrame(
        {
            "open": [1] * 5,
            "high": [1] * 5,
            "low": [1] * 5,
            "close": [1] * 5,
            "volume": [1] * 5,
        },
        index=pd.date_range("2024-01-01", periods=5, tz="UTC"),
    )
    ctx = DummyCtx(df)
    summary = pre_trade_health_check(ctx, ["AAA"], min_rows=5)
    assert summary["missing_columns"] == []
    assert summary["failures"] == []
    assert summary["checked"] == 1


def test_get_daily_df_normalizes_columns(monkeypatch):
    df = pd.DataFrame(
        {
            "t": [pd.Timestamp("2024-01-01", tz="UTC")],
            "o": [1],
            "h": [1],
            "l": [1],
            "c": [1],
            "v": [1],
        }
    )
    monkeypatch.setattr(data_fetch, "should_import_alpaca_sdk", lambda: True)
    monkeypatch.setattr(alpaca_api, "get_bars_df", lambda *a, **k: df)
    out = data_fetch.get_daily_df("AAA")
    for col in ["timestamp", "open", "high", "low", "close", "volume"]:
        assert col in out.columns


def test_get_daily_df_uses_backup_when_primary_normalization_empties(monkeypatch):
    primary_df = pd.DataFrame(
        {
            "timestamp": [None],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [100],
        }
    )
    fallback_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, tz="UTC"),
            "open": [1.0, 1.2],
            "high": [1.1, 1.25],
            "low": [0.95, 1.15],
            "close": [1.05, 1.22],
            "volume": [100, 150],
        }
    )

    monkeypatch.setattr(data_fetch, "should_import_alpaca_sdk", lambda: True)
    monkeypatch.setattr(alpaca_api, "get_bars_df", lambda *a, **k: primary_df)
    monkeypatch.setattr(
        data_fetch,
        "_backup_get_bars",
        lambda *a, **k: fallback_df,
    )

    out = data_fetch.get_daily_df("AAA")

    assert len(out) == len(fallback_df)
    pd.testing.assert_index_equal(out.index, fallback_df.set_index("timestamp").index)


def test_ensure_schema_then_normalize_restores_timestamp_column():
    df = pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2],
            "high": [1.2, 1.2, 1.25],
            "low": [0.9, 1.0, 1.1],
            "close": [1.05, 1.15, 1.2],
            "volume": [100, 120, 110],
        },
        index=pd.date_range("2024-01-01", periods=3, tz="UTC"),
    )

    ensured = data_fetch.ensure_ohlcv_schema(df, source="test", frequency="1Day")
    normalized = data_fetch.normalize_ohlcv_df(ensured, include_columns=("timestamp",))

    assert "timestamp" in normalized.columns
    expected = pd.Series(normalized.index, index=normalized.index, name="timestamp")
    pd.testing.assert_series_equal(normalized["timestamp"], expected)
