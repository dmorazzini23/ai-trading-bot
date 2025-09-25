import types

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.features import compute_macd
from ai_trading.core import bot_engine


def test_compute_macd_skips_all_nan_close(monkeypatch, caplog):
    df = pd.DataFrame({"close": [float("nan"), float("nan"), float("nan")]})

    called = {"ema": False}

    def _guarded_ema(values, period):  # pragma: no cover - ensure guard trips before EMA
        called["ema"] = True
        raise AssertionError("EMA should not be invoked for all-NaN input")

    monkeypatch.setattr("ai_trading.features.indicators.ema", _guarded_ema)
    caplog.set_level("ERROR", logger="ai_trading.features.indicators")

    result = compute_macd(df.copy())

    assert not called["ema"], "EMA helper should not be invoked when input is all NaN"
    assert "MACD computation failed" not in [record.getMessage() for record in caplog.records]
    pd.testing.assert_frame_equal(result, df)


def test_fetch_feature_data_skips_without_finite_closes(monkeypatch, caplog):
    index = pd.date_range("2024-01-01", periods=5, freq="min", tz="UTC")
    inf_df = pd.DataFrame(
        {
            "open": [100.0] * len(index),
            "high": [101.0] * len(index),
            "low": [99.0] * len(index),
            "close": [float("inf")] * len(index),
            "volume": [1_000] * len(index),
        },
        index=index,
    )

    monkeypatch.setattr(bot_engine, "fetch_minute_df_safe", lambda symbol: inf_df.copy())

    if hasattr(bot_engine.CFG, "data_sanitize_enabled"):
        monkeypatch.setattr(bot_engine.CFG, "data_sanitize_enabled", False)
    if hasattr(bot_engine.S, "data_sanitize_enabled"):
        monkeypatch.setattr(bot_engine.S, "data_sanitize_enabled", False)
    if hasattr(bot_engine.CFG, "corp_actions_enabled"):
        monkeypatch.setattr(bot_engine.CFG, "corp_actions_enabled", False)
    if hasattr(bot_engine.S, "corp_actions_enabled"):
        monkeypatch.setattr(bot_engine.S, "corp_actions_enabled", False)

    caplog.set_level("DEBUG", logger="ai_trading.core.bot_engine")

    called = {"prepare": False}

    def _marker(frame):  # pragma: no cover - guard path
        called["prepare"] = True
        return frame

    monkeypatch.setattr(bot_engine, "prepare_indicators", _marker)

    ctx = types.SimpleNamespace(
        data_fetcher=types.SimpleNamespace(get_daily_df=lambda *_a, **_k: inf_df.copy()),
        halt_manager=None,
    )

    raw_df, feat_df, skip_flag = bot_engine._fetch_feature_data(ctx, None, "TEST")

    assert raw_df is not None
    pd.testing.assert_frame_equal(raw_df, inf_df)
    assert feat_df is None
    assert skip_flag is True
    assert not called["prepare"], "prepare_indicators should not be invoked when data is skipped"
    assert not any("MACD computation failed" in record.getMessage() for record in caplog.records)

