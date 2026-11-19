
import pytest

pd = pytest.importorskip("pandas")

from ai_trading.features import compute_macd


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

