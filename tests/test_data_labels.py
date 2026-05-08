from __future__ import annotations

import numpy as np
import pytest

from ai_trading.data.labels import (
    fixed_horizon_return,
    get_daily_vol,
    trade_quality_labels,
    triple_barrier_labels,
)

pd = pytest.importorskip("pandas")


def test_fixed_horizon_return_series_applies_fees_and_name() -> None:
    prices = pd.Series([100.0, 110.0, 121.0], index=pd.date_range("2026-01-01", periods=3, freq="D"))
    out = fixed_horizon_return(prices, horizon_bars=1, fee_bps=10.0)
    expected = np.log(110.0 / 100.0) - 2 * (10.0 / 10000.0)
    assert out.iloc[0] == pytest.approx(expected)
    assert out.name == "future_return_h1_fee10.0bps"
    assert np.isnan(out.iloc[-1])


def test_fixed_horizon_return_dataframe_column_selection() -> None:
    idx = pd.date_range("2026-01-01", periods=4, freq="D")
    close_df = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0], "price": [10.0, 10.0, 10.0, 10.0]}, index=idx)
    price_df = pd.DataFrame({"price": [200.0, 202.0, 204.0, 206.0]}, index=idx)
    first_col_df = pd.DataFrame({"x": [300.0, 303.0, 306.0, 309.0]}, index=idx)

    close_out = fixed_horizon_return(close_df, horizon_bars=1)
    price_out = fixed_horizon_return(price_df, horizon_bars=1)
    first_col_out = fixed_horizon_return(first_col_df, horizon_bars=1)

    assert close_out.iloc[0] == pytest.approx(np.log(101.0 / 100.0))
    assert price_out.iloc[0] == pytest.approx(np.log(202.0 / 200.0))
    assert first_col_out.iloc[0] == pytest.approx(np.log(303.0 / 300.0))


def test_fixed_horizon_return_error_path_returns_empty_series() -> None:
    bad_prices = pd.Series(["a", "b", "c"], index=pd.date_range("2026-01-01", periods=3, freq="D"))
    out = fixed_horizon_return(bad_prices, horizon_bars=1)
    assert out.empty


def test_fixed_horizon_return_rejects_non_positive_horizon() -> None:
    prices = pd.Series([100.0, 101.0], index=pd.date_range("2026-01-01", periods=2, freq="D"))
    with pytest.raises(ValueError, match="horizon_bars must be positive"):
        fixed_horizon_return(prices, horizon_bars=0)


def test_triple_barrier_labels_hits_profit_loss_and_timeout() -> None:
    idx = pd.date_range("2026-01-01 09:30:00", periods=6, freq="min")
    prices = pd.Series([100.0, 103.0, 96.0, 100.0, 101.0, 101.0], index=idx)
    events = pd.DataFrame(index=[idx[0], idx[1], idx[3]])
    barrier_t1 = pd.Series([idx[4], idx[4], idx[5]], index=events.index)

    out = triple_barrier_labels(prices, events=events, pt_sl=(0.02, -0.02), t1=barrier_t1, min_ret=0.0)
    assert len(out) == 3
    assert set(out["bin"]) == {1, -1, 0}
    assert out.iloc[0]["bin"] == 1
    assert out.iloc[1]["bin"] == -1
    assert out.iloc[2]["bin"] == 0


def test_triple_barrier_labels_vertical_barrier_and_min_ret_filter() -> None:
    idx = pd.date_range("2026-01-01 09:30:00", periods=5, freq="min")
    prices = pd.Series([100.0, 103.0, 103.0, 103.0, 103.0], index=idx)
    events = pd.DataFrame(index=[idx[0], idx[1]])
    vertical = pd.Series([idx[3], idx[4]], index=events.index)

    out = triple_barrier_labels(
        prices,
        events=events,
        pt_sl=(0.05, -0.05),
        vertical_barrier_times=vertical,
        min_ret=0.02,
    )
    assert len(out) == 1
    assert (out["ret"].abs() >= 0.02).all()


def test_triple_barrier_labels_accepts_dataframe_input() -> None:
    idx = pd.date_range("2026-01-01", periods=4, freq="D")
    prices_df = pd.DataFrame({"close": [100.0, 102.5, 101.0, 103.0]}, index=idx)
    events = pd.DataFrame(index=[idx[0], idx[1]])
    t1 = pd.Series([idx[2], idx[3]], index=events.index)
    out = triple_barrier_labels(prices_df, events=events, pt_sl=(0.01, -0.01), t1=t1)
    assert not out.empty
    assert {"t1", "ret", "bin"} <= set(out.columns)


def test_triple_barrier_labels_preserves_event_index_when_events_skipped() -> None:
    idx = pd.date_range("2026-01-01 09:30:00", periods=5, freq="min")
    missing_event = idx[0] - pd.Timedelta(minutes=1)
    prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=idx)
    events = pd.DataFrame(index=[missing_event, idx[1], idx[2]])
    t1 = pd.Series([idx[2], idx[3], idx[4]], index=events.index)

    out = triple_barrier_labels(prices, events=events, pt_sl=(0.01, -0.01), t1=t1)

    assert out.index.tolist() == [idx[1], idx[2]]


def test_triple_barrier_labels_empty_result_has_canonical_columns() -> None:
    idx = pd.date_range("2026-01-01", periods=3, freq="D")
    prices = pd.Series([100.0, 100.5, 100.75], index=idx)
    events = pd.DataFrame(index=[idx[0]])
    t1 = pd.Series([idx[1]], index=events.index)

    out = triple_barrier_labels(prices, events=events, pt_sl=(0.10, -0.10), t1=t1, min_ret=0.05)

    assert out.empty
    assert list(out.columns) == ["t1", "ret", "bin"]


def test_get_daily_vol_returns_expected_indexed_series() -> None:
    idx = pd.date_range("2025-01-01", periods=140, freq="D")
    prices = pd.Series(np.linspace(100.0, 150.0, len(idx)), index=idx)
    vol = get_daily_vol(prices, span0=20)
    assert not vol.empty
    assert vol.index.is_monotonic_increasing
    assert float(vol.dropna().iloc[-1]) >= 0.0


def test_trade_quality_labels_multi_horizon_costs_and_end_timestamps() -> None:
    idx = pd.date_range("2026-01-01 09:30:00", periods=5, freq="min")
    prices = pd.DataFrame(
        {
            "close": [100.0, 101.0, 99.0, 104.0, 103.0],
            "high": [100.5, 102.0, 100.0, 105.0, 104.0],
            "low": [99.5, 100.5, 98.0, 103.0, 102.0],
            "spread_bps": [1.0, 1.5, 2.0, 2.5, 3.0],
        },
        index=idx,
    )

    out = trade_quality_labels(
        prices,
        horizons=[1, 3],
        slippage_bps=2.0,
        fee_bps=1.0,
        multiclass_edge_threshold_bps=25.0,
    )

    first_h1 = out[(out["label_start_timestamp"] == idx[0]) & (out["horizon_bars"] == 1)].iloc[0]
    first_h3 = out[(out["label_start_timestamp"] == idx[0]) & (out["horizon_bars"] == 3)].iloc[0]
    assert first_h1["label_end_timestamp"] == idx[1]
    assert first_h3["label_end_timestamp"] == idx[3]
    assert first_h1["round_trip_cost_bps"] == pytest.approx(7.0)
    assert first_h1["gross_return_bps"] == pytest.approx(100.0)
    assert first_h1["net_edge_after_cost_bps"] == pytest.approx(93.0)
    assert first_h1["mae_bps"] == pytest.approx(50.0)
    assert first_h1["mfe_bps"] == pytest.approx(200.0)
    assert first_h1["quality_binary"] == 1
    assert first_h1["quality_multiclass"] == 1
    assert out["label_end_timestamp"].notna().all()
    assert not ((out["label_start_timestamp"] == idx[-1]) & (out["horizon_bars"] == 1)).any()


def test_trade_quality_labels_short_side_and_negative_quality() -> None:
    idx = pd.date_range("2026-01-01 09:30:00", periods=3, freq="min")
    prices = pd.DataFrame(
        {
            "close": [100.0, 102.0, 103.0],
            "high": [100.5, 103.0, 104.0],
            "low": [99.5, 101.0, 102.0],
        },
        index=idx,
    )

    out = trade_quality_labels(
        prices,
        horizons=1,
        spread_bps=1.0,
        slippage_bps=1.0,
        fee_bps=1.0,
        side="short",
        multiclass_edge_threshold_bps=25.0,
    )

    first = out.iloc[0]
    assert first["gross_return_bps"] == pytest.approx(-200.0)
    assert first["net_edge_after_cost_bps"] == pytest.approx(-205.0)
    assert first["mae_bps"] == pytest.approx(-300.0)
    assert first["mfe_bps"] == pytest.approx(-100.0)
    assert first["risk_adjusted_return_bps"] == pytest.approx(-505.0)
    assert first["quality_binary"] == 0
    assert first["quality_multiclass"] == -1
