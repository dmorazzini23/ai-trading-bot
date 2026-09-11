import math

import pytest

from ai_trading.tools.adjustment_diagnostics import diagnose, factor_interval


def sample():
    bars = {}
    for side, ts in (("before", "2025-12-04T20:59:00+00:00"), ("after", "2025-12-05T20:59:00+00:00")):
        bars[side] = {}
        for adj in ("raw", "split", "all"):
            price = 100 if adj == "raw" and side == "before" else 50
            bars[side][adj] = {"request": {"start": ts, "adjustment": adj},
                               "response": {"bars": [{"t": ts, "c": price}]}}
    return {"kind": "forward_splits", "event": {"symbol": "XLE", "ex_date": "2025-12-05", "old_rate": 1, "new_rate": 2}, "bars": bars}


def test_split_matches_and_wrong_factor_fails():
    value = sample()
    assert diagnose(value)["consistent"]
    value["event"]["new_rate"] = 3
    assert not diagnose(value)["consistent"]


@pytest.mark.parametrize("bad", [0, -1, math.nan, math.inf])
def test_bad_prices_rejected(bad):
    with pytest.raises(ValueError, match="invalid adjustment price"):
        factor_interval(bad, 100, 50, 50)


def test_holdout_rejected_before_prices():
    with pytest.raises(ValueError, match="outside development"):
        diagnose({"event": {"ex_date": "2026-09-09"}})


def test_missing_or_wrong_timestamp_rejected():
    value = sample()
    value["bars"]["before"]["raw"]["response"]["bars"][0]["t"] = "2025-12-04T20:58:00Z"
    with pytest.raises(ValueError, match="timestamp mismatch"):
        diagnose(value)


def test_dividend_matches():
    value = sample()
    value["kind"] = "cash_dividends"
    value["event"]["rate"] = 1
    for side in value["bars"].values():
        for item in side.values():
            item["response"]["bars"][0]["c"] = 100
    value["bars"]["before"]["all"]["response"]["bars"][0]["c"] = 99
    assert diagnose(value)["consistent"]
