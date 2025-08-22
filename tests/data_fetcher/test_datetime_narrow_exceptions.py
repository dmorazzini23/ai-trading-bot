import pandas as pd
import pytest
from ai_trading.data_fetcher import ensure_datetime


def test_dt_invalid_raises_typeerror():
    with pytest.raises(TypeError):
        ensure_datetime(object())


def test_dt_oob_raises_typeerror():
    def bad_ts():
        raise pd.errors.OutOfBoundsDatetime("bad")

    with pytest.raises(TypeError):
        ensure_datetime(bad_ts)
