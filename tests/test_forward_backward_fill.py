import pandas as pd
import pandas.testing as pdt


def test_ffill_bfill_equivalence():
    s = pd.Series([None, 1, None, None, 4, None], dtype=float)
    result = s.ffill().bfill()
    expected = pd.Series([1.0, 1.0, 1.0, 1.0, 4.0, 4.0])
    pdt.assert_series_equal(result, expected)
