import pandas as pd

from ai_trading.core.bot_engine import _ensure_df


def test_ensure_df_none_and_dict():
    assert _ensure_df(None).empty
    d = {"close": [1, 2, 3]}
    df = _ensure_df(d)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

