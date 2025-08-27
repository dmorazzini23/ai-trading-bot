import pytest
pd = pytest.importorskip("pandas")
from ai_trading.data.bars import _ensure_df

from tests.helpers.asserts import assert_df_like


def test_ensure_df_none_and_dict():
    assert _ensure_df(None).empty
    d = {"close": [1, 2, 3]}
    df = _ensure_df(d)
    assert isinstance(df, pd.DataFrame)
    assert_df_like(df)  # AI-AGENT-REF: allow empty in offline mode
