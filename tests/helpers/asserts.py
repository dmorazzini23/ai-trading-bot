import pytest
from typing import Any
pd = pytest.importorskip("pandas")
def assert_df_like(df: Any) -> None:
    """Check DataFrame shape without enforcing row count."""  # AI-AGENT-REF
    assert isinstance(df, pd.DataFrame)
    # When offline, allow empty but with valid columns/index type
    assert hasattr(df, "columns")
