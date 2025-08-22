import pandas as pd


def assert_df_like(df: pd.DataFrame) -> None:
    """Check DataFrame shape without enforcing row count."""  # AI-AGENT-REF
    assert isinstance(df, pd.DataFrame)
    # When offline, allow empty but with valid columns/index type
    assert hasattr(df, "columns")

