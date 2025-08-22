from pandas import DataFrame


def assert_df_like(df: DataFrame, *, columns: list[str] | None = None) -> None:
    assert isinstance(df, DataFrame)
    if columns is not None:
        missing = [c for c in columns if c not in df.columns]
        assert not missing, f"missing columns: {missing}"
