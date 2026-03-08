from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pandas import DataFrame as PandasDataFrame
else:
    PandasDataFrame = Any

DataFrame = pytest.importorskip("pandas").DataFrame


def assert_df_like(df: PandasDataFrame, *, columns: list[str] | None = None) -> None:
    assert isinstance(df, DataFrame)
    if columns is not None:
        missing = [c for c in columns if c not in df.columns]
        assert not missing, f"missing columns: {missing}"
