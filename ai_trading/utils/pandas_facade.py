"""Minimal pandas facade used by tests when ``pandas`` is absent.

This module provides the common attributes accessed throughout the code base so
import-time does not fail in environments where ``pandas`` is intentionally
missing. When real pandas is available, objects are forwarded.
"""

try:  # pragma: no cover - exercised in environments with pandas installed
    import pandas as _pd  # type: ignore

    Series = _pd.Series
    DataFrame = _pd.DataFrame
    Index = _pd.Index
    DatetimeIndex = _pd.DatetimeIndex
    Timestamp = _pd.Timestamp
    to_numeric = _pd.to_numeric
    isna = _pd.isna
    read_csv = _pd.read_csv
    __file__ = getattr(_pd, "__file__", __file__)
except Exception:  # pragma: no cover - stubbed fallback
    class _Stub:
        def __init__(self, *args, **kwargs):
            self._data = kwargs.get("data")

        def __bool__(self):  # avoid ValueError from ambiguous DataFrame truth
            return True

    class Series(_Stub):
        pass

    class DataFrame(_Stub):
        @property
        def empty(self) -> bool:
            return not bool(self._data)

    class Index(_Stub):
        pass

    class DatetimeIndex(Index):
        pass

    class Timestamp:
        def __init__(self, *args, **kwargs):
            pass

    def to_numeric(x, *args, **kwargs):  # noqa: D401
        """Return the input unchanged (stub)."""
        return x

    def isna(x):  # noqa: D401
        """Always return ``False`` (stub)."""
        return False

    def read_csv(*args, **kwargs):  # noqa: D401
        """Return an empty :class:`DataFrame` (stub)."""
        return DataFrame()

    __file__ = __file__


__all__ = [
    "Series",
    "DataFrame",
    "Index",
    "DatetimeIndex",
    "Timestamp",
    "to_numeric",
    "isna",
    "read_csv",
]

