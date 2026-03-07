"""Minimal pandas facade used by tests when ``pandas`` is absent.

This module provides the common attributes accessed throughout the code base so
import-time does not fail in environments where ``pandas`` is intentionally
missing. When real pandas is available, objects are forwarded.
"""

try:  # pragma: no cover - exercised in environments with pandas installed
    import pandas as _pd  # type: ignore

    _Series = _pd.Series
    _DataFrame = _pd.DataFrame
    _Index = _pd.Index
    _DatetimeIndex = _pd.DatetimeIndex
    _Timestamp = _pd.Timestamp
    _to_numeric = _pd.to_numeric
    _isna = _pd.isna
    _read_csv = _pd.read_csv
    __file__ = getattr(_pd, "__file__", __file__)
except Exception:  # pragma: no cover - stubbed fallback
    class _Stub:
        def __init__(self, *args, **kwargs):
            self._data = kwargs.get("data")

        def __bool__(self):  # avoid ValueError from ambiguous DataFrame truth
            return True

    class _Series(_Stub):
        pass

    class _DataFrame(_Stub):
        @property
        def empty(self) -> bool:
            return not bool(self._data)

    class _Index(_Stub):
        pass

    class _DatetimeIndex(_Index):
        pass

    class _Timestamp:
        def __init__(self, *args, **kwargs):
            pass

    def _to_numeric(x, *args, **kwargs):  # noqa: D401
        """Return the input unchanged (stub)."""
        return x

    def _isna(x):  # noqa: D401
        """Always return ``False`` (stub)."""
        return False

    def _read_csv(*args, **kwargs):  # noqa: D401
        """Return an empty :class:`DataFrame` (stub)."""
        return _DataFrame()

    __file__ = __file__

Series = _Series
DataFrame = _DataFrame
Index = _Index
DatetimeIndex = _DatetimeIndex
Timestamp = _Timestamp
to_numeric = _to_numeric
isna = _isna
read_csv = _read_csv


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
