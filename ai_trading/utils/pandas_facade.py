"""Minimal pandas facade used by tests when ``pandas`` is absent.

This module provides the common attributes accessed throughout the code base so
import-time does not fail in environments where ``pandas`` is intentionally
missing. When real pandas is available, objects are forwarded.
"""

from typing import Any, Callable

Series: Any
DataFrame: Any
Index: Any
DatetimeIndex: Any
Timestamp: Any
to_numeric: Callable[..., Any]
isna: Callable[..., Any]
read_csv: Callable[..., Any]

try:  # pragma: no cover - exercised in environments with pandas installed
    import pandas as _pd  # type: ignore
except Exception:  # pragma: no cover - stubbed fallback
    _pd = None

if _pd is None:
    class _Stub:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._data = kwargs.get("data")

        def __bool__(self) -> bool:  # avoid ValueError from ambiguous DataFrame truth
            return True

    class _SeriesStub(_Stub):
        pass

    class _DataFrameStub(_Stub):
        @property
        def empty(self) -> bool:
            return not bool(self._data)

    class _IndexStub(_Stub):
        pass

    class _DatetimeIndexStub(_IndexStub):
        pass

    class _TimestampStub:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    def to_numeric(x: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: D401
        """Return the input unchanged (stub)."""
        return x

    def isna(x: Any) -> bool:  # noqa: D401
        """Always return ``False`` (stub)."""
        return False

    def read_csv(*args: Any, **kwargs: Any) -> _DataFrameStub:  # noqa: D401
        """Return an empty :class:`DataFrame` (stub)."""
        return _DataFrameStub()

    Series = _SeriesStub
    DataFrame = _DataFrameStub
    Index = _IndexStub
    DatetimeIndex = _DatetimeIndexStub
    Timestamp = _TimestampStub

    __file__ = __file__
else:
    class _Fallback:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    Series = getattr(_pd, "Series", _Fallback)
    DataFrame = getattr(_pd, "DataFrame", _Fallback)
    Index = getattr(_pd, "Index", _Fallback)
    DatetimeIndex = getattr(_pd, "DatetimeIndex", Index)
    Timestamp = getattr(_pd, "Timestamp", _Fallback)
    to_numeric = getattr(_pd, "to_numeric", lambda x, *args, **kwargs: x)
    isna = getattr(_pd, "isna", lambda _x: False)
    read_csv = getattr(_pd, "read_csv", lambda *args, **kwargs: DataFrame())
    __file__ = getattr(_pd, "__file__", __file__)


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
