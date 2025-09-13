"""Lightweight stand-in for pandas ``Series`` used in tests.

The real project uses ``pandas.Series`` extensively but the full dependency
is heavy to import in unit tests.  This minimal implementation provides only
the pieces of the interface that our tests exercise.  It purposely mimics the
behaviour of a subset of pandas so tests can perform simple computations
without pulling in the entire library.
"""
from __future__ import annotations

from math import isnan
from typing import Iterable, Iterator, List


class MockSeries:
    """Minimal Series-like object for testing without pandas."""

    def __init__(self, values: Iterable[float]):
        # Store a concrete list so operations like ``diff`` can iterate multiple
        # times over the data.  Values are kept as provided; callers are
        # responsible for ensuring numeric types where appropriate.
        self._values = list(values)

    # ------------------------------------------------------------------
    # Python protocol methods
    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._values)

    def __getitem__(self, idx):  # pragma: no cover - trivial
        return self._values[idx]

    def __iter__(self) -> Iterator[float]:  # pragma: no cover - trivial
        return iter(self._values)

    # ------------------------------------------------------------------
    # ``.iloc`` helper used by a few tests.  It intentionally only supports
    # basic indexing via ``__getitem__``.
    class _ILoc:
        def __init__(self, outer: "MockSeries") -> None:
            self._outer = outer

        def __getitem__(self, idx):  # pragma: no cover - simple
            return self._outer._values[idx]

    @property
    def iloc(self) -> "MockSeries._ILoc":  # pragma: no cover - simple
        return MockSeries._ILoc(self)

    # ------------------------------------------------------------------
    # Convenience helpers ------------------------------------------------
    def tail(self, n: int = 5) -> "MockSeries":  # pragma: no cover - trivial
        return MockSeries(self._values[-n:])

    def tolist(self) -> List[float]:  # pragma: no cover - trivial
        return list(self._values)

    # ------------------------------------------------------------------
    # Pandas-like numeric operations ------------------------------------
    def diff(self, periods: int = 1) -> "MockSeries":
        """Return the first discrete difference of the series.

        Only ``periods=1`` is implemented which covers the needs of the test
        suite.  The first value mirrors pandas' behaviour by being ``NaN``.
        """

        if periods != 1:  # pragma: no cover - defensive, not exercised
            raise NotImplementedError("MockSeries.diff only supports periods=1")

        if not self._values:
            return MockSeries([])

        diffs = [float("nan")]
        diffs.extend(self._values[i] - self._values[i - 1] for i in range(1, len(self._values)))
        return MockSeries(diffs)

    def isna(self) -> "MockSeries":
        """Boolean mask indicating ``NaN`` values.

        ``None`` is also treated as ``NaN`` to match the behaviour of
        ``pandas.isna``.
        """

        def _is_na(v: float) -> bool:
            if v is None:  # pragma: no cover - simple guard
                return True
            try:
                return isnan(v)
            except TypeError:  # pragma: no cover - non-numeric
                return False

        return MockSeries(_is_na(v) for v in self._values)

    # Boolean reductions -------------------------------------------------
    def any(self) -> bool:  # pragma: no cover - trivial
        return any(self._values)

    def all(self) -> bool:  # pragma: no cover - trivial
        return all(self._values)

    def sum(self) -> float:  # pragma: no cover - trivial
        return sum(self._values)


__all__ = ["MockSeries"]
