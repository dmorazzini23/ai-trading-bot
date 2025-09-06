"""Lightweight stand-in for pandas Series used in tests."""
from __future__ import annotations

from typing import Iterable, List


class MockSeries:
    """Minimal Series-like object for testing without pandas."""

    def __init__(self, values: Iterable[float]):
        self._values = list(values)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._values)

    def __getitem__(self, idx):  # pragma: no cover - trivial
        return self._values[idx]

    class _ILoc:
        def __init__(self, outer: "MockSeries") -> None:
            self._outer = outer

        def __getitem__(self, idx):  # pragma: no cover - simple
            return self._outer._values[idx]

    @property
    def iloc(self) -> "MockSeries._ILoc":  # pragma: no cover - simple
        return MockSeries._ILoc(self)

    def tail(self, n: int = 5) -> "MockSeries":  # pragma: no cover - trivial
        return MockSeries(self._values[-n:])

    def tolist(self) -> List[float]:  # pragma: no cover - trivial
        return list(self._values)

    def diff(self, *_args, **_kwargs) -> "MockSeries":
        """Placeholder diff returning self."""
        return self


__all__ = ["MockSeries"]
