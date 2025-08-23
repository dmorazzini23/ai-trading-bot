from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
from collections.abc import Mapping


class Allocator(Protocol):
    def allocate(self, cash: float, prices: Mapping[str, float]) -> Mapping[str, float]: ...


@dataclass(frozen=True)
class Allocation:
    quantities: Mapping[str, float]
    cash_left: float = 0.0
