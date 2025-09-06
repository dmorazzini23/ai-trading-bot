"""Stub TimeFrame and TimeFrameUnit for tests."""
from dataclasses import dataclass
from enum import Enum

class TimeFrameUnit(Enum):
    Minute = "Minute"
    Hour = "Hour"
    Day = "Day"
    Week = "Week"

@dataclass(frozen=True)
class TimeFrame:
    amount: int = 1
    unit: TimeFrameUnit = TimeFrameUnit.Day

__all__ = ["TimeFrame", "TimeFrameUnit"]
