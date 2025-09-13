"""Stub TimeFrame and TimeFrameUnit for tests."""
from dataclasses import dataclass
from enum import Enum

class TimeFrameUnit(Enum):
    Minute = "Minute"
    Hour = "Hour"
    Day = "Day"
    Week = "Week"
    Month = "Month"

@dataclass(frozen=True)
class TimeFrame:
    amount: int = 1
    unit: TimeFrameUnit = TimeFrameUnit.Day

    def __str__(self) -> str:
        return f"{self.amount}{self.unit.name}"


# Pre-defined shorthand attributes mirroring alpaca-py
TimeFrame.Minute = TimeFrame(1, TimeFrameUnit.Minute)  # type: ignore[attr-defined]
TimeFrame.Hour = TimeFrame(1, TimeFrameUnit.Hour)  # type: ignore[attr-defined]
TimeFrame.Day = TimeFrame()  # type: ignore[attr-defined]
TimeFrame.Week = TimeFrame(1, TimeFrameUnit.Week)  # type: ignore[attr-defined]
TimeFrame.Month = TimeFrame(1, TimeFrameUnit.Month)  # type: ignore[attr-defined]

__all__ = ["TimeFrame", "TimeFrameUnit"]
