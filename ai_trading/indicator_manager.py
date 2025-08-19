from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class StreamingSMA:
    def __init__(self, period: int):
        self.p = int(period)
        self.q = []
        self.s = 0.0

    def update(self, x: float) -> float:
        x = float(x)
        self.q.append(x)
        self.s += x
        if len(self.q) > self.p:
            self.s -= self.q.pop(0)
        n = min(len(self.q), self.p)
        return self.s / max(n, 1)


class StreamingEMA:
    def __init__(self, period: int):
        self.a = 2.0 / (period + 1.0)
        self.e: float | None = None

    def update(self, x: float) -> float:
        x = float(x)
        self.e = x if self.e is None else (x - self.e) * self.a + self.e
        return self.e


class StreamingRSI:
    def __init__(self, period: int = 14):
        self.up = StreamingEMA(period)
        self.dn = StreamingEMA(period)
        self.prev: float | None = None

    def update(self, x: float) -> float:
        x = float(x)
        if self.prev is None:
            self.prev = x
            return 50.0
        d = x - self.prev
        self.prev = x
        au = self.up.update(max(d, 0.0))
        ad = self.dn.update(max(-d, 0.0))
        rs = au / (ad + 1e-12)
        return 100.0 - (100.0 / (1.0 + rs))


@dataclass
class IndicatorSpec:
    kind: str  # "sma" | "ema" | "rsi"
    period: int


class IndicatorManager:
    def __init__(self):
        self._ind: dict[str, object] = {}

    def add(self, name: str, spec: IndicatorSpec) -> None:
        if spec.kind == "sma":
            self._ind[name] = StreamingSMA(spec.period)
        elif spec.kind == "ema":
            self._ind[name] = StreamingEMA(spec.period)
        elif spec.kind == "rsi":
            self._ind[name] = StreamingRSI(spec.period)
        else:
            raise ValueError(f"Unknown indicator kind: {spec.kind}")

    def update(self, price: float) -> dict[str, float]:
        return {k: v.update(price) for k, v in self._ind.items()}


# AI-AGENT-REF: legacy indicators for tests
class IndicatorType(str, Enum):
    SMA = "SMA"


class CircularBuffer:
    def __init__(self, maxsize: int, dtype=float):
        self.maxsize = int(maxsize)
        self.dtype = dtype
        self._buf: list[float] = []

    def append(self, x: float) -> None:
        self._buf.append(self.dtype(x))
        if len(self._buf) > self.maxsize:
            self._buf.pop(0)

    def get_array(self) -> list[float]:
        return list(self._buf)

    def size(self) -> int:
        return len(self._buf)


class IncrementalSMA:
    def __init__(self, window: int, name: str):
        self.window = int(window)
        self.name = name
        self.buf = CircularBuffer(window)

    def update(self, x: float) -> float | None:
        self.buf.append(x)
        if self.buf.size() < self.window:
            return None
        return sum(self.buf.get_array()[-self.window:]) / self.window


class IncrementalEMA:
    def __init__(self, window: int, name: str):
        self.alpha = 2.0 / (window + 1.0)
        self.name = name
        self.last_value: float | None = None
        self.is_initialized = False

    def update(self, x: float) -> float:
        if self.last_value is None:
            self.last_value = x
        else:
            self.last_value = self.last_value + self.alpha * (x - self.last_value)
        self.is_initialized = True
        return self.last_value


class IncrementalRSI:
    def __init__(self, period: int, name: str):
        self.name = name
        self.period = int(period)
        self.up = IncrementalEMA(period, name + "_up")
        self.down = IncrementalEMA(period, name + "_down")
        self.prev: float | None = None
        self.last_value = 0.0
        self.is_initialized = False

    def update(self, x: float) -> float | None:
        x = float(x)
        if self.prev is None:
            self.prev = x
            return None
        change = x - self.prev
        self.prev = x
        up = max(change, 0.0)
        down = max(-change, 0.0)
        avg_up = self.up.update(up)
        avg_down = self.down.update(down)
        rs = avg_up / (avg_down + 1e-12)
        self.last_value = 100.0 - (100.0 / (1.0 + rs))
        self.is_initialized = True
        return self.last_value


__all__ = [
    "StreamingSMA",
    "StreamingEMA",
    "StreamingRSI",
    "IndicatorSpec",
    "IndicatorManager",
    "IndicatorType",
    "CircularBuffer",
    "IncrementalSMA",
    "IncrementalEMA",
    "IncrementalRSI",
]
