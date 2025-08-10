from __future__ import annotations

from dataclasses import dataclass


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
