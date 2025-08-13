"""
Test-only shim for 'slippage' module used in CI.
Production code must NOT import this – it exists only for tests.
Provides a deterministic, zero-slippage default model.
"""

# AI-AGENT-REF: add test slippage module for deterministic behavior in CI
from dataclasses import dataclass


@dataclass(frozen=True)
class SlippageParams:
    bps: float = 0.0  # basis points
    fixed: float = 0.0  # absolute currency per order


class NullSlippageModel:
    """Deterministic zero/constant slippage for tests."""

    def __init__(self, params: SlippageParams | None = None) -> None:
        self.params = params or SlippageParams()

    def estimate(self, notional: float) -> float:
        # simple: bps * notional + fixed
        return (self.params.bps / 1_0000) * float(notional) + float(self.params.fixed)


def get_default_model() -> NullSlippageModel:
    """Return default null slippage model."""
    return NullSlippageModel()
