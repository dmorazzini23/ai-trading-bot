"""capital_scaling.py

Utilities for adaptive capital allocation and risk-based position sizing.
"""


class _CapScaler:
    def scale_position(self, size):
        return size


class CapitalScalingEngine:
    def __init__(self, params=None):
        # AI-AGENT-REF: accept params but scaler no longer uses them
        self.params = params or {}
        self.scaler = _CapScaler()

    def scale_position(self, value):
        return self.scaler.scale_position(value)

    def update(self, ctx, equity_init):
        # Placeholder for future scaling logic
        pass


__all__ = ["CapitalScalingEngine"]
