"""capital_scaling.py

Utilities for adaptive capital allocation and risk-based position sizing.
"""


class _CapScaler:
    def __init__(self, params):
        self.params = params or {}

    def scale_position(self, value):
        """Scale a numeric position using params['x']."""  # AI-AGENT-REF: unify scaling logic
        return value * self.params.get("x", 1)


class CapitalScalingEngine:
    def __init__(self, params=None):
        # AI-AGENT-REF: allow optional params with sensible default
        self.params = params or {}
        self.scaler = _CapScaler(self.params)

    def scale_position(self, value):
        return self.scaler.scale_position(value)

    def update(self, ctx, equity_init):
        # Placeholder for future scaling logic
        pass


__all__ = ["CapitalScalingEngine"]
