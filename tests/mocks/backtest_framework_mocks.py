# Extracted from scripts/backtest_framework.py
# Keep class names/methods identical for test imports.

class MockSignal:
    def __init__(self):
        self.symbol = "AAPL"
        self.confidence = 0.8
        self.side = "buy"
