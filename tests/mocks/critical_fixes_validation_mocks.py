# Extracted from scripts/critical_fixes_validation.py

class MockOrder:
    def __init__(self, filled_qty):
        self.filled_qty = filled_qty
        self.id = "test_order_123"

class MockContext:
    def __init__(self):
        self.api = None

class MockSignal:
    def __init__(self, symbol, side, confidence):
        self.symbol = symbol
        self.side = side
        self.confidence = confidence

class MockContextShortSelling:
    def __init__(self):
        self.api = None
        self.allow_short_selling = True  # Enable short selling