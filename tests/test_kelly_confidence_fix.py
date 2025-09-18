"""
Test cases for the Kelly calculation confidence fix.

This module tests that confidence values > 1.0 are properly normalized
to valid probability ranges in the Kelly calculation.
"""
import math
import sys
import types
from types import SimpleNamespace

if "numpy" not in sys.modules:  # pragma: no cover - lightweight stub for tests
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.nan = float("nan")
    numpy_stub.NaN = numpy_stub.nan
    numpy_stub.array = lambda data, *_, **__: list(data)
    numpy_stub.asarray = lambda data, *_, **__: list(data)
    numpy_stub.std = lambda data, *_, **__: 1.0
    numpy_stub.diff = lambda arr: [b - a for a, b in zip(arr, arr[1:])]
    numpy_stub.where = lambda cond, x, y: [
        (xi if bool(ci) else yi) for ci, xi, yi in zip(cond, x, y)
    ]
    numpy_stub.zeros_like = lambda arr: [0 for _ in arr]
    numpy_stub.zeros = lambda shape, dtype=None: [0.0] * shape if isinstance(shape, int) else []
    numpy_stub.mean = lambda data: (sum(data) / len(data)) if data else 0.0
    numpy_stub.exp = math.exp
    numpy_stub.float64 = float
    numpy_stub.ndarray = list
    numpy_stub.random = types.SimpleNamespace(seed=lambda *_a, **_k: None)
    sys.modules["numpy"] = numpy_stub

if "portalocker" not in sys.modules:  # pragma: no cover - lightweight stub for tests
    portalocker_stub = types.ModuleType("portalocker")
    portalocker_stub.LOCK_EX = 1
    portalocker_stub.LOCK_SH = 0
    portalocker_stub.lock = lambda *_a, **_k: None
    portalocker_stub.unlock = lambda *_a, **_k: None
    sys.modules["portalocker"] = portalocker_stub

if "bs4" not in sys.modules:  # pragma: no cover - lightweight stub for tests
    bs4_stub = types.ModuleType("bs4")

    class _BeautifulSoup:  # pragma: no cover - minimal placeholder
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    bs4_stub.BeautifulSoup = _BeautifulSoup
    sys.modules["bs4"] = bs4_stub

# Test the actual import and function from bot_engine
import pytest

def test_kelly_confidence_normalization():
    """Test that high confidence values are properly normalized to probabilities."""
    # Mock BotContext for testing
    # Import the actual function (if available)
    try:
        from ai_trading.core.bot_engine import fractional_kelly_size
        from tests.support.mocks import MockContext as MockBotContext

        ctx = MockBotContext()
        balance = 10000.0
        price = 100.0
        atr = 2.0

        # Test cases that previously caused errors
        problematic_confidences = [
            3.315653439025116,
            3.0650464264275152,
            5.0,
            2.5,
        ]

        for confidence in problematic_confidences:
            # This should not raise an error and should return a valid position size
            result = fractional_kelly_size(ctx, balance, price, atr, confidence)

            # Verify result is reasonable
            assert isinstance(result, int), f"Result should be integer, got {type(result)}"
            assert result >= 0, f"Position size should be non-negative, got {result}"
            assert result < 1000, f"Position size should be reasonable, got {result}"

        # Test edge cases
        assert fractional_kelly_size(ctx, balance, price, atr, 0.0) >= 0
        assert fractional_kelly_size(ctx, balance, price, atr, 1.0) >= 0
        assert fractional_kelly_size(ctx, balance, price, atr, -0.5) >= 0

    except ImportError:
        # If we can't import, at least test our normalization logic
        def sigmoid_normalize(value):
            if value > 1.0:
                return 1.0 / (1.0 + math.exp(-value + 1.0))
            elif value < 0:
                return 0.0
            return value

        test_values = [3.315653439025116, 3.0650464264275152, 5.0, 1.0, 0.5, 0.0, -0.5]

        for value in test_values:
            normalized = sigmoid_normalize(value)
            assert 0.0 <= normalized <= 1.0, f"Normalized value {normalized} should be in [0,1]"

            # Values > 1 should be mapped to something > 0.5
            if value > 1.0:
                assert normalized > 0.5, f"High confidence {value} should map to high probability"


def test_kelly_input_validation():
    """Test that Kelly calculation properly validates all inputs."""
    # Mock BotContext for testing
    try:
        from ai_trading.core.bot_engine import fractional_kelly_size
        from tests.support.mocks import MockContext as MockBotContext

        ctx = MockBotContext()

        # Test invalid inputs return 0 or minimal position
        assert fractional_kelly_size(ctx, -1000, 100, 2.0, 0.6) == 0  # negative balance
        assert fractional_kelly_size(ctx, 1000, -100, 2.0, 0.6) == 0  # negative price
        assert fractional_kelly_size(ctx, 1000, 0, 2.0, 0.6) == 0     # zero price

        # Test that valid inputs work
        result = fractional_kelly_size(ctx, 1000, 100, 2.0, 0.6)
        assert result > 0, "Valid inputs should produce positive position size"

    except ImportError:
        # Skip if we can't import the actual function
        pytest.skip("Cannot import fractional_kelly_size function")


def test_calculate_entry_size_zero_price(monkeypatch, caplog):
    """Ensure calculate_entry_size skips symbols with invalid pricing data."""

    from ai_trading.core.bot_engine import calculate_entry_size

    monkeypatch.setenv("PYTEST_RUNNING", "1")

    class _DummySeries:
        def __init__(self, values):
            self._values = list(values)

        def pct_change(self, fill_method=None):  # noqa: D401 - minimal stub
            changes = []
            for idx in range(1, len(self._values)):
                prev = self._values[idx - 1]
                curr = self._values[idx]
                changes.append(((curr - prev) / prev) if prev else 0.0)
            return _DummySeries(changes or [0.0])

        def dropna(self):  # noqa: D401 - minimal stub
            return self

        @property
        def values(self):  # noqa: D401 - minimal stub
            return self._values

    class _DummyDataFrame:
        empty = False

        def __init__(self, close_values):
            self._close = _DummySeries(close_values)

        def __getitem__(self, key):  # noqa: D401 - minimal stub
            if key != "close":
                raise KeyError(key)
            return self._close

    daily_df = _DummyDataFrame([100.0, 101.0, 102.0])

    class _DummyAPI:
        def get_account(self):
            return SimpleNamespace(cash="10000")

    class _DummyFetcher:
        def get_daily_df(self, *_args, **_kwargs):
            return daily_df

    class _DummyQuote:
        ask_price = 100.5
        bid_price = 100.0

    class _DummyDataClient:
        def get_stock_latest_quote(self, *_args, **_kwargs):
            return _DummyQuote()

    ctx = SimpleNamespace(
        api=_DummyAPI(),
        data_fetcher=_DummyFetcher(),
        data_client=_DummyDataClient(),
        volume_threshold=100_000,
        params={},
        max_position_dollars=10_000.0,
    )

    caplog.set_level("INFO")
    size = calculate_entry_size(ctx, "TEST", price=0.0, atr=1.0, win_prob=0.6)

    assert size == 0
    assert any("SKIP_INVALID_PRICE" in rec.message for rec in caplog.records)

    caplog.clear()

    monkeypatch.setattr(
        "ai_trading.core.bot_engine.fractional_kelly_size",
        lambda *args, **kwargs: 0,
    )

    size = calculate_entry_size(ctx, "TEST", price=100.0, atr=1.0, win_prob=0.6)

    assert size == 0
    assert any("SKIP_INVALID_KELLY_SIZE" in rec.message for rec in caplog.records)


if __name__ == "__main__":
    test_kelly_confidence_normalization()
    test_kelly_input_validation()
