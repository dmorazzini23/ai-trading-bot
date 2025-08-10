# tests/support/mocks_runtime.py
# Placeholders for any legacy Mock* classes moved out of runtime.
# Import and use within tests only.

"""
Mock classes that were previously used for import fallbacks.
These are now only for testing purposes.
"""

class MockNumpy:
    """Mock NumPy implementation for testing environments."""
    def __init__(self):
        # Constants
        self.nan = float("nan")
        self.inf = float("inf")
        self.pi = 3.141592653589793
        self.e = 2.718281828459045

        # Data types
        self.float64 = float
        self.int64 = int
        
    def array(self, *args, **kwargs):
        return list(args[0]) if args else []
        
    def mean(self, arr):
        return sum(arr) / len(arr) if arr else 0
        
    def std(self, arr):
        return 1.0  # Mock implementation
        
    def zeros(self, shape):
        if isinstance(shape, int):
            return [0] * shape
        return []


class MockPandas:
    """Mock Pandas implementation for testing environments."""
    def __init__(self):
        self.DataFrame = MockDataFrame
        self.Series = MockSeries
        
    def read_csv(self, *args, **kwargs):
        return MockDataFrame()


class MockDataFrame:
    """Mock DataFrame implementation."""
    def __init__(self, data=None):
        self.data = data or {}
        
    def __getitem__(self, key):
        return MockSeries()
        
    def __setitem__(self, key, value):
        pass
        
    def head(self, n=5):
        return self
        
    def tail(self, n=5):
        return self
        
    def dropna(self):
        return self


class MockSeries:
    """Mock Series implementation."""
    def __init__(self, data=None):
        self.data = data or []
        
    def __getitem__(self, key):
        return 0
        
    def mean(self):
        return 0.0
        
    def std(self):
        return 1.0


# Add more mock classes as needed when moving them from runtime code

class MockFinBERT:
    """Mock FinBERT model for testing."""
    
    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return lambda *args, **kwargs: self

    def tolist(self):
        return [0.33, 0.34, 0.33]  # neutral sentiment


class MockSklearn:
    """Mock scikit-learn for testing."""
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class MockTalib:
    """Mock TA-Lib for testing."""
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class MockPortalocker:
    """Mock portalocker for testing."""
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class MockSchedule:
    """Mock schedule for testing."""
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class MockYfinance:
    """Mock yfinance for testing."""
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: self
        
    def Ticker(self, symbol):
        return MockTicker()


class MockTicker:
    """Mock yfinance Ticker for testing."""
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class MockBeautifulSoup:
    """Mock BeautifulSoup for testing."""
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class MockFlask:
    """Mock Flask for testing."""
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class MockAlpacaClient:
    """Mock Alpaca trading client for testing."""
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class MockCircuitBreaker:
    """Mock circuit breaker for testing."""
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class MockMetric:
    """Mock prometheus metric for testing."""
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class MockIndex:
    """Mock pandas Index for testing."""
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class MockRolling:
    """Mock pandas rolling for testing."""
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class MockTradingClient:
    """Mock Alpaca trading client for testing."""
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class MockMarketOrderRequest:
    """Mock Alpaca market order request for testing."""
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class MockLimitOrderRequest:
    """Mock Alpaca limit order request for testing."""
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class MockGetOrdersRequest:
    """Mock Alpaca get orders request for testing."""
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class MockOrder:
    """Mock Alpaca order for testing."""
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class MockTradingStream:
    """Mock Alpaca trading stream for testing."""
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class MockStockHistoricalDataClient:
    """Mock Alpaca stock historical data client for testing."""
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class MockQuote:
    """Mock Alpaca quote for testing."""
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class MockStockBarsRequest:
    """Mock Alpaca stock bars request for testing."""
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class MockStockLatestQuoteRequest:
    """Mock Alpaca stock latest quote request for testing."""
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class MockTimeFrame:
    """Mock Alpaca time frame for testing."""
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: self