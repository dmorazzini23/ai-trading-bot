# Mock for yfinance
def download(*args, **kwargs):
    import pandas
    return pandas.DataFrame()

def Ticker(*args):
    class MockTicker:
        def history(self, *args, **kwargs):
            import pandas
            return pandas.DataFrame()
    return MockTicker()

def __getattr__(name):
    return lambda *args, **kwargs: None
