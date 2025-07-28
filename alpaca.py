# Mock for alpaca
from types import ModuleType

trading = ModuleType('trading')
data = ModuleType('data')

# Trading client mock
class TradingClient:
    def __init__(self, *args, **kwargs):
        pass
    def get_account(self):
        return {'equity': 100000, 'buying_power': 50000}
    def get_positions(self):
        return []
    def submit_order(self, *args, **kwargs):
        return {'id': 'mock_order'}

# Data client mock  
class StockHistoricalDataClient:
    def __init__(self, *args, **kwargs):
        pass
    def get_stock_bars(self, *args, **kwargs):
        return []

class MarketDataRequest:
    def __init__(self, *args, **kwargs):
        pass

class OrderRequest:
    def __init__(self, *args, **kwargs):
        pass

trading.client = ModuleType('client')
trading.client.TradingClient = TradingClient
trading.requests = ModuleType('requests')
trading.requests.MarketDataRequest = MarketDataRequest
trading.requests.OrderRequest = OrderRequest

data.historical = ModuleType('historical')
data.historical.StockHistoricalDataClient = StockHistoricalDataClient

globals()['trading'] = trading
globals()['data'] = data

def __getattr__(name):
    return lambda *args, **kwargs: None
