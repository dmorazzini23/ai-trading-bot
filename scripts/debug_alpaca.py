import os

from alpaca_trade_api import REST

api = REST(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    base_url="https://paper-api.alpaca.markets",
)

bars = api.get_bars(["SPY"], "1Day", limit=5).df
print(bars.head())
