# Risk Engine

## ATR Fallback

`RiskEngine._get_atr_data` normally retrieves historical bars via a data
client's `get_bars` method. When this method is unavailable, the engine now
falls back to existing OHLC data on the runtime context. If `ctx.minute_data`
or `ctx.daily_data` contain the required arrays, the ATR is derived directly
from those values. This allows volatility checks without a network data client.
