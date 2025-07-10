# Strategies Overview

This folder contains example strategies used by the trading bot. Each strategy implements
a simple `generate` method that returns buy or sell signals from a price series.

## Using drawdown_adjusted_kelly

The `ai_trading.capital_scaling` module provides a `drawdown_adjusted_kelly` helper which scales
your raw Kelly fraction based on current account drawdown:

```python
from ai_trading.capital_scaling import drawdown_adjusted_kelly

kelly_frac = 0.6
account_value = 9000
peak_equity = 10000
size_factor = drawdown_adjusted_kelly(account_value, peak_equity, kelly_frac)
```

This returns a reduced position size factor when the account is below its peak.
