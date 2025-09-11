# ATR Stop Max Factor

The ATR-based stop helper scales take-profit targets using a maximum
factor. The default is `2.0` as defined in `ai_trading.config.scaling`.
Deployments may override this by setting the `TAKE_PROFIT_FACTOR`
environment variable:

```bash
export TAKE_PROFIT_FACTOR=3.0
```

Only positive numeric values are accepted. The parameter validator will
reject `None` values before trading starts.
