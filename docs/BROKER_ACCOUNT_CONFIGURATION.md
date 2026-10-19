# Broker Account Configuration Backstop

Alpaca account configurations are enforced by the broker, so they keep holding
even if a bug gets past the bot's own risk checks. The bot only **reads** them;
it never writes them. The owner sets them by hand.

## What the bot checks

`ai_trading/broker/account_configuration.py` derives the expected values from
the active launch profile (`launch_profile` in `/healthz`):

| Setting | Expected | What it backstops |
| --- | --- | --- |
| `no_shorting` | `true` when the profile has `shorts_allowed=false` (`TRADING__ALLOW_SHORTS=0`) | Alpaca rejects any sell that would open or grow a short position. A sizing or side bug cannot leave the account short. Long exits and covering existing shorts are unaffected. |
| `max_margin_multiplier` | `"1"` when `max_gross_exposure <= 1.0` (`"2"` up to 2.0, else `"4"`) | Buying power is capped at equity, so a bug cannot build leveraged gross exposure above the profile's 1.0x. |
| `dtbp_check`, `pdt_check` | `"entry"` or `"both"` when the API reports them | Day-trade buying-power and pattern-day-trader checks run before new entries. If the API does not report these fields they are listed as `unreported_fields`, not counted as drift. |
| `fractional_trading` | reported only | The runtime trades whole shares and has no fractional-sizing setting to derive an expectation from. |
| `trade_confirm_email` | ignored | Not a risk control. |

`suspend_trade` and `closing_transactions_only` are reported too. They work as
manual kill switches (no new orders, or exits only), but the check does not
expect either value.

## Where it shows up

- **Startup log:** after the first broker sync, the runtime logs
  `BROKER_ACCOUNT_CONFIGURATION_OK`, or `BROKER_ACCOUNT_CONFIGURATION_DRIFT` at
  WARNING with the mismatch list. If the API call fails it logs
  `BROKER_ACCOUNT_CONFIGURATION_UNAVAILABLE` and retries at most every 5
  minutes, until one read succeeds.
- **Health:** the `broker_account_configuration` section of `/healthz` has
  `ok`, `available`, `reason`, `mismatches`, `unreported_fields` and
  `fetched_at`. When there are mismatches, the attention flag
  `broker_account_configuration_drift` is added. Drift does **not** change
  `ok`, does not block trading, and does not gate readiness. An unavailable
  check gives `available=false` and no flag.
- **CLI (read-only):**

  ```bash
  ./venv/bin/python -m ai_trading.tools.broker_account_configuration --check
  ```

  It loads `./.env` (or `--env-file PATH`) and the managed secrets, then prints
  observed values, expected values and mismatches as JSON. Exit codes: `0`
  matching, `2` drift, `3` unavailable. It has no `--apply`.

## Setting it by hand

**Paper and live accounts are configured separately.** Each Alpaca account
(each paper account, and the live account) has its own configuration. Set the
account that matches the credentials and `ALPACA_TRADING_BASE_URL` you are
using, then run `--check` against that same account.

### Dashboard

Sign in to the Alpaca dashboard. Select the paper or live account, open the
account configuration settings, turn shorting off and set the margin
multiplier to 1. Then run `--check` and confirm exit code `0`.

### SDK (alpaca-py 0.42.1)

The typed call sends a full `AccountConfiguration`, so every field must be
supplied. Take the non-backstop values from the `observed` section of
`--check`:

```python
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import DTBPCheck, PDTCheck, TradeConfirmationEmail
from alpaca.trading.models import AccountConfiguration

client = TradingClient(API_KEY, SECRET_KEY, paper=True)  # paper=False for live
client.set_account_configurations(
    AccountConfiguration(
        no_shorting=True,
        max_margin_multiplier="1",
        dtbp_check=DTBPCheck.ENTRY,
        pdt_check=PDTCheck.ENTRY,
        fractional_trading=True,
        suspend_trade=False,
        trade_confirm_email=TradeConfirmationEmail.ALL,
        ptp_no_exception_entry=False,
    )
)
```

As of 2026-10-19 the paper API does not return `dtbp_check` or `pdt_check`.
alpaca-py 0.42.1 cannot parse that response into `AccountConfiguration`, so
`get_account_configurations()` fails, and the typed `set_account_configurations`
call may raise a `ValidationError` while parsing the response even after the
write succeeded. The bot's read path falls back to a raw GET through the same
client. To change only the backstop fields, use a partial PATCH through the
same client:

```python
client.patch(
    "/account/configurations",
    {"no_shorting": True, "max_margin_multiplier": "1"},
)
```

Neither write path has been run from this repository. Whichever you use, run
`--check` afterwards to confirm the result.
