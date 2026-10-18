# Day-sleeve horizon trial: inconclusive, budget consumed (October 18, 2026)

Campaign `day_sleeve_horizon_trial_20261018` ran its single trial on
October 18, 2026 at 04:57 UTC from a closed, flat window, under the
[reset amendment](RESEARCH_RESET_AMENDMENT_20261018.md). Outcome:
**`inconclusive_budget_consumed`**. The hypothesis
`fixed_histgb_day_sleeve_60_minute_horizon_after_costs_v1` is retired. No model
artifact was saved, no registry entry was created, no gate was changed, and the
service was not touched. The live stale-model readiness failure remains.

## Result against the frozen acceptance criteria

| Criterion (contract) | Required | Observed | Pass |
| --- | --- | ---: | :---: |
| Net bps per common opportunity at 10 bps, abstentions as zero | > 0 | -0.854 | no |
| Whole-session bootstrap 95 % lower bound | > 0 | -1.490 | no |
| Profitable folds | ≥ 4 of 5 | 1 of 5 | no |
| Selected trades per fold | ≥ 250 each | 104, 177, 305, 152, 107 | no |
| Improvement over always-long on identical opportunities | > 0 | +8.713 bps | yes |

Always-long net per opportunity at 10 bps: -9.567 bps. The classifier selected
845 of 7,434 out-of-fold opportunities (11.4 %). Fold detail:

| Fold | Train rows | Test window | Selected | Net bps / opportunity |
| ---: | ---: | --- | ---: | ---: |
| 1 | 1,482 | 2024-05-06 to 2024-09-04 | 104 | +0.014 |
| 2 | 2,985 | 2024-09-05 to 2025-01-02 | 177 | -1.176 |
| 3 | 4,461 | 2025-01-03 to 2025-05-05 | 305 | -1.031 |
| 4 | 5,955 | 2025-05-06 to 2025-09-03 | 152 | -1.580 |
| 5 | 7,440 | 2025-09-04 to 2025-12-31 | 107 | -0.507 |

The ledger records the outcome as inconclusive because four of five folds fell
below the 250-selection support requirement. The economic criteria also failed,
so this is not a case where more support would plausibly have rescued the
result: the point estimate and the lower bound are both negative in four folds.

## What the 60-minute horizon changed versus the September 21 five-minute trial

| Metric | Five-minute (Sept 21) | 60-minute (this trial) |
| --- | ---: | ---: |
| Net bps per opportunity | -3.197 | -0.854 |
| Bootstrap 95 % lower bound | -3.446 | -1.490 |
| Profitable folds | 0 of 5 | 1 of 5 |
| Always-long net per opportunity | -9.947 | -9.567 |
| Improvement over always-long | +6.750 | +8.713 |
| Selected trades | 30,802 of 94,200 | 845 of 7,434 |

Two readings. First, the longer horizon helps: losses shrink by about three
quarters and the classifier abstains more usefully. Second, it does not help
enough: always-long over 60 minutes still loses 9.6 bps at a 10 bps cost, which
means the average gross 60-minute move in these three names is about +0.4 bps.
The feature contract cannot find enough of the rare large moves to pay a fixed
10 bps round trip at either horizon. The limiting factor is cost relative to the
gross move, not the horizon alone.

## What this does and does not authorize

- The stop rule applies: no threshold, horizon, feature or family tuning on this
  campaign, and no rerun. A different hypothesis needs a new pre-registered
  contract and its own amendment.
- The sampling-support lesson is real: non-overlapping 60-minute holds yield
  about six opportunities per symbol-session, so a 250-selection-per-fold
  requirement needs either a selection rate above 17 % or a wider universe. Any
  future intraday contract should run the timestamp-only support audit against
  the acceptance thresholds, not only against existence of opportunities.
- Nothing here changes the research reset, the holdout, the promotion gates or
  the December 9 review. The stale-model deadlock is not resolved; retraining the
  day-sleeve model at a 60-minute horizon is **not** supported by this evidence.

## Evidence

| Artifact | SHA-256 |
| --- | --- |
| `artifacts/day_sleeve_horizon_trial_20261018/report.json` | `06e5ce1ba8ac941798afd692b9cc0492dae3c255799667a2930c8485107fd71a` |
| `artifacts/day_sleeve_horizon_trial_20261018/campaign_state.json` | `abca33c131a483bff42ea0f9cab589257ccee96f5e3eebb7f1dde6713c2e5af5` |
| `artifacts/day_sleeve_horizon_trial_20261018/preflight.json` | `37e79570a53897642be59bbb498fea0d8f6a1a2288e8625c985c214a3f1a34fd` |
| Contract hash (ledger) | `46604293c025e39726e662b4060ab06f8cfefe7d4e6e2517eaf2c596802dce29` |
| Evidence signature | `cc0eb0017534f2e6eac047bc371a9504e24ff2eb277b155d95c69470f182c1f1` |

Evidence type: historical bar-open proxy with fixed costs, not executable fills.
Holdout evaluated: no. Development data 2024–2025 SIP for AAPL, AMZN, MSFT.
Command, run from the `691a33ece` maintenance checkout against the registered
ledger in the runtime checkout:

```bash
./venv/bin/python -m ai_trading.tools.model_replacement_trial \
  --repository-root /home/aiuser/ai-trading-bot \
  --contract /home/aiuser/ai-trading-bot/config/day_sleeve_horizon_trial_campaign.json --fit
```
