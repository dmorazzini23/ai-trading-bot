# Research reset amendment: November 15, 2026 (intraday momentum trial)

**Decision:** after the swing reversal trial was retired on November 15
([result](RESEARCH_RESET_AMENDMENT_20261115_SWING_REVERSAL.md#result-november-15-2026-retired)),
the owner chose to continue ("Let's keep trying"), approved the written rules
([INTRADAY_MOMENTUM_TRIAL_RULES_20261115.md](INTRADAY_MOMENTUM_TRIAL_RULES_20261115.md))
the same day ("Yeah, go for it"), and authorizes one registered development
trial, `intraday_momentum_trial_20261115`, under the
`registered_strategy_research` allowance of the
[November 14 review](RESEARCH_RESET_REVIEW_20261114.md). Scheduled training
stays blocked, the holdout stays unscored, consumed budgets stay consumed and
the real-money gate is unchanged.

## Scope of the authorization

- Contract: `config/intraday_momentum_trial_campaign.json`. SPY only. Signal:
  sign of the return from the prior session's final regular close to 10:00 New
  York time. Trade: open of the 15:30 bar to its close, exposure 1.0, nothing
  overnight. **Long-only is the decision** (in the trade on a positive signal,
  flat otherwise). Long-short (also short on a negative signal) is reported as
  `secondary_decision` and cannot rescue a failed long-only version.
- Acceptance (long-only, all required): bootstrap lower bound of the mean net
  return per eligible day above zero at 1 bp per side; mean above zero at
  2 bps; mean above zero in at least 6 of 10 calendar years; mean ahead of
  always-in (15:30 to close every day, same cost); adequate support. The owner
  was told the always-in comparison requires a positive mean only, and that
  part of it is saved cost, and approved it as written.
- Data: SIP 30-minute bars, `adjustment=all`, 2016-01-04 to 2025-12-31,
  acquired once after registration with a hash manifest. No 2026 data and no
  holdout data is read.
- Budget: one trial. A failing or inconclusive result retires both versions and
  authorizes no tuning.

## Changes made at registration (before any download)

The approved rules are unchanged. `status` and `authorization` were set and
`draft_notice` removed. An independent pre-run review found no scoring defect;
on its findings the runner now refuses to run without the real exchange
calendar and refuses a regular-hours bar dated on a day the calendar lacks
(`4118b3f6d`). Before scoring, the overnight gap on SPY ex-dividend dates is
inspected to confirm that the provider adjusts half-hour bars consistently;
this reads about 40 overnight returns and no strategy outcome.

## What a pass would permit

Planning only: one untouched consistency check on the reserved holdout
(September 9–December 8, 2026), then at least eight weeks on the paper
account. No promotion, runtime or real-money authority.

## Change record

- `config/research_reset.json`: campaign appended to `registered_trials`, this
  record appended to `amendments`.
- `config/intraday_momentum_trial_campaign.json`: `draft_not_registered` →
  `registered`.
- Ledger: `artifacts/intraday_momentum_trial_20261115/campaign_state.json` in
  the research worktree (gitignored). The running bot's checkout and data
  directory are not touched.
- Rollback: revert the config edits and this record. Keep the ledger as
  evidence.

## Result (November 15, 2026): retired

Claimed once at 2026-11-15T18:40:33Z (market closed), after the acquisition
(2,514 sessions with bars, 21 early closes, no incomplete session), the
ex-dividend check (mean overnight gap on 40 ex-dividend dates −0.05%, against
about −0.3% to −0.45% if the bars were unadjusted), the support audit (2,492
eligible days, 10 supported years) and the non-claiming preflight (no rejected
prices) passed. Ledger status `retired`. The holdout was not evaluated; no
model was fitted; no orders were sent.

Figures are basis points per eligible day at 1 bp per side unless noted.

| Long-only criterion (decision) | Result | Met |
| --- | --- | --- |
| Adequate support | 2,492 days, 10 years, 1,390 traded | yes |
| Net mean vs cash, lower bound > 0 | −1.55 (−3.9% a year); lower bound −2.71 | **no** |
| Net mean > 0 at 2 bps | −2.66 | **no** |
| Positive in at least 6 of 10 years | 1 (2020) | **no** |
| Mean ahead of always-in | +0.96 (+0.08 before costs) | yes |

| Long-short criterion (secondary) | Result | Met |
| --- | --- | --- |
| Net mean vs cash, lower bound > 0 | −2.34 (−5.9% a year); lower bound −3.87 | **no** |
| Net mean > 0 at 2 bps | −4.33 | **no** |
| Positive in at least 6 of 10 years | 1 (2020) | **no** |

Reported only: before any cost the long-only version averaged −0.43 and the
long-short version −0.34 per eligible day, and the last half hour itself
averaged −0.51. The always-in comparison was passed almost entirely on saved
cost. Neither reported-only variant (both predictors agreeing; high
volatility) was positive at any cost tier.

The published effect is not present in SPY half-hour bars for 2016–2025 even
before costs, so both versions are retired. Per the stop rule no tuning is
authorized and the holdout stays unread.
