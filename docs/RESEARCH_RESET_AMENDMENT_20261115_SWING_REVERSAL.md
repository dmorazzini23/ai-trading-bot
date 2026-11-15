# Research reset amendment: November 15, 2026 (swing reversal trial)

**Decision:** the owner approved the written rules
([SWING_REVERSAL_TRIAL_RULES_20261115.md](SWING_REVERSAL_TRIAL_RULES_20261115.md))
on November 15 and authorizes one registered development trial,
`swing_reversal_trial_20261115`, under the `registered_strategy_research`
allowance adopted by the
[November 14 review](RESEARCH_RESET_REVIEW_20261114.md). Scheduled training
stays blocked, the holdout stays unscored, consumed budgets stay consumed and
the real-money gate is unchanged.

## Scope of the authorization

- Contract: `config/swing_reversal_trial_campaign.json`. One hypothesis with
  two versions scored in the same run from the same frozen data: among the 300
  most liquid US stocks, buy the 30 with the lowest 5-session return relative
  to their sector at the next open, hold 5 sessions with a hold band, sized by
  trailing SPY volatility. **Long-only is the decision.** Long-short (the same
  book plus shorts of the 30 highest) is reported as `secondary_decision` and
  cannot rescue a failed long-only version.
- Acceptance (long-only, all required): beats an SPY position of the same size
  on the same days at 5 bps per side with a positive bootstrap lower bound;
  positive return after costs with a positive lower bound; still ahead of
  matched SPY at 10 bps; ahead in at least 6 of 10 calendar years; survives the
  −30% survivorship stress; adequate support.
- Data: daily and monthly SIP bars, `adjustment=all`, 2016-01-04 to
  2025-12-31, acquired once by `ai_trading.tools.swing_reversal_trial
  --acquire` after registration, with a hash manifest. No 2026 data and no
  holdout data is read.
- Budget: one trial. A failing or inconclusive result retires both versions and
  authorizes no tuning of any parameter, universe, mapping, cost or acceptance
  rule.

## Survivorship probe (run before registration, dates and counts only)

Run on November 15 (`artifacts/swing_reversal_trial_20261115/survivorship_probe.json`).
The provider returned monthly bars for all 16 delisted probe symbols, but its
asset list contains only 9 of them (FTR, LNKD, MNK, MON, TWTR, TWX and YHOO are
absent), and some symbols have been reused by other companies (EMC, SHLD, MON).
The contract therefore records
`probe_status: delisted_symbols_missing_or_partial`, and the report labels
every result `upward_biased`. The owner approved running on this data with that
label. The −30% stress does not cover companies that never enter the universe.

Consequence, agreed before any result: a fail is a reliable negative, because
the bias favours passing. A pass is not established by this trial alone.

## Changes made at registration (before any price was read for analysis)

The approved rules are unchanged. The contract's draft markers were replaced
with the facts above: `status`, `authorization`, `survivorship.probe_status`,
`survivorship.probe_finding`, `survivorship.bias_label`, `availability_check`
and the first entry of `limitations`; `draft_notice` was removed.

The runner's asset-list read was changed on November 15 (`0b8d88f39`) to take
unvalidated rows, because the pinned SDK rejected the whole list on a new
exchange code. This affects which rows can be read, not any rule.

An independent pre-run review on November 15 found no defect in scoring,
accounting or acceptance, and three ways the download could silently degrade.
The runner was changed before the ledger existed and before any download:
transient provider errors are retried; the monthly request starts on the 1st
so the first month is not omitted; the sector lookup runs before the daily
download and waits out rate limits; and acquisition refuses to write a
manifest if more than 1% of candidates fail the monthly fetch, more than 1% of
universe members have no daily bars, or more than half have no sector. These
are data-integrity checks; no strategy rule or acceptance threshold changed.

## What a pass would permit

Planning only: one untouched consistency check on the reserved holdout
(September 9–December 8, 2026, about 12 periods, rules unchanged), then at
least eight weeks on the paper account. No promotion, runtime or real-money
authority (`promotion_authority: false`, `runtime_authority: false`).

## Preconditions before the trial is claimed

1. Registration (this record, the config entries and the ledger).
2. `--acquire` writes the manifest.
3. `--preflight` passes: timestamp-only support audit, source hashes,
   2016–2025 bounds, price validity, complete SPY.
4. `--fit` once. The owner's instruction of November 15 ("You have my
   approval. Go ahead and continue") covers registration, the download and the
   single scoring of each version.

## Change record

- `config/research_reset.json`: campaign appended to `registered_trials`, this
  record appended to `amendments`.
- `config/swing_reversal_trial_campaign.json`: `draft_not_registered` →
  `registered` with the fields listed above.
- Ledger: `artifacts/swing_reversal_trial_20261115/campaign_state.json` in the
  research worktree (gitignored), created by
  `experiment_ledger.register_campaign`. The running bot's checkout and data
  directory are not touched.
- Rollback: revert the config edits and this record. Keep the ledger as
  evidence; a registered campaign is never silently unregistered.
