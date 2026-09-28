# September 28 data, execution and recovery check

This is a read-only research and execution audit plus a local recovery rehearsal.
It did not evaluate a strategy, train a model, read the September 9–December 8
holdout, submit an order, change a gate, or authorize a trial or promotion.

## Weekly ETF data preflight

`ai_trading.tools.weekly_etf_data_preflight` verified the exact source hashes of
the governed 2024–2025 SIP, all-adjusted minute bars for DIA, IWM, QQQ, SPY,
XLE and XLF. For every exchange-open Friday with enough history and future
observations, it required 21 completed session closes for a trailing 20-session
lookback, the next session's actual opening bar, and the opening bar five
sessions after entry. It checked positive, coherent OHLC values and common
dates without calculating a rank, signal, selection, return or cost result.

| Measure | Result |
| --- | ---: |
| Candidate Friday decision weeks | 96 |
| Weeks valid across all six symbols | 96 |
| Valid weeks by quarter, 2024 Q1–2025 Q4 | 8, 13, 13, 13, 13, 12, 12, 12 |
| Captured development cash dividends | 64 |
| Captured development forward splits | 1, XLE |
| Captured actions with valid previous close and ex-date open/close bars | 65/65 |
| Sampled adjustment diagnostics consistent under stated rounding assumption | 19/19 |
| Sample-to-governed-dataset bar checks matched | 38/38 |

The corporate-action capture has distinct IDs, positive rates, an ex-date on a
development trading session for each included action, and completed provider
pagination. Its request spans December 2023–March 2026 because Alpaca's
corporate-action `start`/`end` filters use **process date**, while the join uses
**ex-date**. The prior adjustment sample hash and all six dataset linkage hashes
still match their saved source bytes. Alpaca says `data_quality=complete` filters
some incomplete, unprocessed events and does not guarantee creation time; a
complete response is therefore not independent proof of every distribution or
point-in-time availability. See [Alpaca corporate actions API](https://docs.alpaca.markets/us/reference/corporateactions-1)
and [the earlier adjustment review](ENDPOINT_PREPARATION_RESULT.md).

**Decision:** observation support passes for this draft Friday schedule.
Distribution completeness, any future revision policy, point-in-time action
availability, executable opening cost, and a frozen trial contract remain
unverified. The 96 weeks are an upper bound on possible opportunities, not
selections or positive economics. Keep the earlier consumed trials and reserved
holdout unchanged. A new trial requires a separately approved, immutable
contract with its own support threshold and paired after-cost baselines.

Reproduce the outcome-free preflight:

```bash
./venv/bin/python -m ai_trading.tools.weekly_etf_data_preflight \
  --acquisition artifacts/endpoint_proposal/acquisition_absolute.json \
  --corporate-actions artifacts/endpoint_proposal/corporate_actions_expanded.json \
  --output artifacts/research_reset/weekly_etf_data_preflight_20260928.json
```

The local generated report includes all six source hashes, the capture hash,
per-symbol rejection counts and common Friday dates. It is ignored under
`artifacts/`; this document preserves the decision in the repository.

## Decision-to-fill and fee audit

At 03:16–03:18 UTC, stable reads of the seven-day local sources covered 2,414
decision rows, 61 order events, 13 unique observed fills and 23 TCA rows.
The window was September 21 03:16 to September 28 03:16 UTC. The existing
reconciler matched all 13 fills to an order, 11 to a decision and 11 to a
nonpending TCA record. The two unmatched decision/TCA rows are the recorded
September 22 AMZN and MSFT operational EOD exits, whose order role and EOD
client-order identities match. No missing strategy decisions or arrival quotes
were manufactured for them.

All 13 local fills have unknown total fee provenance. The stricter research
scorecard found **0/13 complete causal chains**: all 13 lacked a qualifying
decision-time quote and verified per-fill total fee; two additionally lacked
an explicit decision/source-bar/order chain because they were operational
exits. A matched TCA row is not fee evidence. The latest broker accounting
snapshot (September 27 14:03 UTC) had 369 raw fill-activity rows with no fee
amount and 69 fee-activity rows with no execution reference. The 335 compared
execution quantities had unknown total fees. Account-level charges remain
separate and are not allocated to guessed fills; neither an absent charge nor
zero in an estimate establishes a zero broker fee. Gross or historical reported
P&L remains unverified net profitability.

The full local reconciliation is at
`/tmp/execution_evidence_20260928.json`, and the read-only scorecard is at
`/tmp/research_reset_scorecard_20260928.json`. The four input files were stable
during each read. This audit does not independently certify the broker's
execution-feed completeness. Continue collecting natural paper fills with
explicit decision/quote observation timestamps; seek a broker source that
identifies complete USD total fees per fill. If none exists, retain the
`unknown` label in every net-performance decision.

## Recovery rehearsal and blocked host work

The installed backup-sync timer is **disabled/inactive**. The installed backup
service is the older upload-only variant, while the checked-in unit creates a
fresh verified bundle before optional S3 sync. `systemd-analyze verify`, Bash
syntax, and the 13 uploader tests passed for the checked-in files. An attempted
host install stopped at `sudo: a password is required`; no unit was installed,
no timer enabled and no S3 transfer was made.

A first direct backup invocation lacked the service runtime environment and
temporarily recorded `failed` in the local backup status. It was rerun with
the same runtime environment as the unit in the host context; status is now
`ok`. The fresh bundle
`recovery.bak.20260928T032649Z-27ba9464.gz` has SHA-256
`24a5768eba333170648013679e9c4c50d49c485abd916266a41511953b43dbe4`,
3,045 entries, and restored in isolation with both SQLite integrity checks
`ok`. It was not activated. The older eight bundles remained in place.
The paper service remained active with zero restarts; host health was HTTP 503
for `required_model_stale`, with fresh broker state, zero positions and zero
open orders. The audit tool and local backup did not change trading behavior.

**Next host action:** an administrator installs the reviewed
`packaging/systemd/ai-trading-runtime-backup-sync.service`, runs
`systemctl daemon-reload`, and confirms byte identity and
`NeedDaemonReload=no`. Recurring S3 upload remains off pending a reviewed
destination/scope and successful scheduled read-back/isolated restore. Recovery
of an older S3 object version remains unproven: the prior version-pinned read
was denied `s3:GetObjectVersion`. Once AWS access is recovered, grant that
permission on the backup prefix, download a known older version by version ID,
verify its checksum, and restore it in isolation. The current-object restore
from September 23 does not meet that older-version criterion.

## Validation

- Weekly preflight: seven focused tests passed; Ruff and mypy passed.
- Existing reconciliation and research-scorecard regressions: 38 passed.
- Backup uploader regressions: 13 passed; systemd unit verification and Bash
  syntax passed.
- A fresh current-state local bundle and two-database isolated restore passed.
- No replay or backtest was run because this task only checked data support
  and evidence integrity; no trading decision changed.
