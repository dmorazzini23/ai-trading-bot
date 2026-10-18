# Declared fee schedule: decision memo (2026-10-18)

**Status:** capability built, **disabled**. No schedule is declared. Nothing is
backfilled. This memo is for the owner. It covers what turning the capability on
would change and how to do it.

## Why this exists

Research reset evidence gap 3 says Alpaca's paper activity schema does not give
per-fill total fees. `FILL` activities have no fee field. `FEE` activities are
daily account aggregates with no order or fill reference. Alpaca's Paper Trading
page says paper trading does not account for regulatory fees
(`docs/PAPER_ROUND_TRIP_COST_EVIDENCE_20261018.md` section 5). The reset forbids
imputing zero and forbids spreading account-level charges across fills. As a
result, every fill record has `fee_source="missing"`, so TCA net cost and
net-profitability claims cannot be known.

## What enabling changes

- **New** fill records written after a restart get `fee_amount`, `fee_bps`,
  `fee_currency="USD"`, `fee_basis="declared_schedule_per_fill"`,
  `fee_source="declared_schedule"`, `fee_schedule_id`, `fee_schedule_hash`
  (the sha256 of the schedule file bytes), `fee_components` and
  `fee_schedule_status`. These values are computed from your declared schedule
  when the record is written.
- Pending TCA rows resolved from those fills carry the same labels, and
  `total_fee_verified` stays `false`.
- `runtime_performance_report` gives these trades fee evidence
  `declared_schedule`. When no trade has unknown or estimated fees, P&L is
  reported with `pnl_basis="declared_fee_schedule_not_broker_verified"`. That
  makes net-cost figures visible, under a label.
- The scorecard tools add separate, labeled counts:
  - `fee_provenance_counts` in the research reset execution chain
  - `declared_schedule_fee_broker_total_unknown` in broker accounting
    `fee_coverage`
  - `fees_declared_schedule` in execution reconciliation
  - `declared_schedule_observed_*` fields in paper evidence review
- Precedence for each fill: broker payload fee first, then the declared schedule,
  then `AI_TRADING_ESTIMATED_FEE_BPS`, then missing. A declared fee never
  overrides a broker-sourced fee.
- If the source is on but the schedule is missing, invalid, still contains
  placeholders, does not cover the fill date, or does not cover the venue, the
  fee stays missing. The record gets a `fee_schedule_status` reason, and
  `FEE_SCHEDULE_DECLARED_UNAVAILABLE` is logged once for each reason.

## What it does not prove

- It is **not broker evidence**. The research reset's complete causal chain
  still requires broker-sourced USD per-fill totals. Declared fills still report
  `verified_total_fees_missing`. The `complete_causal_execution_chains_unverified`
  and `verified_total_execution_costs_unavailable` blockers stay in place.
  Paper net-cost validation stays unvalidated.
- It does not show what Alpaca charged. Alpaca rounds **daily account totals**
  up to the cent. A per-fill rule cannot reproduce that, so expect per-fill sums
  to differ from the daily `FEE` activities.
- It does not cover borrow fees, exchange or ECN access fees, rebates, market
  impact, or queue position. It has no promotion authority. The schedule itself
  must declare `"promotion_authority": false`.
- It does not change historical records. Fills written before enabling stay
  `missing`.
- A rate is only as good as the source you copied it from. The code never
  supplies figures.

The cost report's suggested rule kept modeled fees out of `fee_amount`. This
implementation does write `fee_amount`, but it isolates it with a distinct
`fee_source`, a distinct `fee_basis` and a mandatory hash. Every consumer that
checks broker evidence compares `fee_source` against
`{broker_payload, broker_activity}` and `fee_basis == "per_fill_total"`. A
declared fee therefore fails both checks.

## Exact enabling steps (owner only)

1. Record a research-reset amendment that authorizes a declared schedule as a
   distinct evidence class. Note its path.
2. Copy `config/fee_schedule.paper_alpaca.example.json` to
   `config/fee_schedule.paper_alpaca.json`. Fill every `<FILL: ...>` value from
   the [Alpaca Regulatory Fees](https://docs.alpaca.markets/us/docs/regulatory-fees)
   page and the brokerage fee schedule at <https://alpaca.markets/disclosures>.
   Required values: commission, SEC Section 31 (sells), FINRA TAF (sells, with
   its per-trade cap), CAT (buys and sells), the rounding rule,
   `effective_from`, `source_url`, `retrieved_at`, `declared_by` and
   `declared_at`. Either fill the optional provenance keys
   (`document_sha256`, `document_version_marker`, `reset_amendment`) or remove
   them.
3. Validate the file without enabling it. Expect `FEE_SCHEDULE_VALIDATION_OK`
   and note the hash:
   `./venv/bin/python -m ai_trading.analytics.fee_schedule config/fee_schedule.paper_alpaca.json`
4. Set these in `/home/aiuser/ai-trading-bot/.env`:
   `AI_TRADING_FEE_SCHEDULE_SOURCE=declared` and
   `AI_TRADING_FEE_SCHEDULE_PATH=/home/aiuser/ai-trading-bot/config/fee_schedule.paper_alpaca.json`
5. Run `sudo systemctl restart ai-trading.service`, then
   `curl -sS http://127.0.0.1:9001/healthz`.
6. After the next fill, confirm that the newest
   `/var/lib/ai-trading-bot/runtime/fill_events.jsonl` row has
   `fee_source="declared_schedule"` and the hash from step 3. Confirm that no
   `FEE_SCHEDULE_DECLARED_UNAVAILABLE` appears in
   `journalctl -u ai-trading.service`.

**Rollback:** set `AI_TRADING_FEE_SCHEDULE_SOURCE=missing`, or remove the line,
and restart. Records already written keep their declared label and hash. They
are never rewritten. **Changing rates:** write a new file with a new
`schedule_id` and `effective_from`, validate it, then restart. The file is
re-read whenever its size or modification time changes, so in-place edits take
effect on the next fill. The hash on each record identifies the schedule that
produced it.
