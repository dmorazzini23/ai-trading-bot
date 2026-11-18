# Research reset amendment: November 17, 2026 (opening-range breakout trial)

**Decision:** after the intraday momentum trial was retired on November 15
([result](RESEARCH_RESET_AMENDMENT_20261115_INTRADAY_MOMENTUM.md)), the owner
approved the written rules
([OPENING_RANGE_TRIAL_RULES_20261117.md](OPENING_RANGE_TRIAL_RULES_20261117.md))
"as recommended" on November 17 and authorizes one registered development
trial, `opening_range_trial_20261117`, under the
`registered_strategy_research` allowance of the
[November 14 review](RESEARCH_RESET_REVIEW_20261114.md). Scheduled training
stays blocked, the holdout stays unscored, consumed budgets stay consumed and
the real-money gate is unchanged.

## Scope of the authorization

- Contract: `config/opening_range_trial_campaign.json`. Source: Zarattini,
  Barbon and Aziz, "A Profitable Day Trading Strategy For The U.S. Equity
  Market" (2024, SSRN 4729284).
- Hypothesis: each session, among US common stocks opening above $5 with a
  14-session average volume of at least 1,000,000 shares and a 14-session
  average true range above $0.50, the 20 with the highest first-five-minute
  volume relative to their own 14-session average (at least 100%), bought
  when a later one-minute bar trades through the first five-minute bar's high
  (first bar closed up) or sold short through its low (closed down), with a
  stop loss of 10% of the average true range from the fill and an exit at the
  session close, earn more than cash after commission and 5 bps per side;
  and the long-only version earns more than the same dollars held in SPY over
  the same minutes.
- **Long-only is the decision.** Long-short is scored for information. If it
  meets its criteria while long-only does not, nothing is permitted.
- Data: SIP raw bars, 2016-01-04 to 2025-12-31: daily bars for the filters,
  the 09:30 five-minute bar for the ranking, one-minute bars for the selected
  names and for SPY. Acquired once after registration with a hash manifest.
  No 2026 data and no holdout data is read.
- Budget: one trial. A failing or inconclusive result retires both versions
  and authorizes no tuning.

## What is fixed

- **Filters, ranking, stop, commission:** the paper's numbers, unchanged
  ($5, 1,000,000 shares, $0.50, 14 sessions, top 20, relative volume at least
  1.0, stop at 10% of ATR, $0.0035 per share).
- **Fill model that decides (`bar_executable`):** enter at the open of the
  one-minute bar after the trigger bar; stop measured from that fill; a
  stop-out fills at the stop price, or at the bar's open if it opens beyond
  the stop; otherwise the last bar's close. A halted position is closed at
  its last price less one ATR, never below one cent.
- **Sizing:** 20 equal slots; a stop-out loses 1% of a slot; each position is
  capped at 4 times its slot.
- **Costs:** 5 bps per side decides; the mean must still be positive at
  7.5 bps. 10 bps, 2.5 bps and commission-only are reported.
- **Acceptance (long-only, all required):** bootstrap lower bound (1.25th
  percentile, 21-day blocks, 10,000 replicates) of the mean net return per
  eligible session above zero at 5 bps; mean above zero at 7.5 bps; mean
  above zero in at least 6 of 10 calendar years; bootstrap lower bound of the
  difference against exposure-matched SPY above zero; adequate support (2,300
  eligible days, 9 years of 230, a median of 10 selected names, 5,000 fills).
- **Reported only:** the `conservative`, `paper_exact` and `lower_bound` fill
  models, the every-short-taken book, equal-notional sizing.
- **Survivorship label:** `incomplete_direction_unknown`.

## Review history

1. **First draft (commit `97ab4fb68`).** As specified, it settled every doubt
   inside a one-minute bar against the strategy: entry at the worse of the
   trigger and the next open, a stop-out whenever a bar touched both levels,
   exits at the worse of the stop and the next open.
2. **Independent review, blocker.** On a simulated price walk with no edge
   those rules scored −0.2 to −0.6 planned losses per fill before any cost,
   and erased a true edge of +0.15 to +0.22. The trial would have failed
   whatever the data said. The review also found: acquisition could pass with
   one session's opening bars missing; the selection step needed 1.6 GB on a
   machine with about 1 GB free; a halted cheap stock could be marked below
   zero and crash scoring after the claim; a ticker spelled `NA` was read as
   blank; several omissions in the rules sheet.
3. **Revision (commit `ed690a39e`).** The deciding fill model became
   `bar_executable`, which scores the same no-edge walk at −0.02 to +0.09
   (+0.04 in the unit test that now guards it, standard error 0.03). The old
   rules are kept as `lower_bound`, reported only. The stress cost moved from
   10 to 7.5 bps. Acquisition checks every session on its own and records the
   counts. The selection step peaks at 515 MB at full synthetic scale. The
   other findings were fixed and tested.
4. **Owner decisions, November 17.** (a) Run at 5 bps per side knowing a fail
   is the likely outcome. (b) The `conservative` fill model, drafted in the
   revision as a pass condition, is reported only: on a no-edge walk it
   scores −0.1 to −0.2 per fill and could fail a real edge by itself. (c) The
   missing delisted companies are accepted with the label; no vendor list is
   added. Everything else stands as drafted.

5. **Second review, before the ledger.** It passed the scoring logic with no
   blocker and asked for small fixes, none of which changes a trade, a
   criterion or an approved parameter. (a) The per-session opening-bar
   coverage check is now judged after the SPY bars are fetched and refuses
   only for sessions that will be scored; on 2020-03-16 the market-wide halt
   a second after the open leaves the 09:30 bar nearly empty, that session is
   unscored, and its shortfall is recorded instead of refusing the download.
   Names without an opening bar that day are unranked through 2020-04-03.
   (b) The contract's description of the matched-market control now matches
   the runner: SPY's open of the entry minute for a fill at a bar's open.
   (c) A selected name whose minute request failed was counted twice against
   the 1% limit; it is counted once. (d) The one-missing-name allowance is
   written into the contract. (e) The support audit shows the number of split
   events and the opening-bar counts for the four March 2020 halt days before
   the claim. (f) Tests added: missing minutes under the deciding fill model,
   and the selection against a brute-force reference on gappy data.

## Data check before registration

A probe on November 17 kept bar counts and time stamps only. Minute bars exist
from 2016-01-04. The five-minute opening bar matched its one-minute bars in 25
of 25 cases. For 21 large delisted companies the provider served minute bars
for all 21, but only 10 are in the broker's asset list, so the other 11 could
never be candidates. No return was computed.

## The expectation, stated before scoring

A typical position is 10% to 12.5% of equity, so a round trip at 5 bps per
side costs 1.0 to 1.25 bps of equity. With 12 to 15 fills a day that is about
15 bps of equity a day. The paper's published 41.6% a year is 13.8 bps a day,
after commission, with ideal fills and no spread. The assumed cost is
therefore about as large as the whole published return. **A fail is the
expected outcome even if the paper's gross figures are right.** The owner
chose to run the test on that understanding.

## What a pass would permit

Planning only: one untouched consistency check on the reserved holdout
(September 9–December 8, 2026), then at least eight weeks on the paper
account. No promotion, runtime or real-money authority. A long-short pass
alone permits nothing.

## Change record

- `config/research_reset.json`: campaign appended to `registered_trials`, this
  record appended to `amendments`.
- `config/opening_range_trial_campaign.json`: `draft_not_registered` →
  `registered`; the owner's decisions recorded; the conservative fill model
  moved from pass condition to reported-only.
- Ledger: `artifacts/opening_range_trial_20261117/campaign_state.json` in the
  research worktree (gitignored). **Not yet created at the time of this
  record**; it is created before any price is read for scoring. The
  acquisition is run outside market hours because the running bot shares the
  provider's rate limit.
- The running bot's checkout and data directory are not touched.
- Rollback: revert the config edits and this record. Keep any ledger as
  evidence.

## Addendum, November 18, 2026: one session unscored, trial re-registered

- **Refusal.** The one-time acquisition ran on November 17 and refused at its
  last check, writing no manifest:
  `opening bars cover too few names:2022-03-08:returned=1041:asked=1161`
  (89.7% against the 0.90 floor). No trial was claimed.
- **Finding (bar counts only; no price, volume total or return looked at).**
  On 2022-03-08 the provider's archive (Alpaca SIP) has no intraday bars for
  NYSE-listed stocks after the 09:30 minute for the whole session: JPM, XOM,
  GE and F have exactly one one-minute bar, at 09:30; KO and BAC have none;
  116 names with an opening bar on 03-07 and 03-09 have none on 03-08, all
  NYSE-listed. AAPL, MSFT, SPY and IWM are complete, so the incomplete-SPY
  rule does not mark the session. It is the only such session of the 2,513;
  the only larger shortfall, 2020-03-16, is already unscored.
- **Decision (owner, November 18, 2026, on data availability, before any
  scoring).** 2022-03-08 is unscored as a documented provider outage, treated
  the same way as a session with an incomplete SPY record. The 0.90 floor is
  not lowered and no other rule changes. The contract carries the session in
  `excluded_sessions` with its reason and evidence; the runner leaves it out
  of the trades of every fill model, the exposure-matched SPY comparison and
  the support counts, records it in the acquisition manifest
  (`excluded_sessions`, with its asked/returned/failed counts and whether it
  fell short) and shows it in the support audit's `pre_claim_review`. A
  listed session with full coverage is still unscored and the manifest says
  so. Fetch-failure limits and the floor on every other session are
  unchanged. Stated consequence, as for 2020-03-16: names without the
  2022-03-08 opening bar are unranked through 2022-03-28, and a truncated
  opening bar on that day lowers the 14-session volume average used by the
  ranking on those sessions.
- **Registration.** A registered contract is immutable in the ledger, so the
  amended contract is registered as `opening_range_trial_20261118` with its
  own directory `artifacts/opening_range_trial_20261118/`.
  `opening_range_trial_20261117` is withdrawn before any claim: it is removed
  from `registered_trials` and listed under `withdrawn_trials` in
  `config/research_reset.json` with zero trials claimed and no budget
  consumed or carried over. Its ledger and refused partial download stay on
  disk as evidence and are not read by the new campaign.
- **Hashes.** Withdrawn contract:
  `bd0ab065cf1460873df8c85fa30a5eb1894840ac4819e9368b18440fc4c75998`.
  New contract:
  `3c767e3abbc5a7212e2047ee684502c3f42c2f3f2486cb102b8cccc9f0582363`.
- **Not yet done at the time of this record:** the new ledger
  (`artifacts/opening_range_trial_20261118/campaign_state.json`) and a fresh
  acquisition, each on the owner's word, outside market hours.

## Result

_Empty. To be filled once, after the single `--fit`, with the decision, the
criteria table for both versions, the reported-only figures and the counts of
skipped sessions. No result exists at the time of registration._
