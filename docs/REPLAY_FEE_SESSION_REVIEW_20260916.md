# Replay, fee-source and completed-session review

Verified September 16, 2026, approximately 04:48 UTC. No runtime code, strategy
parameters, gates, budgets or historical records were changed.

## All 24 missing simulation comparisons traced

The referenced governance replay was generated September 15 at 09:29 UTC with
894 normalized input rows ending September 14 at 19:35 UTC. Normalizing the
retained source reproduces its source hash. Re-running the identical fixed
strategy/seed/caps/input reproduces the complete saved output hash. This was a
diagnostic reconstruction, not a new strategy trial or holdout evaluation.

| Cause | Actual fills | Evidence |
| --- | ---: | --- |
| After replay input cutoff | 10 | All September 15 fills postdate the recorded input window |
| Replay caps removed order | 9 | Same symbol/time/side input has adjusted quantity zero |
| Simulated order expired unfilled | 3 | Reproduced order_expired events at the September 10/11 session close |
| Absent source lineage | 2 | Older AAPL/AMZN market exits have order/fill records but no matching decision or TCA rows |

The candidate has 90 fills and 90 markouts, with no markout exclusions. The
baseline has 536 fills, 527 markouts, eight horizon exclusions and one missing
subsequent observation. Candidate missing comparisons are therefore not silently
dropped markouts. The two benchmark mismatches remain separate from these 24
missing comparisons and remain excluded.

The historical exit records do not establish a causal decision benchmark. Do not
fabricate a decision or substitute fill price to include them. A future normal
closeout with positions should verify that exit decision/quote/TCA evidence is
captured; that check remains pending naturally occurring market activity.

Private audit artifacts: /tmp/replay-missing-observations.json,
/tmp/replay-trace-reproduction.log, /tmp/reproduce_replay_trace.py. Temporary
runtime directories isolate the reconstruction; it sent no broker orders and
did not replace the production replay artifact.

## Broker fee sources

The unfiltered activity collector already requests every activity type with
pagination. Additional direct FEE, PTC and PTR requests since September 15 each
returned zero rows, with pagination complete, at 04:44 UTC September 16.
Evidence: /tmp/replay-fee-followup-broker.json.

[Alpaca account activities](https://docs.alpaca.markets/us/docs/account-activities)
documents fill/order identifiers and USD FEE activity, plus pass-through charges
and rebates. Its documented fill schema does not supply a complete per-fill fee
total. [The activity endpoint](https://docs.alpaca.markets/us/reference/getaccountactivitiesbyactivitytype-1)
uses creation dates; non-trade fee activities can appear the next day. Therefore
an empty early query alone cannot establish final zero fees.

More fundamentally, [current paper-trading documentation](https://docs.alpaca.markets/us/docs/paper-trading)
states regulatory fees are not modeled. Waiting for paper activity records alone
cannot establish complete real-world execution costs. Broker API partner document
endpoints are not established as an alternate source for this paper Trading API
account. No live account was accessed or created. Explicit cost assumptions may
remain labeled assumptions, but cannot satisfy broker-observed fee gates.

## September 15 completed-session review

The updated scheduled verification was run for September 15. Ten unique fills
match decisions, orders and TCA; ten cumulative order quantities match. Complete
opening/closing position boundaries reconcile. No invalid/conflicting fills or
missing identities were observed. Fees_missing=10 remains the sole session gap.
The review remains evidence_pending, not fully reconciled net-cost evidence.
Artifact: /tmp/replay-fee-session-review/latest.json.

Earlier service/broker checks confirmed no positions/open orders after close and
no service restart through close. Positions had closed earlier, so this session
does not validate liquidation with remaining exposure. A new-session closeout
test remains pending; no background watcher or forced trade was installed.

## Validation and disposition

Exact replay input/output hashes matched. The updated session CLI completed and
retained the evidence gates. Fee endpoint pagination completed; health snapshot
was refreshed. Docs-only validation and diff checks apply; no new runtime tests
or service restart are required for this read-only investigation.

No missing-comparison implementation defect was established by this trace.
Remaining boundaries are explicit: historical exit telemetry, later natural
closeout verification, and fee evidence unavailable from the observed paper
sources. Training remains paused under the existing explicit-review policy.
