# Reporting provenance and closeout follow-up — September 15

The live submission path now passes the primary netted sleeve's strategy ID into
the canonical durable intent store. Primary selection matches the decision
journal (largest absolute proposal score); this label does not grant model or
strategy qualification authority. Missing strategy identity remains unknown.

Decision receipts now persist requested_qty, submitted_qty and filled_qty
separately. The receipt's qty reflects the observed broker order quantity;
missing broker quantity stays null. The journal order intent retains requested
quantity, while its broker snapshot never substitutes requested quantity for
missing submitted quantity. Both accepted and rejected broker responses use
these fields. Existing historical journals and intents were not rewritten.

Changed runtime paths: core/netting_submit_execution.py,
core/execution_outcome.py, execution/live_trading.py, contracts/decisioning.py.
Regression coverage includes sampled and partial fills, unknown/zero broker
quantities, nested live broker results, and durable strategy propagation.

Validation: 54 focused tests passed; 32 final journal/receipt tests passed;
agent_validate_changed.sh --market-hours --skip-runtime-smoke passed 426 selected
tests, lint, mypy (35 files) and compilation. Final contract/test type check
passed (2 files). Non-sending incident snapshot passed. No decision gates,
order sizing, research trials, holdout boundaries or model authority changed;
replay was not needed for this reporting-only change.

Read-only broker snapshot at 18:38 UTC: market open, ten orders, no positions.
All ten broker cumulative fill quantities match durable fill totals; all ten
pre-deployment intent strategy IDs remain null. Artifacts:
/tmp/provenance-broker.json and /tmp/provenance-intraday-reconciliation.json.
These are intraday evidence, not completed-session reconciliation.

Closeout remains to be observed around 19:55–20:00 UTC. Verify service continuity,
broker positions and open orders; the earlier EOD exception fix does not prove
successful flattening or eliminate broker transport delays. After 20:15 UTC:

```bash
./venv/bin/python -m ai_trading.tools.scheduled_evidence_verification \
  --runtime-dir /var/lib/ai-trading-bot/runtime --session 2026-09-15 \
  --output-dir /tmp/provenance-evidence
```

The preparation run correctly reports awaiting_session_close. Scheduled daily
evidence may arrive later; missing per-fill fees and causal quotes remain gaps.
No background closeout watcher was installed. Natural post-deployment orders
must confirm the new persisted fields; no trades are forced for validation.

Deployment: service restarted at 18:40:05 UTC, active with NRestarts=0. Health at
18:40:39 shows a fresh connected broker, zero positions and zero open orders.
Startup logs contain existing stale-model/parity, minute-gap and sampling blocks;
no new reporting error observed. Health remains degraded by the existing gates.
Evidence: /tmp/provenance-post-health.json and /tmp/provenance-postrestart.log.
Docs-only validation and diff whitespace checks passed.

Rollback: revert only these reporting hunks and restart the service. No database
migration or historical repair is required; reverting restores the old reporting
ambiguity and does not repair previously written receipts.
