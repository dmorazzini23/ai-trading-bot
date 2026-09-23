# Remaining-work handoff (historical September 22 snapshot)

Intended task: `01a0ca5a-a752-7440-b20d-9364624cb334`.
User requested transferring remaining work to this task. Direct delivery was
unavailable on this host; this file is the shared-workspace handoff.

## Verified completed at September 22, 2026, 23:04 UTC

- Production and remote main match `43abd3d0f5712d8d865fcd90e98492a203f0e352`.
- NumPy contamination was traced to `tests/execution/test_execution_imports.py`:
  the test reloaded bot_engine while fake NumPy was installed. Removed the fake
  dependency and added an assertion preserving canonical NumPy.
- CI35771599946 passed: 7,066 passed, four skipped, 80.04% coverage. CodeQL,
  Workflow Lint and SBOM passed on the same revision. The 80% floor is unchanged.
- After-close deployment/restart completed at 20:00:40 UTC. Service active,
  NRestarts=0; no structured ERROR/CRITICAL journal entries since restart when
  checked at 23:04 UTC.
- Live health reports `paper_diagnostics_only`, model_ready=false,
  promotion_authority=false. Only readiness failure: `required_model_stale`.
  Replay counterfactual qualification still fails; live new exposure is blocked,
  paper evidence and risk-reducing orders remain allowed.

## Remaining work, in priority order

1. Verify postdeployment behavior during the next regular market session.
   Read broker clock/account/positions/orders, service health and bounded logs.
   Confirm paper diagnostics are labeled correctly and order/fill behavior is
   sound. Check after-close reconciliation artifacts separately. Do not force
   trades to obtain samples or treat abstention as model performance.
2. Assess the missing qualified model using current governance evidence.
   The September 21 bounded replacement trial was rejected and its one-trial
   budget consumed. Read `docs/MODEL_REPLACEMENT_20260921.md`, the campaign
   configuration and existing artifacts before proposing next steps. Do not
   rerun, tune, promote or start a new experiment under this handoff.
3. Diagnose the remaining replay qualification failure (`replay_counterfactual`)
   against current artifacts. Separate implementation defects from insufficient
   evidence; preserve qualification gates and provenance requirements.
4. Reconcile remaining historical ledger discrepancies against broker evidence.
   Consult `docs/LEDGER_AND_COVERAGE_20260919.md` and existing ledger-rebuild
   artifacts. Preserve original history; unsupported periods stay explicitly
   unverified. Do not manufacture opening balances or executions.

## September 22, 23:14 UTC continuation findings

- The first complete **postdeployment** regular session is September 23, so
  priority 1 cannot be completed tonight. The September 22 after-close report
  is a baseline: seven accepted fills, seven order-quantity matches and no
  closing position difference. It still has seven unknown total fees and two
  fills without decision/TCA matches; it does not support net-cost or strategy
  performance claims. `capture_readiness` names September 23 as the next full
  session. Check live diagnostic labels, broker positions/orders and bounded
  logs during that session, then review its close artifacts separately.
- The single replacement trial remains `hypothesis_rejected`. Current campaign
  state contains one claimed trial; the report and prediction hashes match.
  Its mean is -3.197464 bps per common opportunity, with a negative bootstrap
  lower bound. No holdout was evaluated and no serving model was saved.
- The September 22 replay artifact has fresh source data and zero replay
  violations. Its counterfactual fails two requirements: 191 candidate fill
  samples versus 250 required, and -0.129341 bps net edge versus positive
  edge required. The refresh reports `REPLAY_POLICY_NON_REGRESSION_FAILED`.
  This is a qualification failure on available evidence; no implementation
  defect was established. Do not waive or lower either threshold.
- The September 18 broker-backed quantity ledger matches eight executions;
  the extended ledger matches 50 executions from the earliest complete
  September 7 anchor. The legacy AAPL -3 / AMZN +1 discrepancy was already
  present at the verified flat September 18 opening. Its pre-anchor origin
  remains unverified. No original history was changed.

No code or runtime change followed this read-only assessment. The next action
is the September 23 regular-session verification, subject to the same evidence
and trading gates.

## Constraints and working state

- Read AGENTS.md and `docs/CODEX_HANDOFF.md`; use one agent by default and
  apply_patch for edits. Keep work bounded and conserve the user's usage.
- Keep all commits on main; leave historical branches archived/untouched.
- No weakening model, replay, cost, freshness, provenance or promotion gates.
  Preserve the September 9–December 8 holdout and consumed campaign budgets.
- No new models, searches, ticker expansion or paid real-time SIP assumptions.
- Use targeted validation during market hours; runtime changes need health
  smoke checks, ops changes a non-sending incident check, and decision changes
  appropriate replay/backtest verification. Reuse unchanged successful checks.
- The production checkout has an uncommitted docs/CODEX_HANDOFF.md update and
  this untracked handoff. Do not discard either. Preserved earlier edits are in
  stash@{0} and /tmp/sep21-production-predeploy/local-work.tgz; do not blindly
  restore them.
- Useful verification artifacts: /tmp/completion-ci.log,
  /tmp/completion-health.json, /tmp/completion-journal.log. These are snapshots;
  refresh time-sensitive state before acting.
- Do not redeploy or restart again merely to repeat completed work.

This handoff does not establish that a qualified strategy is profitable or that
historical accounting is fully reconciled. Those remain evidence gaps.
