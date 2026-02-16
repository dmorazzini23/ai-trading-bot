# Runbook: Deployment Rollback

## Trigger Conditions
- Institutional gate failure after deploy.
- SLO critical breach sustained beyond threshold.
- Unexpected reject-rate spike, reconciliation mismatch, or breaker storms.

## Pre-Rollback Capture
1. Capture current commit and deploy metadata.
2. Export last 30 minutes of service logs.
3. Snapshot runtime artifacts (`decision_records`, `tca_records`, `execution_reports`).

## Rollback Steps
```bash
# Example flow (adapt to your release tooling)
git checkout <last-known-good-commit>
systemctl restart ai-trading.service
```

## Post-Rollback Verification
1. Health endpoint returns HTTP 200 with healthy JSON.
2. Breakers closed for critical dependencies.
3. Order flow in canary mode only until stability window passes.
4. Institutional gate script passes on rollback commit.

## Stabilization Window
1. Keep canary-only routing for at least one full market session.
2. Review TCA and reject metrics before expanding symbol scope.
