# Environment and training reset corrections

September 10, 2026. The user authorized the configuration/code fixes found in the
environment review. The source `.env` now disables after-hours training, catch-up
training and automatic disabling of eligible gates. The unused
`AI_TRADING_AFTER_HOURS_LOOKBACK_DAYS=720` assignment was replaced by
`AI_TRADING_DAY_SLEEVE_INTRADAY_LOOKBACK_DAYS=60`, preserving the previously effective
window. Paper execution and model/replay eligibility controls remain in force.

The shared policy implementation is `ai_trading/config/research_policy.py`.
The existing research scorecard uses it, and after-hours training, market-close
dispatch and legacy daily model loading now check it before training. Active,
missing or malformed reset policy blocks training. Passing the review date does
not release the reset. This closes the separate service training path missed by
the earlier research-automation reset. A historical `no_qualified_candidate`
result means the earlier attempt evaluated candidates; it did not mean all
training had been paused. Historical holdout isolation across those earlier
service attempts has not been established by this patch.

Gate adaptation now stays suppressed if its go/no-go report is absent, unreadable,
malformed or lacks a literal boolean passing result. The suppression option's
explicit operator override remains available; `.env` disables adaptation entirely
during the reset. The change does not certify report freshness or relax any gate.

The environment validation CLI hydrates credentials through the configured secret
backend before checking the merged configuration. Backend failure returns nonzero
with a redacted diagnostic. Supplying an explicit empty mapping to the pure
validator no longer falls back to process credentials. Secret values are not
written to files by validation.

Regression coverage includes reset persistence after review, missing/malformed
policy, blocking before input reads or marker claims, managed-secret success and
failure, and missing/invalid versus explicitly passing gate-adaptation evidence.
See `docs/CODEX_HANDOFF.md` for final validation and deployment results.

Rollback: revert only this patch's code and source environment assignments,
then refresh the service. Preserve unrelated work and research ledgers. Restoring
the old enabled training settings would reopen the identified reset bypass, so
retain the three disabled switches while diagnosing any unrelated runtime issue.
