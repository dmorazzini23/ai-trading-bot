# September 7-8 evidence verification

Prepared September 7, 2026, before tonight's daily research run and tomorrow's
paper session. Neither future outcome is claimed as completed.

## Automated follow-through

`ai_trading-evidence-verification.timer` schedules read-only checks at 23:00 UTC
on September 7 and September 8 under the `aiuser` systemd user manager. User
lingering is enabled, so logout does not stop the manager. These are two dated checks, not a new recurring
research or trading job. They do not retrain, promote, place orders, or send
alerts. The existing daily research timer continues to run normally.

The canonical `ai_trading.tools.scheduled_evidence_verification` module checks:

- A report generated on the expected day, terminal workflow status, actual step
  results, exit codes, and output artifacts for training, selection, accounting,
  and paper review. Plans and skipped steps cannot prove execution.
- Whether training verified its data or blocked with an explicit reason.
- A current operator report containing accounting and completion diagnostics.
- September 8 session boundaries, decisions, orders, fills, TCA and source
  integrity, once the session has closed. Zero recorded executions are reported
  explicitly and do not qualify execution evidence or prove strategy abstention.

Each run saves an immutable timestamped report and refreshes:

`/var/lib/ai-trading-bot/runtime/research_reports/evidence_verification/latest.json`

An executed workflow with blocked model gates is distinguished from an
automation failure. A pending report identifies evidence that still cannot be
verified. If a workflow is late or fails, the checker retains the incomplete
status; it does not rerun trading or research automatically.

## Fee evidence

The current pinned paper Trading API client has no document-retrieval method.
Alpaca documents account statements and trade confirmations through the separate
[Broker API document endpoint](https://docs.alpaca.markets/us/reference/getdocsforaccount).
This is an access limitation of the configured integration, not a claim that no
confirmation exists. No transaction-level confirmation was obtained, and paper
Trading API credentials were not repurposed for a different API.

The accounting report now aggregates the 83 fetched fee activities into 35 USD
accounting-date groups, preserving charges, credits, net charges, invalid amounts,
and currency. It does not create zero-fee rows for days without fee activity.
Reported accounting dates are not assumed to be certified trade-session dates,
and daily amounts are not allocated across fills. Per-fill fee validation
remains pending.

The current read-only run and aggregate fee evidence are saved under
`artifacts/scheduled_evidence_verification/`. The current result is
`awaiting_or_incomplete_run` and `awaiting_session_close`, as expected before
the scheduled events.

## Validation and operation

Tests cover stale/planned/skipped workflows, current operator diagnostics,
accounting-date/currency separation, credits, invalid amounts, and a session with
no recorded executions. Unit files were checked with `systemd-analyze verify`.
Changed-file validation is recorded in `/tmp/scheduled-evidence-validation.log`;
the final focused checks are in `/tmp/scheduled-evidence-final-tests.log`.
The changed-file validation passed 732 tests, lint, mypy and compilation;
the final focused suite passed seven tests. The installed user service was
started once and returned `Result=success`, `ExecMainStatus=0`. Its saved result
correctly remains pending before the target events. The timer is enabled and
the installed units match their checked-in sources. No administrator password
was needed for the user-manager deployment.

To inspect execution, use `systemctl --user status ai-trading-evidence-verification.timer`
and `journalctl --user -u ai-trading-evidence-verification.service`, then read the saved
JSON. No notification delivery is configured.

Rollback: use `systemctl --user disable --now ai-trading-evidence-verification.timer`, remove only these two
dated verification units, and revert this task's verifier and daily-fee aggregation
changes. Preserve the earlier accounting, fee-contract and candidate fixes.
