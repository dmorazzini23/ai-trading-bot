# IEX provenance and gap review — September 17, 2026

## Follow-up implementation

The intraday gap inconsistency described below is now corrected. A shared
`validate_day_sleeve_history` preserves the serving timestamp checks and is
called before after-hours/replay-aligned indicator computation. Replay feature
caches use a new schema key and validate history on cache hits. Offline model
replay enforces this rule for explicit 5Min declarations; generic other-timeframe
replay is not reclassified. Newly trained replay models will declare the
validated timeframe. No existing artifact was modified and no training was run.

Changed-file validator: 193 tests passed, lint/type/compile checks passed. Final
focused regressions: 12 passed. Extended replay-inclusive run: 190 passed and
four failures, all reproduced with the HEAD pre-change offline replay function.
Those existing netting/trade-count tests remain unresolved; do not claim a clean
full suite. Logs and exact test names are in docs/CODEX_HANDOFF.md and
`/tmp/iex-history-baseline.log`. Live health remains healthy. No restart was
needed because the running serving timestamp rule is unchanged.

This check rejects internal intraday gaps; it does not prove complete sessions,
minute-level aggregate coverage, feed equivalence or original training provenance.
It may reject entire incomplete symbol histories, intentionally matching serving
strictness. Existing model qualification remains unchanged.

Scope: recover existing evidence, compare gap handling, inspect natural runtime
events. Paid SIP is unavailable by user decision. No new research or training.

## Recovered provenance

The selected day model's registry, original artifact manifest and original report
`/var/lib/ai-trading-bot/runtime/research_reports/after_hours_training_20260717_200036.json`
agree on IEX and dataset fingerprint
`7b82924e435aa47ea6998fddff59c06ccda28c8c0f16c187b1d213cbad4395b5`.
Manifest creation: July 17 20:03:49 UTC; declared training range May 21 13:50
through July 17 19:50 UTC. Metadata declares 5Min and the feature contract version.
These inspected records do not recover adjustment, session, history or finality
policies, or a complete versioned input contract. Fingerprint agreement links
records; it does not reconstruct original input data or prove compatibility.
Do not rewrite old provenance using present configuration.

## Gap handling finding: unresolved consistency issue

- Live `features/day_sleeve.py::build_day_sleeve_features` rejects any same-date
  missing/irregular five-minute interval in the supplied history.
- Training `training/after_hours.py::_validate_day_sleeve_bar_cadence` accepts
  aligned multiples of five minutes, including a missing interval. Indicators
  are computed over surviving rows. `_build_symbol_dataset` excludes labels
  crossing a gap, but that does not exclude the gap from indicator history.
- Replay-aligned `_feature_frame` computes indicators after
  `_sanitize_model_feature_index`; that helper sorts and handles duplicate/null
  labels but does not enforce the serving five-minute continuity requirement.
  This is a helper-level finding, not proof every replay input contains gaps.

The existing missing-bar training regression passed: labels cannot cross the
gap. It does not establish feature-history equivalence. The 23 input-contract
and serving tests also passed. No evidence supports loosening serving validation.

A missing IEX minute is not automatically a missing provider five-minute bar;
do not apply a requirement of five observed minute bars to provider aggregates
without defining that separate policy. The confirmed inconsistency concerns
missing five-minute intervals in model history.

Next corrective scope: share explicit history eligibility across day training,
serving and the corresponding replay path, with a synthetic missing-interval
regression. Preserve live strictness, label timing and qualification gates;
do not fabricate prices, retrain or broaden replay scope. Current audit does not
claim this correction is implemented.

## Runtime and validation

At approximately 03:45 UTC, service active, NRestarts=0, health healthy; existing
required_model_stale and replay_live_parity_gate_failed flags remain. The 87
available journal lines since deployment at 03:31:15 contain no new skew breach,
closeout event or error match. Market remains closed; natural verification is
pending. No background watcher was installed.

Commands: targeted pytest contract/serving files: 23 passed; after-hours and
serving selection `-k 'missing_bar or cadence or irregular or gap'`: 1 passed,
121 deselected. Logs: `/tmp/iex-contract-tests.log`, `/tmp/iex-gap-tests.log`.
Runtime artifacts: `/tmp/iex-followup-journal.log`, `/tmp/iex-followup-health.json`.
No runtime changes or restart. Documentation and diff checks passed.
