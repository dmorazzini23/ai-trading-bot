# Documentation index

Current documents, grouped by purpose. For setup commands and a quick tour of
the runtime, see [index.md](index.md). Authority order is in `AGENTS.md`
section 3.

## Operating the service
- [DEPLOYING.md](DEPLOYING.md): deployment guide.
- [OPERATIONS.md](OPERATIONS.md): day-to-day operations, reports and retention.
- [RELEASE_IDENTITY.md](RELEASE_IDENTITY.md): release identity preflight checks.
- [RUNTIME_RECOVERY.md](RUNTIME_RECOVERY.md): backups and restore rehearsal.
- [INDEPENDENT_HOST_MONITOR.md](INDEPENDENT_HOST_MONITOR.md): off-host failure alert.
- [MARKET_PREFLIGHT.md](MARKET_PREFLIGHT.md): scheduled market preflight report.
- [LIVE_CAPITAL_RUNBOOK.md](LIVE_CAPITAL_RUNBOOK.md): path from paper to a real-money canary.
- [runbooks/](runbooks/): incident runbooks (broker outage, kill switch, stale data and so on).
- [SECRETS_MANAGER_MIGRATION.md](SECRETS_MANAGER_MIGRATION.md): moving secrets to AWS Secrets Manager.
- [slo_alerts.md](slo_alerts.md), [rate_limits_and_slos.md](rate_limits_and_slos.md), [degraded_data_playbook.md](degraded_data_playbook.md): alerts, limits and degraded-data handling.

## Working on the code
- [DEVELOPMENT.md](DEVELOPMENT.md): development environment setup.
- [CODEX_HANDOFF.md](CODEX_HANDOFF.md): current handoff, read first when continuing work.
- [CODEX_EFFICIENCY.md](CODEX_EFFICIENCY.md), [CODEX_MARKETPLACE_GUARDRAILS.md](CODEX_MARKETPLACE_GUARDRAILS.md), [CODEX_MCP_AND_SKILLS_SETUP.md](CODEX_MCP_AND_SKILLS_SETUP.md): agent working rules and tooling.
- [SUPPORTED_TRADING_PATH.md](SUPPORTED_TRADING_PATH.md): the supported trading path.
- [LEGACY_INVENTORY.md](LEGACY_INVENTORY.md): retained legacy surfaces.
- [CENTRALIZED_IMPORTS.md](CENTRALIZED_IMPORTS.md), [coding/](coding/), [persistence/](persistence/): code conventions and design notes.
- [ARTIFACT_AUTHORITY.md](ARTIFACT_AUTHORITY.md): which artifacts count as evidence.

## Configuration and data
- [API_KEY_SETUP.md](API_KEY_SETUP.md), [BROKER_ACCOUNT_CONFIGURATION.md](BROKER_ACCOUNT_CONFIGURATION.md), [provider_configuration.md](provider_configuration.md), [data-providers.md](data-providers.md): keys, broker account and data providers.
- [SEED_TRADE_HISTORY.md](SEED_TRADE_HISTORY.md), [TRADE_UPDATES_STREAM.md](TRADE_UPDATES_STREAM.md): trade history seeding and the trade-updates stream.

## Strategy, risk and models
- [ENTRY_EXIT_SIGNALS.md](ENTRY_EXIT_SIGNALS.md), [MAX_FACTOR.md](MAX_FACTOR.md), [risk_engine.md](risk_engine.md), [precision_and_costs.md](precision_and_costs.md), [peak_performance.md](peak_performance.md): signals, stops, risk and costs.
- [model_governance.md](model_governance.md): model promotion framework.
- [SMALL_ACCOUNT_RISK_CONTRACT.md](SMALL_ACCOUNT_RISK_CONTRACT.md), [SMALL_ACCOUNT_EVALUATION.md](SMALL_ACCOUNT_EVALUATION.md): small-account limits.
- [ROBUSTNESS_AUDIT.md](ROBUSTNESS_AUDIT.md), [TRADING_SYSTEM_CHECKLIST.md](TRADING_SYSTEM_CHECKLIST.md): robustness standard and checklist.
- [institutional_roadmap_30_60_90.md](institutional_roadmap_30_60_90.md), [phase_gated_roadmap.md](phase_gated_roadmap.md), [acceptance_matrix.md](acceptance_matrix.md): roadmaps.
- [PORTFOLIO_MANAGER_MODE_PROPOSAL_20261025.md](PORTFOLIO_MANAGER_MODE_PROPOSAL_20261025.md): portfolio manager mode proposal.

## Research reset record
[RESEARCH_AUTOMATION.md](RESEARCH_AUTOMATION.md) describes the after-hours research loop.

These are the registered contracts, amendments, trial rules and results. They
are a fixed record and are not edited or moved. Start with
[RESEARCH_RESET.md](RESEARCH_RESET.md) and
[RESEARCH_RESET_REVIEW_20261114.md](RESEARCH_RESET_REVIEW_20261114.md); the
`RESEARCH_RESET_AMENDMENT_*`, `*_TRIAL_*`, `FOCUSED_RESEARCH_*` and
`REVISED_RESEARCH_*` files and the earlier research notes and decision records (including the cost, fee and quote-staleness studies and the model replacement decision) sit next to them.
The rules the code reads are in `config/research_reset.json`.

## Archive
[archive/](archive/) holds history: dated implementation snapshots, old audits
and reports, and the handoff sections older than November 15, 2026. Nothing in
it is current guidance; use it only to look up how something was done.
