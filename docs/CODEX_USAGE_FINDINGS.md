# Codex usage audit and applied changes

September 8, 2026. Source: local rollout telemetry, summarized without exporting
prompts, commands, credentials or reasoning. Artifact:
`artifacts/codex_efficiency/usage_audit.json`.

## Measured findings

Five recent user-task logs for this checkout were inspected; automatic guardian
and subagent sessions were excluded. Three older logs lack per-response usage
records and are explicitly marked unavailable, not zero usage. The other older
task has partial per-response records; this is not a complete account-usage audit.

For this long task, the snapshot contains 532 recorded responses:

| Measure | Observed |
| --- | ---: |
| Input tokens, including cached | 69,881,890 |
| Cached input tokens | 67,702,272 (96.88%) |
| Uncached input tokens | 2,179,618 |
| Output tokens, including reasoning | 292,133 |
| Reasoning subset of output | 76,457 |
| Median input tokens per response | 137,156.5 |
| Calls containing terminal polling | 90 |
| Detectably empty polling results | 11 |
| Tool outputs over 16,000 characters | 23 |

Repeated large context is the dominant observed token volume. Cached tokens
are not equivalent to uncached tokens for cost, and these counters cannot be
converted into subscription usage percentages or billed dollars. Polling and
large-output counts identify opportunities, not independently measured savings.
Tool output is measured in characters. Exact repeated-read detection is a lower
bound and does not identify every semantically repeated command.

The project already specified Astra/Low, while all 44 recorded contexts in this
task specified Astra/Medium. Therefore existing task overrides, not a missing
project default, explain the effort mismatch. A fresh task must verify its picker.

## Seven implemented workstreams

1. `docs/CODEX_HANDOFF.md` provides current state, evidence, checks and next action.
2. Project retained-tool-output budget reduced from 4,000 to 2,000 tokens; agent
   guidance defaults individual calls to 1,500 with full artifacts preserved.
3. Agent guidance uses 20-30 second waits and avoids repeated empty polling.
4. Validation reuse rules record exact coverage and invalidate on relevant edits.
5. Project overrides disable five optional integrations without uninstalling them
   or changing global settings. The CLI confirms Sites, Visualize, Slack and
   Supabase disabled; Hugging Face is configured disabled but was absent from
   the returned local list. Security, GitHub and Alpaca were confirmed enabled;
   Sentry's existing global enabled setting is preserved. Remote catalog lookup
   was unavailable in the sandbox, so full app-injected catalog reduction is
   not claimed. Safety hooks, approval policy and mandatory checks remain intact.
6. `scripts/codex_task.sh` offers explicit routine (Luna/Low), standard (Astra/Low)
   and deep (Astra/Medium) CLI presets. All model IDs exist in the local catalog.
   No automatic delegation, model API calls or in-flight model switch occurred.
7. `scripts/codex_usage_audit.py` makes the count-only audit repeatable. It deduplicates
   response IDs and separates cached/uncached input without double-counting reasoning.

These are workflow/configuration changes, not a claimed measured reduction yet.
Start a fresh bounded task using the handoff; compare input sizes and poll/output
counts for similar work while checking completion quality. The current transcript
and managed instructions cannot be removed by a project setting.

## Validation and rollback

Regression tests cover usage deduplication, cached-token accounting, empty polls,
missing telemetry, exclusion of reviewer tasks and omission of prompt text.
The launcher passes shell syntax checking and a CLI `--help` smoke test. TOML
parsing and a fresh CLI plugin listing verify the applicable project overrides.
Final validation with `bash scripts/agent_validate_changed.sh --market-hours
--skip-runtime-smoke` passed lint, type checks and 55 targeted tests. Output is
saved in `/tmp/codex-efficiency-validation.log`.
No trading runtime code changed in this task and no service restart was needed.

Undo only this task's `.codex/config.toml`/AGENTS efficiency edits, new helper
scripts and efficiency documentation to roll back. Preserve unrelated trading
changes and audit artifacts. The prior output budget was 4,000; remove the five
project plugin overrides to inherit global settings again.

Settings semantics are documented in the
[official Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).
