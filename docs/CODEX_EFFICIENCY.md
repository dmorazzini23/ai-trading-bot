# Codex efficiency

Project defaults in `.codex/config.toml` keep Astra, use low reasoning and
standard speed, and limit each tool result retained in history to 2,000 tokens.
This is a history-output budget, not a per-task spending cap. Preserve full
logs as artifacts and read the relevant sections when a result is truncated.

Existing tasks can retain model-picker overrides. For the next run, select
Astra, Low reasoning, and standard speed. Raise reasoning for a specific
difficult investigation when low proves insufficient. Return to Low afterward.
Project configuration applies when the project is trusted; verify the picker
in a new task rather than assuming an existing task adopted changed defaults.

For a fresh task continuing long work, provide a concise handoff with current
state, changed files, checks, blockers, and the next action. Continue related
work within this saved project. Avoid importing the entire old transcript.

Run commands as aiuser from the repository root using the existing venv.
Validation commands and runtime invariants remain in AGENTS.md:

- Runtime/library changes: `bash scripts/agent_validate_changed.sh`.
- During market hours: append `--market-hours`.
- Documentation-only changes: `bash scripts/agent_validate_changed.sh --docs-only`.
- Full validation requires the conditions documented in AGENTS.md.

No services, trading parameters, hooks, security checks, or broker access are
changed by these defaults. Experimental context management and custom context
window/compaction limits are not enabled as unverified cost-saving measures.

## Working defaults

- Start with `docs/CODEX_HANDOFF.md`. Read exact referenced artifacts only when
  needed; do not reload the full conversation or historical report directories.
- Use `max_output_tokens: 1500` for routine calls. For JSON, extract requested
  fields in one Python command. For logs, save full output and return counts or
  a short excerpt. Avoid `cat` on manifests, journals or large reports.
- Batch independent reads. Sequence edits and their validation. Use explicit
  working directories and fail-fast shell execution for dependent operations.
- Once a job returns a session ID, use a 20-30 second `write_stdin` wait. Do not
  alternate empty one-second polls and unchanged `tail` calls. Give a meaningful
  progress update within 60 seconds. New user input should remain interruptible.
- Keep scope fixed. Complete the requested outcome, report missing external
  evidence, and do not launch a new experiment to fill waiting time.
- Reuse a passing validation only when its covered files and dependencies have
  not changed. Record the command, scope, outcome and artifact in the handoff.
  New changes require relevant checks; this does not waive AGENTS.md validation.
- At a meaningful milestone, hand off to a fresh task instead of carrying the
  full transcript forward. Task creation still requires an explicit user request.

## Model presets

`bash scripts/codex_task.sh standard` starts Astra with explicit Low effort.
`routine` starts the locally catalogued GPT-5.6 Luna with Low effort, suitable
for bounded read-only status or mechanical work. `deep` starts Astra at Medium
for difficult diagnosis/research design. Append normal Codex arguments or a
prompt. These presets do not alter permissions or spawn agents automatically.
They are CLI launch presets, not controls for an already-running desktop task.
In the desktop picker, use Astra/Low for the next task and raise effort only
when the problem warrants it. The current long task retained Medium despite
the pre-existing project Low default. Global defaults are unchanged for other
projects.

## Optional plugin scope

Project overrides disable Supabase, Slack, Hugging Face, Sites and Visualize
by their installed identifiers. Security, GitHub, Alpaca and Sentry are retained.
Nothing is uninstalled and other projects' settings are unchanged. Re-enable
the specific override when a task actually needs it. Injected app catalogs and
managed/system instructions may remain outside project control; reductions in
token use must be measured in a fresh task rather than assumed.

## Repeatable usage measurement

```bash
./venv/bin/python scripts/codex_usage_audit.py \
  --limit 5 --output artifacts/codex_efficiency/usage_audit.json
```

This reads the latest five user-task rollouts for the current checkout, excluding
automatic reviewer/subagent sessions. It exports counts only: deduplicated
per-response usage, cached versus uncached input, input size, effort, large tool
outputs, detectable empty polls and exact repeated reads. Missing usage remains
explicitly unavailable. Character counts are not token counts. Input already
includes cached tokens; output already includes reasoning tokens. Do not add
these subsets twice or translate them into billed dollars/subscription quotas.

Compare a fresh task after these changes with tasks of similar scope. Look for
lower median input per response, fewer oversized outputs and empty polls, and
unchanged completion/validation quality. No savings percentage is claimed yet.

Official settings reference:
https://learn.chatgpt.com/docs/config-file/config-reference

Official speed reference:
https://learn.chatgpt.com/docs/agent-configuration/speed
