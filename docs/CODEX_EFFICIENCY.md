# Codex efficiency

Project defaults in `.codex/config.toml` keep Astra, use low reasoning and
standard speed, and limit each tool result retained in history to 4,000 tokens.
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

Official settings reference:
https://learn.chatgpt.com/docs/config-file/config-reference

Official speed reference:
https://learn.chatgpt.com/docs/agent-configuration/speed
