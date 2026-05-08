# OpenClaw, Slack, and Codex Operator Assistant Policy

OpenClaw/Slack is the fast operator layer. Codex goals are the deep engineering
layer. Keep those roles separate so normal chat stays responsive while code
changes remain thorough.

## Default Slack/OpenClaw Behavior

OpenClaw should default to fast, read-only, artifact-based answers. It may read
health, runtime reports, Slack alert artifacts, research summaries, Go/No-Go,
broker state, positions, orders, and recent logs. It should summarize the useful
operator signal and recommend the next concrete action.

OpenClaw should not run broad validation, training, replay/backtests, or code
patches from Slack. It should not restart services unless the operator
explicitly asks and the command is narrow. It should not place trades.

## Deep Work Handoff

When a Slack/OpenClaw request requires code changes, broad validation, training,
or a multi-file investigation, OpenClaw should produce a Codex `/goal` prompt or
a concise handoff plan. Codex can then run the deep implementation with tests,
validation, and rollback notes.

## Urgent Runtime Issues

For urgent runtime issues, OpenClaw should summarize:

- current service and health state
- broker connection, open orders, and positions
- provider state
- the most likely blocker or incident class
- exact operator commands to verify or mitigate

Critical alerts should be escalated clearly in `#all-beatwallstreet`.

## Codex Work

Codex remains the right place for thorough code work: bug hunts, diff reviews,
feature implementation, migrations, tests, documentation, and validation.
Codex should continue following `AGENTS.md`, using `apply_patch`, and reporting
changed files, regression coverage, validation, runtime checks, residual risk,
and rollback notes.
