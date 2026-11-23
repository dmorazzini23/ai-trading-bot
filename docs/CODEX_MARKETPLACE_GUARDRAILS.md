# Codex Marketplace Guardrails

This repository uses project-scoped Codex Marketplace hooks and skills as an
operator guardrail layer. These tools are for safer Codex work only; they are
not part of the trading runtime, model pipeline, broker path, or systemd
service.

## Installed Hooks

The project hook registry is:

```text
/home/aiuser/ai-trading-bot/.codex/hooks.json
```

Installed hook packages:

- `aitmpl-codex--automation--agents-md-loader`: loads `AGENTS.md` at Codex
  session start so project rules are visible to future agents.
- `aitmpl-codex--security--secret-scanner`: blocks `git commit` commands when
  staged files contain likely secrets. The project copy also detects Slack
  incoming webhook URLs.
- `aitmpl-codex--security--dangerous-command-blocker`: blocks catastrophic
  shell commands and protects critical paths such as `.git`, `.env`, and
  dependency manifests.
- `aitmpl-codex--development-tools--command-logger`: logs Codex tool usage to
  `~/.codex/command-log.txt` for auditability.
- `aitmpl-codex--automation--change-logger`: logs file mutations and
  non-read-only Bash commands to `.codex/critical_log_changes.csv`. The project
  copy redacts common secret patterns before persisting command snippets.
- `aitmpl-codex--quality-gates--plan-gate`: `PreToolUse` hook on `Edit`,
  `MultiEdit` and `Write`. Warns when source code is edited and no
  `*.spec.md` file modified in the last 14 days exists. Non-blocking.
- `aitmpl-codex--quality-gates--scope-guard`: `Stop` hook. Warns when
  git-modified files fall outside the files declared in a recent `*.spec.md`.
  Non-blocking, and silent when no spec exists.

## Installed Skills

The project skill directory is:

```text
/home/aiuser/ai-trading-bot/.codex/skills
```

Installed skills:

- `bug-hunt-swarm`: read-only multi-agent root-cause investigation.
- `review-swarm`: read-only multi-agent diff or file-scope review.
- `project-skill-audit`: recommends project-specific Codex skills from real
  project history.
- `orchestrate-batch-refactor`: coordinates larger refactors with explicit
  file ownership and validation. Use only with clear write scopes.
- `context7-cli`: fetches current library documentation with `ctx7`.

Existing project skills remain:

- `trading-ops-runbook`
- `trading-ops-shift`

## Installed Plugins

The project plugin marketplace file is:

```text
/home/aiuser/ai-trading-bot/.agents/plugins/marketplace.json
```

Installed plugin:

- `sentry`: official OpenAI plugin for Sentry issue/event triage. It requires
  Sentry authentication before it can inspect project issues.
- `hugging-face`: inspect models, datasets, Spaces, and research. It is
  registered in the marketplace file but disabled in `.codex/config.toml`.

## Rules For Future Agents

- These guardrails do not replace `bash scripts/agent_validate_changed.sh`.
- Do not install auto-git-add, smart-commit, auto-push, auto-format,
  broad auto-test, deployment, or Slack notification hooks in this repository.
- Do not route Codex hook notifications into `#all-beatwallstreet`; the trading
  bot already has its own deduped Slack alert path.
- Use `bug-hunt-swarm` and `review-swarm` read-only unless the user explicitly
  asks to patch.
- Use `orchestrate-batch-refactor` only for large scoped work with disjoint file
  ownership and targeted validation.
- Keep marketplace changes project-scoped unless the user explicitly asks for
  global Codex behavior.

## Verification

Run:

```bash
bash scripts/verify_codex_marketplace_guardrails.sh
```

This checks that the original five hooks, the five installed skills, and the
`sentry` plugin files exist and that forbidden hook types were not installed.
It does not check for `plan-gate`, `scope-guard` or the `hugging-face` plugin.

## Rollback

To remove a project-scoped hook or skill, delete its directory under
`.codex/hooks` or `.codex/skills` and remove the matching entry from
`.codex/hooks.json` when applicable (`plan-gate` has three entries under
`PreToolUse`, `change-logger` has four under `PostToolUse`, `scope-guard` has
one under `Stop`). To remove a plugin (`sentry` or `hugging-face`), remove its
entry from `.agents/plugins/marketplace.json` and delete `plugins/<name>`; for
`hugging-face` also remove its override in `.codex/config.toml`.
