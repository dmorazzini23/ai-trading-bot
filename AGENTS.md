# AGENTS: Operating Contract & Playbook

**Last updated:** 2026-04-15
**Runtime targets:** Ubuntu 24.04 • Python 3.12.3 • zoneinfo-only • API on :9001 • health on shared `:9001` or standalone `HEALTHCHECK_PORT` (default :8081)

This document is the authoritative playbook for Codex-style editing in this repository. If reality drifts, update this file immediately.

---

## 1. Runtime Invariants
- **Python:** 3.12 (`requires-python=">=3.12"`). Target `cp312` wheels only.
- **Timezones:** Use stdlib **zoneinfo** exclusively; `pytz` is forbidden.
- **Service topology:**
  - Flask **API on :9001** (0.0.0.0:9001).
  - The main API serves `/health`, `/healthz`, and `/metrics` on `API_PORT`.
  - A standalone health app may be launched with `RUN_HEALTHCHECK=1 python -m ai_trading.app` on `HEALTHCHECK_PORT` (default 8081).
  - In the main runtime, if `HEALTHCHECK_PORT == API_PORT`, the API serves health endpoints on the shared port; otherwise an auxiliary health thread/process may bind `HEALTHCHECK_PORT`.
  - During startup validation, `RUN_HEALTHCHECK=1` requires `HEALTHCHECK_PORT != API_PORT`.
  - Health routes must be registered through the shared canonical helpers; do not add bespoke per-entrypoint health implementations.
- **Configuration access:** via `ai_trading.config.management` only (`get_env`, `reload_env`, `SEED`). No ad-hoc `os.environ` walks in runtime code.
- **SDK policy:**
  - Runtime: **`alpaca-py` 0.42.1**.
  - Avoid `alpaca-trade-api` in runtime paths.
- **Logging:** Structured logger only; no raw `print` in runtime or tests unless asserting stdout.
- **Shims:** **No shims.** Fix root causes directly; do not add compatibility layers or wrapper modules.

---

## 2. Editing Contract (Codex Editing Contract)
- All edits must use **`apply_patch`**.
- Keep diffs surgical; touch only what is necessary and preserve context.
- Do not introduce new shims, compatibility facades, or bulk rewrites.
- Respect existing module boundaries; no large refactors unless explicitly requested.
- Honor runtime invariants (SDK choice, ports, zoneinfo, logging).
- Ensure new imports do not execute work at import time.

---

## 3. Documentation Authority
- Current runtime authority lives in:
  - `AGENTS.md`
  - `README.md`
  - `ARCHITECTURE.md`
  - `API_DOCUMENTATION.md`
  - `DEPLOYING.md`
  - `docs/DEPLOYING.md`
  - `docs/OPERATIONS.md`
- Root-level `*_SUMMARY.md`, `*_FIX_*`, `*_REPORT.md`, `*_REMOVAL_*`, and similar implementation-snapshot documents are archival unless explicitly refreshed. Do not treat them as the sole source of truth for ports, env vars, entrypoints, or deployment behavior.

## 4. Validation Requirements
Agents must run and report these checks when changing runtime or library code (docs-only changes may explain why checks were skipped):
- `pytest -q`
- `ruff` (limit to changed paths when possible)
- `mypy` (at least on changed files/modules)
- `bash scripts/typecheck_strict.sh`
- `python3 -m py_compile $(git ls-files '*.py')`

Always add or update unit tests when fixing bugs or adding behavior.

---

## 5. PR Deliverables
Every agent-authored PR must include:
- **WORKLOG** — summary of intent, root cause, and scope.
- **PATCHSET** — list of `apply_patch` diffs (no file dumps or editors).
- **VALIDATION** — commands executed (`pytest -q`, `ruff`, `mypy`, `py_compile`) with outcomes.
- **RISK & ROLLBACK** — risk assessment and how to revert.

---

## 6. Greppable Anchors
Use these stable strings to anchor surgical edits:
- `IMPORT_PREFLIGHT_OK`
- `HEALTHCHECK_PORT_CONFLICT`
- `TRADING_PARAMS_VALIDATED`
- `ExecutionEngine initialized`

---

## 7. Operational Notes
- The checked-in `packaging/systemd/ai-trading.service` currently exposes `/healthz` on `:9001` because it sets `HEALTHCHECK_PORT=9001`.
- Validate the surface that matches the deployment mode:
  - packaged main service: `curl -sS http://127.0.0.1:9001/healthz`
  - standalone health app: `curl -sS http://127.0.0.1:${HEALTHCHECK_PORT}/healthz`
- **Repo ownership:** this checkout is owned by `aiuser`. Do not run repo-local
  development commands with `sudo` because they create root-owned source files,
  caches, test artifacts, venv packages, and model directories that later break
  tests and editing. Use the repo venv as `aiuser`, for example
  `./venv/bin/python`, `./venv/bin/pytest`, `./venv/bin/ruff`, and
  `./venv/bin/mypy`. `sudo` is acceptable for system-level service control that
  does not write into the checkout, such as
  `sudo systemctl restart ai-trading.service`,
  `sudo systemctl status ai-trading.service`, and
  `sudo journalctl -u ai-trading.service`.
- Health endpoints must degrade gracefully (never raise uncaught exceptions); log structured diagnostics.
- Keep health response construction and route registration on the shared canonical path; entrypoints may adapt payload context but must not fork health semantics.
- Respect fail-fast configuration: missing required env vars should raise immediately with actionable errors.

---

## 8. Anti-Patterns to Avoid
- Reintroducing shims, optional import helpers, or dynamic SDK swaps.
- Adding new direct `os.getenv` / `os.environ` runtime access outside `ai_trading.config.management`, except for tightly justified non-runtime validation code.
- Adding raw `print` statements or silent exception handling.
- Migrating runtime off pinned `alpaca-py` without explicit approval.
- Conflating API and health ports, or assuming shared-port deployment without checking `HEALTHCHECK_PORT`.
- Running `sudo pytest`, `sudo python`, `sudo pip`, `sudo git`, or other
  repo-local write commands from `/home/aiuser/ai-trading-bot`.
