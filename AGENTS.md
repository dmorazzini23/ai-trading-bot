# AGENTS: Operating Contract & Playbook

**Last updated:** 2025-10-02  
**Runtime targets:** Ubuntu 24.04 • Python 3.12.3 • zoneinfo-only • API on :9001 • Health server on `HEALTHCHECK_PORT` (default :8081)

This document is the authoritative playbook for Codex-style editing in this repository. If reality drifts, update this file immediately.

---

## 1. Runtime Invariants
- **Python:** 3.12 (`requires-python=">=3.12"`). Target `cp312` wheels only.
- **Timezones:** Use stdlib **zoneinfo** exclusively; `pytz` is forbidden.
- **Service topology:**
  - Flask **API on :9001** (0.0.0.0:9001).
  - Dedicated **Health server on `HEALTHCHECK_PORT`** (default 8081) serving `GET /healthz` with JSON and HTTP 200 when healthy.
  - If `HEALTHCHECK_PORT == API_PORT`, the API may serve health endpoints on the shared port; otherwise run health in its own thread/process.
- **Configuration access:** via `ai_trading.config.management` only (`get_env`, `reload_env`, `SEED`). No ad-hoc `os.environ` walks in runtime code.
- **SDK policy:**
  - Runtime: **`alpaca-trade-api` 3.2.0**.
  - Dev/test-only: **`alpaca-py (dev/test-only)`** 0.42.0 is available for mocks, fixtures, or tooling—never ship it in production execution paths.
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

## 3. Validation Requirements
Agents must run and report these checks when changing runtime or library code (docs-only changes may explain why checks were skipped):
- `pytest -q`
- `ruff` (limit to changed paths when possible)
- `mypy` (at least on changed files/modules)
- `python -m py_compile $(git ls-files '*.py')`

Always add or update unit tests when fixing bugs or adding behavior.

---

## 4. PR Deliverables
Every agent-authored PR must include:
- **WORKLOG** — summary of intent, root cause, and scope.
- **PATCHSET** — list of `apply_patch` diffs (no file dumps or editors).
- **VALIDATION** — commands executed (`pytest -q`, `ruff`, `mypy`, `py_compile`) with outcomes.
- **RISK & ROLLBACK** — risk assessment and how to revert.

---

## 5. Greppable Anchors
Use these stable strings to anchor surgical edits:
- `IMPORT_PREFLIGHT_OK`
- `HEALTHCHECK_PORT_CONFLICT`
- `TRADING_PARAMS_VALIDATED`
- `ExecutionEngine initialized`

---

## 6. Operational Notes
- The API and health server are distinct by default. Validate both surfaces when debugging deployments:
  - `curl -sS http://127.0.0.1:9001/` (API route as applicable)
  - `curl -sS http://127.0.0.1:${HEALTHCHECK_PORT}/healthz`
- Health endpoints must degrade gracefully (never raise uncaught exceptions); log structured diagnostics.
- Respect fail-fast configuration: missing required env vars should raise immediately with actionable errors.

---

## 7. Anti-Patterns to Avoid
- Reintroducing shims, optional import helpers, or dynamic SDK swaps.
- Adding raw `print` statements or silent exception handling.
- Migrating runtime off `alpaca-trade-api` without explicit approval.
- Conflating API and health ports, or assuming shared-port deployment without checking `HEALTHCHECK_PORT`.

