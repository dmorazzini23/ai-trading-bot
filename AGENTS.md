# AGENTS: Operating Contract & Playbook

**Last updated:** 2025-08-25
**Runtime targets:** Ubuntu 24.04 • Python 3.12 • zoneinfo-only • Flask on :9001

This document defines what automated agents (including LLM coding agents) may do in this repository and how they must do it.

## 1) Non-negotiable invariants
- **Python:** 3.12 (`requires-python=">=3.12"`). Tooling targets **py312**.
- **Timezones:** Use stdlib **zoneinfo**; never depend on `pytz`.
- **Service:** `ai-trading.service` runs a Flask API on **0.0.0.0:9001**.
- **Health:** `GET /healthz` must always return JSON and never 500.
- **Config access:** via `ai_trading.config.management`:
  - `get_env(key, default=None, cast=None, required=False)`
  - `reload_env(path=None, override=True)` (use sparingly, not in hot paths)
  - `SEED` (default **42**; may be overridden in `.env`)
- **Single Alpaca SDK in production:** prefer `alpaca-trade-api`. Do **not** mix with `alpaca-py` in prod.
- **No production shims:** Do **not** introduce or rely on `optional_import(...)` in runtime code paths.

## 2) Performance & resource guardrails
- Target machine: 2 GB RAM / 1 vCPU. Keep startup lean.
- Gate heavy imports (`pandas`, `sklearn`, `torch`, `matplotlib`, `gymnasium`) inside functions unless needed at startup.
- Never write large artifacts to disk by default. No diagnostics tarballs in CI or systemd units.

## 3) Editing & PR policy for agents
- Use small, reviewable PRs grouped by concern (docs vs code).
- When changing behavior, update docs in the same PR.
- Include:
  - Motivation & before/after.
  - Test plan (commands and expected output).
  - Rollback plan.

## 4) Installation & local run (docs reference)
```bash
python -m pip install -U pip
pip install -e .
ruff check .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
curl -sf http://127.0.0.1:$HEALTHCHECK_PORT/healthz
```

## 5) Timezone examples (docs reference)

```py
from datetime import datetime
from zoneinfo import ZoneInfo
now_ny = datetime.now(ZoneInfo("America/New_York"))
```

## 6) Config examples (docs reference)

```py
from ai_trading.config import management as config

api_port = config.get_env("API_PORT", "9001", cast=int)
paper = config.get_env("ALPACA_PAPER", "true", cast=bool)
seed = config.SEED  # defaults to 42

# Reload if you must re-read .env after edits (avoid in hot paths):
config.reload_env()
```

## 7) Health & metrics endpoints (docs reference)

* Routes: `GET /healthz`, `GET /metrics`
* `/healthz` JSON: `{"ok": true, "ts": "...", "service": "ai-trading"}`
* `/metrics` exposes Prometheus format
* Set `RUN_HEALTHCHECK=1` to serve these on `$HEALTHCHECK_PORT` (default **9001**)
* Requirement: Endpoints must not raise exceptions; log and return `ok: false` if degraded.

## 8) Alpaca SDK stance

* Production default: `alpaca-trade-api`.
* If switching to `alpaca-py`, update all broker modules, tests, and docs in the same PR. Do not document both as active simultaneously.

## 9) What agents must not do

* Don’t introduce `pytz`.
* Don’t re-add `requirements.txt` workflows.
* Don’t bind new ports without updating docs & systemd.
* Don’t add background file dumps or large archives by default.

## 10) Troubleshooting quick map (docs reference)

* **ImportError/AttributeError** for `get_env` or `reload_env`: update to latest `ai_trading/config/management.py`.
* **/health returns 500:** fix the route; health must swallow exceptions and return JSON.
* **Alpaca errors about missing client:** install the single chosen SDK and align import paths.
