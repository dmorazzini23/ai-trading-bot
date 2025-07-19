# 📄 AGENTS.md

## AI-Only Maintenance & Refactoring Policy

> This repository is maintained by a single human engineer (Dom) using advanced AI agents (Codex, GPT-4o).
> This document exists to **guide these AI agents** in how they operate on this codebase.

---

## 🔍 Prime Directives

### ✅ Maintain critical production logic

* **Never rewrite or remove the core execution logic** in:

  * `bot_engine.py`
  * `runner.py`
  * `trade_execution.py`
* Only evolve these files incrementally (via `update_functions` or `append_content`), not full replacements.

### ✅ Keep the trading bot stable

* Always preserve existing safety checks, such as:

  * Position sizing via Kelly or volatility constraints.
  * Health checks in `pre_trade_health_check` and related functions.
* Any changes must **not risk unexpected trades** or skipping critical risk logic.

---

## 🚀 Specific AI Agent Behavior

### Logging & Debugging

* Always use the centralized `logger` module for all output.
* Never introduce raw `print()` statements.

### Time & UTC

* Globally replace any `datetime.utcnow()` with `datetime.now(datetime.UTC)`.

### Parallel & Data

* When refactoring concurrency (like `concurrent.futures`), ensure:

  * No data race or shared state corruption.
  * That `to_parquet` still functions using `pyarrow` engine.

---

## 🧪 Testing & Validation

* After any AI modifications, always run:

  ```
  pytest -n auto --disable-warnings
  ```
* Ensure `0 failed`, no broken logs, and no trade logic skipping.

---

## 📝 Minimalistic Commit Messages

* Since only Dom reviews them, be concise, e.g.:

  ```
  fix: add signal_period to hyperparams.json
  refactor: logger test ensure rotating path
  ```

---

## 🤖 Special Codex Directives

When running a Codex prompt to refactor or fix:

1. **Never use `replace_content` on:**

   * `bot_engine.py`, `runner.py`, or `trade_execution.py`.
2. Prefer:

   * `update_functions`, `append_content`, or targeted `sed` replacements.
3. Annotate new code with:

   ```python
   # AI-AGENT-REF: short note about the change
   ```

---

## 🛠 Safe File Areas

| File / Module       | AI can freely modify                  |
| ------------------- | ------------------------------------- |
| `signals.py`        | Extend indicators carefully           |
| `logger.py`         | Update handlers or test hooks         |
| `tests/`            | Adapt tests for new behavior          |
| `requirements*.txt` | Add packages like `pyarrow` if needed |
| `ai_trading/*`      | Capital scaling & trade logic modules |

---

## 📂 Minimal Human Audit

Since this is **AI + Dom only**, be strict:

* Any AI-proposed changes that violate these principles must be **explicitly confirmed** by Dom before being applied.

---

## ✅ Reminder to Codex

> “Respect these directives. If unsure, halt and wait for Dom’s explicit prompt.”

---

## 🔒 Additional Safeguards

* Never hardcode API keys, secrets, or credentials. Always load them via environment variables managed through `.env` files and `validate_env.py`.
* Do not replace or degrade parallel indicator processing (`concurrent.futures`) with purely serial logic unless explicitly instructed.
* Never downgrade essential libraries such as `pandas`, `numpy`, or `scikit-learn` below the versions pinned in `requirements.txt` or `pyproject.toml`.

---

## 🚀 Reminder on Project Scope

This repository is strictly an advanced **AI-powered trading platform**, with extensive use of:

* financial indicators,
* backtesting engines,
* multi-timeframe strategy orchestration,
* and robust live trading safety checks.

Unrelated general-purpose refactor patterns (e.g. CRUD templates, simplistic microservices restructuring) should **never override these domain priorities**.

---

## 📜 Recent PR Prompts

- **PR1**: initial package restructure into `ai_trading` modules
- **PR2**: introduced async trade execution helpers
- **PR3**: migrated configuration to `pydantic-settings`
- **PR4**: tightened CI coverage checks

---

## ✅ Final Instruction to AI Agents

> **“When in doubt, defer to Dom for explicit clarification.
> Always prioritize safety, precision, and trading domain expertise.”**

---
