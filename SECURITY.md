# Security Policy

## Reporting A Vulnerability

Please do not open a public issue for a suspected vulnerability involving
credentials, broker access, order execution, account data, or deployment
hardening.

Use GitHub private vulnerability reporting if it is enabled for this repository.
If it is not enabled, contact the repository owner directly and include:

- Affected commit or release.
- Steps to reproduce.
- Expected and observed behavior.
- Whether broker credentials, order submission, secrets, or account data are
  involved.

## Trading Safety Scope

Security issues in this project include traditional software vulnerabilities and
trading-safety failures that could cause unintended broker actions, credential
exposure, degraded live execution, or bypassed risk controls.

Examples worth reporting:

- Exposure or logging of Alpaca credentials, OAuth tokens, webhook secrets, or
  operator tokens.
- Live-mode execution paths that bypass required broker, data-feed, or risk
  validation.
- Health, diagnostics, or operator endpoints that expose sensitive data without
  the intended controls.
- Dependency or import behavior that swaps the pinned runtime SDK or silently
  falls back to an unsafe implementation.

## Operational Expectations

- Start with paper trading.
- Keep live credentials out of the repository.
- Use `ALPACA_TRADING_BASE_URL` for broker endpoint selection.
- Run configuration validation before starting the runtime.
- Treat generated logs, runtime artifacts, and model files as sensitive.
