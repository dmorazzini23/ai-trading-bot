# Developer Report

> Historical note: This file is an archival implementation snapshot. It may
> mention older filenames, scripts, env vars, or deployment assumptions. For
> current runtime behavior, use `AGENTS.md`, `README.md`, `ARCHITECTURE.md`,
> `API_DOCUMENTATION.md`, `DEPLOYING.md`, `docs/DEPLOYING.md`, and
> `docs/OPERATIONS.md`.

## Verification & Tests

Run the lightweight verification suite:

```sh
make verify
```

Execute the test suite:

```sh
make test
```

Integration and broker tests automatically skip when required `ALPACA_*` credentials are absent or markets are closed.

- Added console-script entry points and corrected CLI smoke test.
