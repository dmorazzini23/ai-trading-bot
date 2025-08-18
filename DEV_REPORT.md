# Developer Report

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
