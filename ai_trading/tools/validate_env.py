"""Environment validation entrypoint used by tests."""

from __future__ import annotations

import os

REQUIRED_VARS = (
    "WEBHOOK_SECRET",
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "ALPACA_BASE_URL",
)


def _validate() -> tuple[bool, list[str]]:
    missing = [k for k in REQUIRED_VARS if not os.getenv(k)]
    return (not missing, missing)


def _main(argv: list[str] | None = None) -> int:
    ok, missing = _validate()
    if ok:
        return 0
    print("missing vars: " + ",".join(missing))  # noqa: T201
    return 1


def main(argv: list[str] | None = None) -> int:
    return _main(argv)


if __name__ == "__main__":  # pragma: no cover - manual execution
    raise SystemExit(_main())
