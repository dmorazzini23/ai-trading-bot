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


def _main() -> int:
    """Validate required environment variables and print summary."""
    ok, missing = _validate()
    if ok:
        print("env ok")  # noqa: T201
        return 0
    print("missing vars: " + ",".join(missing))  # noqa: T201
    return 1


def main() -> int:
    return _main()


__all__ = ["_main", "main"]


if __name__ == "__main__":  # pragma: no cover - manual execution
    raise SystemExit(_main())
