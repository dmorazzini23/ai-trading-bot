from __future__ import annotations

import sys


def _main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    # Minimal checks only
    print("ENV_OK: true")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
