from __future__ import annotations

import json
import os
import sys


def _main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    required = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    print(json.dumps({"ok": not missing, "missing": missing}))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(_main())
