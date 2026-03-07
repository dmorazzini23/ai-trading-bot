#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PATTERN = re.compile(r"\bos\.(?:getenv|environ)\b")

# Ratchet limits for runtime-critical hot paths. These modules must use
# ``ai_trading.config.management`` helpers instead of ad-hoc env access.
# Any non-zero exception here must be explicitly justified and reduced over time.
MAX_DIRECT_ENV_TOUCHES = {
    "ai_trading/main.py": 0,
    "ai_trading/__main__.py": 0,
    "ai_trading/logging/__init__.py": 0,
    "ai_trading/http/pooling.py": 0,
    "ai_trading/data/provider_monitor.py": 0,
    "ai_trading/execution/live_trading.py": 1,
    "ai_trading/core/bot_engine.py": 2,
    "ai_trading/data/fetch/__init__.py": 0,
    "ai_trading/strategy_allocator.py": 0,
}


def main() -> int:
    failures: list[str] = []
    for rel_path, max_allowed in MAX_DIRECT_ENV_TOUCHES.items():
        path = ROOT / rel_path
        if not path.exists():
            failures.append(f"{rel_path}: missing file")
            continue
        content = path.read_text(encoding="utf-8")
        touches = len(PATTERN.findall(content))
        print(f"{rel_path}: direct_env_touches={touches} max_allowed={max_allowed}")
        if touches > max_allowed:
            failures.append(
                f"{rel_path}: {touches} direct env touches exceeds max {max_allowed}"
            )

    if failures:
        print("RUNTIME_ENV_ACCESS_GUARD_FAILED")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("RUNTIME_ENV_ACCESS_GUARD_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
