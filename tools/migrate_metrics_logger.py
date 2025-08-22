# tools/migrate_metrics_logger.py
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root (…/ai_trading)
PKG = "ai_trading"
TARGET = f"{PKG}/telemetry/metrics_logger.py"

def patch_text(src: str) -> str:
    # 1) from metrics_logger import X → from ai_trading.telemetry.metrics_logger import X
    src = re.sub(
        r"(^|\n)\s*from\s+metrics_logger\s+import\s+",
        r"\1from ai_trading.telemetry.metrics_logger import ",
        src,
        flags=re.MULTILINE,
    )
    # 2) import metrics_logger → from ai_trading.telemetry import metrics_logger
    #    (avoid touching already-correct imports)
    src = re.sub(
        r"(^|\n)\s*import\s+metrics_logger(\s*(?:as\s+\w+)?\s*)$",
        r"\1from ai_trading.telemetry import metrics_logger\2",
        src,
        flags=re.MULTILINE,
    )
    # 3) Guard against a dangling 'except ImportError:' with no preceding try
    #    Only fix the common pattern around alpaca imports.
    src = re.sub(
        r"(\nfrom\s+ai_trading\.alpaca_api\s+import\s+[^\n]+)\n\s*except\s+ImportError\s*:\s*\n(\s*#.*\n)*(\s*def\s+alpaca_get[^\n]*\n(?:\s.*\n)*)?",
        r"\1\n",
        src,
        flags=re.MULTILINE,
    )
    return src

def main() -> int:
    changed = 0
    for p in ROOT.rglob("*.py"):
        # Skip generated/third-party
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith((".git/", "venv/", ".venv/", "build/", "dist/", "site-packages/")):
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        new = patch_text(text)
        if new != text:
            p.write_text(new, encoding="utf-8")
            changed += 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
