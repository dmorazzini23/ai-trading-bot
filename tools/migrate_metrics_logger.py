from __future__ import annotations
import re
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PKG = 'ai_trading'
TARGET = f'{PKG}/telemetry/metrics_logger.py'

def patch_text(src: str) -> str:
    src = re.sub('(^|\\n)\\s*from\\s+metrics_logger\\s+import\\s+', '\\1from ai_trading.telemetry.metrics_logger import ', src, flags=re.MULTILINE)
    src = re.sub('(^|\\n)\\s*import\\s+metrics_logger(\\s*(?:as\\s+\\w+)?\\s*)$', '\\1from ai_trading.telemetry import metrics_logger\\2', src, flags=re.MULTILINE)
    src = re.sub('(\\nfrom\\s+ai_trading\\.alpaca_api\\s+import\\s+[^\\n]+)\\n\\s*except\\s+ImportError\\s*:\\s*\\n(\\s*#.*\\n)*(\\s*def\\s+alpaca_get[^\\n]*\\n(?:\\s.*\\n)*)?', '\\1\\n', src, flags=re.MULTILINE)
    return src

def main() -> int:
    changed = 0
    for p in ROOT.rglob('*.py'):
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith(('.git/', 'venv/', '.venv/', 'build/', 'dist/', 'site-packages/')):
            continue
        text = p.read_text(encoding='utf-8', errors='ignore')
        new = patch_text(text)
        if new != text:
            p.write_text(new, encoding='utf-8')
            changed += 1
    return 0
if __name__ == '__main__':
    sys.exit(main())