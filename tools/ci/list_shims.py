# tools/ci/list_shims.py
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path("ai_trading")
PATTERNS = {
    "import_guard": re.compile(r"(?ms)try:\s*(?:\n\s*(?:from\s+[.\w]+\s+import\s+[^\n]+|import\s+[^\n]+))+?\n\s*except\s+ImportError\s*:\s*"),
    "config_dunder_getattr": re.compile(r"def\s+__getattr__\s*\("),
    "uppercase_alias_property": re.compile(r"@property\s+def\s+[A-Z0-9_]+\s*\("),
    "mock_class": re.compile(r"class\s+Mock[A-Za-z0-9_]+\s*:"),
    "raw_exec": re.compile(r"\bexec\s*\("),
    "raw_eval": re.compile(r"(^|[^.])\beval\s*\("),
}

def main():
    rows = []
    for p in ROOT.rglob("*.py"):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for name, rx in PATTERNS.items():
            for m in rx.finditer(text):
                ln = text.count("\n", 0, m.start()) + 1
                lines = text.splitlines()
                if ln <= len(lines):
                    snippet = lines[ln-1][:200]
                else:
                    snippet = ""
                rows.append((name, str(p), ln, snippet))

    rows.sort()
    out = Path("tools/ci/shims_targets.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["issue","file","line","snippet"])
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
