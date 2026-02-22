#!/usr/bin/env bash
set -euo pipefail
cd /home/aiuser/ai-trading-bot

python3 - <<'PY'
import re
src = ".env"
dst = ".env.runtime"
pat = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*=')
with open(src, "r", encoding="utf-8", errors="ignore") as f, open(dst, "w", encoding="utf-8") as g:
    for raw in f:
        line = raw.rstrip("\n")
        if not line or line.lstrip().startswith("#"):
            continue
        if pat.match(line):
            g.write(line + "\n")
PY

chmod 600 .env.runtime
