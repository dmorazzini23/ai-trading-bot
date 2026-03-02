#!/usr/bin/env bash
set -euo pipefail
cd /home/aiuser/ai-trading-bot

python3 - <<'PY'
import re
src = ".env"
dst = ".env.runtime"
pat = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*=')
deprecated_keys = {
    "AI_TRADING_EXEC_ALLOW_FALLBACK_WITHOUT_NBBO",
}
with open(src, "r", encoding="utf-8", errors="ignore") as f, open(dst, "w", encoding="utf-8") as g:
    for raw in f:
        line = raw.rstrip("\n")
        if not line or line.lstrip().startswith("#"):
            continue
        if pat.match(line):
            key = line.split("=", 1)[0].strip()
            if key in deprecated_keys:
                continue
            g.write(line + "\n")
PY

chmod 600 .env.runtime
