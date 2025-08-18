from __future__ import annotations
# ruff: noqa

import json
import subprocess
import sys

HOTPATHS = [
    "ai_trading/core/bot_engine.py",
    "ai_trading/risk/circuit_breakers.py",
    "ai_trading/monitoring/performance_dashboard.py",
    "ai_trading/portfolio/sizing.py",
    "ai_trading/meta_learning.py",
]

def test_no_broad_except_in_hotpaths():
    p = subprocess.run([sys.executable, "tools/audit_exceptions.py", "--paths", *HOTPATHS], capture_output=True, text=True)
    assert p.returncode == 0
    data = json.loads(p.stdout.splitlines()[0])
    by_file = data.get("by_file", {})
    offenders = {f: hits for f, hits in by_file.items() if hits}
    assert offenders == {}, f"broad except present in: {list(offenders)}"

