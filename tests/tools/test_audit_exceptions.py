from __future__ import annotations
# ruff: noqa

import json
import subprocess
import sys

def test_auditor_runs():
    p = subprocess.run([sys.executable, "tools/audit_exceptions.py", "--paths", "ai_trading"], capture_output=True, text=True)
    assert p.returncode == 0
    data = json.loads(p.stdout.splitlines()[0])
    assert "total" in data and isinstance(data["total"], int)

