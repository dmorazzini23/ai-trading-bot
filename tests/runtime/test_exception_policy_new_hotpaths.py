from __future__ import annotations

import json
import subprocess
import sys


HOTPATHS = [
    "ai_trading/broker/adapters.py",
    "ai_trading/operator_presets.py",
    "ai_trading/operator_ui.py",
]


def test_no_broad_except_in_new_hotpaths() -> None:
    result = subprocess.run(
        [sys.executable, "tools/audit_exceptions.py", "--paths", *HOTPATHS],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout.splitlines()[0])
    by_file = data.get("by_file", {})
    offenders = {path: hits for path, hits in by_file.items() if hits}
    assert offenders == {}, f"broad except present in new hotpaths: {offenders}"
